#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import optuna

from benchmark import parse_outcome


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

CONFIG_DIR = ROOT / "configs" / "trials_v4"
RESULTS_DIR = ROOT / "results"
WORKSPACE_DIR = RESULTS_DIR / "workspaces_v4"
FAILURE_DIR = RESULTS_DIR / "failures_v4"

CANDIDATE_WRAPPER = ROOT / "bots" / "_tuning_candidate.py"

for directory in (
    CONFIG_DIR,
    RESULTS_DIR,
    WORKSPACE_DIR,
    FAILURE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Optimisation settings
# ============================================================

STRONGEST_MATCHES = 8
MIXED_MATCHES = 4
MATCHES_PER_TRIAL = STRONGEST_MATCHES + MIXED_MATCHES

NUMBER_OF_TRIALS = 60
MATCH_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS_MULTIPLIER = 3

KEEP_SUCCESSFUL_WORKSPACES = False
PLAYER_ID = 0

DATABASE_PATH = RESULTS_DIR / "optuna_v4.db"
STUDY_NAME = "bot_tuning_v4"


# Candidate receives the trial config through a small wrapper. Opponents do not
# read CANDIDATE_CONFIG, so they remain fixed during optimisation.
STRONGEST_BOT_ARGUMENTS = [
    "1:bots/_tuning_candidate.py",
    "7:bots/other_bots2.py",
]

MIXED_BOT_ARGUMENTS = [
    "1:bots/_tuning_candidate.py",
    "2:bots/other_bot.py",
    "2:bots/other_bots.py",
    "3:bots/other_bots2.py",
]


# Original Trial 0000 / submitted baseline.
BASELINE_CONFIG: dict[str, float] = {
    "TURN_WEIGHT": 80.0,
    "FOOD_CLUSTER_WEIGHT": 0.55,
    "FOOD_DISTANCE_POWER": 1.15,
    "DANGER_OVERRIDE_RANGE_MULT": 9.0,
    "DANGER_DIRECT_RATIO": 1.06,
    "CHASE_RANGE_MULT": 7.0,
    "CHASE_LEAD_TICKS": 2.6,
    "CHASE_MIN_CLOSING_RATE": -1.25,
    "CHASE_CLOSE_WEIGHT": 24.0,
    "SPLIT_RANGE_SAFETY_MULT": 0.75,
    "SPLIT_TARGET_MIN_RADIUS_MULT": 0.16,
    "VIRUS_FARM_MAX_RADIUS": 18.0,
}


# Best configuration from the first Optuna run.
TRIAL_17_CONFIG: dict[str, float] = {
    "TURN_WEIGHT": 56.519876630868445,
    "FOOD_CLUSTER_WEIGHT": 0.55,
    "FOOD_DISTANCE_POWER": 1.15,
    "DANGER_OVERRIDE_RANGE_MULT": 9.0,
    "DANGER_DIRECT_RATIO": 1.06,
    "CHASE_RANGE_MULT": 6.219934902522074,
    "CHASE_LEAD_TICKS": 2.924711670693439,
    "CHASE_MIN_CLOSING_RATE": -1.25,
    "CHASE_CLOSE_WEIGHT": 39.90857652982805,
    "SPLIT_RANGE_SAFETY_MULT": 0.7631787944167937,
    "SPLIT_TARGET_MIN_RADIUS_MULT": 0.16,
    "VIRUS_FARM_MAX_RADIUS": 21.559608057655243,
}


# ============================================================
# Candidate wrapper
# ============================================================

def ensure_candidate_wrapper() -> None:
    """Create the wrapper that applies BOT_CONFIG only to player 0."""
    wrapper_source = '''import os

config_path = os.environ.get("CANDIDATE_CONFIG")
if not config_path:
    raise RuntimeError("CANDIDATE_CONFIG was not provided")

os.environ["BOT_CONFIG"] = config_path

from my_bot import main

if __name__ == "__main__":
    main()
'''

    CANDIDATE_WRAPPER.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_WRAPPER.write_text(
        wrapper_source,
        encoding="utf-8",
    )


# ============================================================
# Process execution
# ============================================================

def kill_process_group(process: subprocess.Popen[str]) -> None:
    """Kill the simulation launcher and any children it left running."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


def run_simulation(
    command: list[str],
    environment: dict[str, str],
) -> tuple[int, str]:
    """Run one match and clean up every process created for it."""
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    timed_out = False

    try:
        stdout, stderr = process.communicate(
            timeout=MATCH_TIMEOUT_SECONDS,
        )

    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_group(process)
        stdout, stderr = process.communicate()

    finally:
        # Some launcher versions intentionally leave a process alive for an end
        # screen, even in headless mode.
        kill_process_group(process)

    combined_output = (
        (stdout or "")
        + "\n"
        + (stderr or "")
    )

    if timed_out:
        raise RuntimeError(
            f"Simulation timed out after "
            f"{MATCH_TIMEOUT_SECONDS} seconds.\n"
            f"{combined_output[-5000:]}"
        )

    return process.returncode, combined_output


# ============================================================
# Result parsing
# ============================================================

def extract_result_from_mapping(
    result: dict[str, Any],
    player_id: int,
) -> tuple[float, int] | None:
    ranking = result.get("ranking")
    masses = result.get("final_masses")

    if ranking is None or masses is None:
        return None

    result_type = result.get("result_type")
    if result_type is not None and str(result_type) != "SUCCESS":
        raise ValueError(
            f"Match result was {result_type!r}, not SUCCESS."
        )

    ranking_as_ints = [int(value) for value in ranking]

    if player_id not in ranking_as_ints:
        raise ValueError(
            f"Player {player_id} missing from ranking."
        )

    mass_value = None
    if isinstance(masses, dict):
        mass_value = masses.get(player_id)
        if mass_value is None:
            mass_value = masses.get(str(player_id))

    if mass_value is None:
        raise ValueError(
            f"Player {player_id} missing from final_masses."
        )

    rank = ranking_as_ints.index(player_id) + 1
    return float(mass_value), rank


def extract_result_from_payload(
    payload: Any,
    player_id: int,
) -> tuple[float, int] | None:
    if isinstance(payload, list):
        for item in reversed(payload):
            parsed = extract_result_from_payload(
                item,
                player_id,
            )
            if parsed is not None:
                return parsed
        return None

    if not isinstance(payload, dict):
        return None

    direct = extract_result_from_mapping(
        payload,
        player_id,
    )
    if direct is not None:
        return direct

    for key in (
        "result",
        "match_result",
        "data",
        "payload",
        "output",
    ):
        nested = payload.get(key)
        if nested is None:
            continue

        parsed = extract_result_from_payload(
            nested,
            player_id,
        )
        if parsed is not None:
            return parsed

    return None


def find_result_files(workspace: Path) -> list[Path]:
    filenames = {
        "results.json",
        "result.json",
        "match_result.json",
        "match-results.json",
    }

    found = [
        path
        for path in workspace.rglob("*.json")
        if path.name in filenames
    ]

    return sorted(
        found,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def parse_workspace_result(
    workspace: Path,
    output: str,
    player_id: int,
) -> tuple[float, int]:
    """Prefer a result JSON file and fall back to terminal output."""
    deadline = time.monotonic() + 2.0

    while time.monotonic() < deadline:
        result_files = find_result_files(workspace)
        if result_files:
            break
        time.sleep(0.1)
    else:
        result_files = []

    errors: list[str] = []

    for result_path in result_files:
        try:
            payload = json.loads(
                result_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            parsed = extract_result_from_payload(
                payload,
                player_id,
            )
            if parsed is not None:
                return parsed

        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(
                f"{result_path}: {error}"
            )

    try:
        return parse_outcome(output, player_id)

    except ValueError as stdout_error:
        details = "\n".join(errors)

        raise ValueError(
            "No valid completed-match result was found.\n"
            f"JSON errors:\n{details or 'None'}\n"
            f"Stdout parser error: {stdout_error}"
        ) from stdout_error


# ============================================================
# Timeout / ban detection
# ============================================================

def cumulative_timeout_players(output: str) -> set[int]:
    """Return player IDs mentioned on cumulative-timeout/ban lines."""
    players: set[int] = set()

    for line in output.splitlines():
        upper = line.upper()

        if (
            "CUMULATIVE_TIMEOUT" not in upper
            and "PLAYER_BANNED" not in upper
        ):
            continue

        match = re.search(
            r"player(?:_id)?\s*[:=]\s*(\d+)",
            line,
            flags=re.IGNORECASE,
        )

        if match:
            players.add(int(match.group(1)))

    return players


class OpponentTimeoutError(RuntimeError):
    pass


class CandidateTimeoutError(RuntimeError):
    pass


# ============================================================
# Failure diagnostics
# ============================================================

def save_failure_log(
    trial_name: str,
    lineup_name: str,
    match_number: int,
    attempt_number: int,
    command: list[str],
    workspace: Path,
    output: str,
    error: Exception,
) -> Path:
    log_path = (
        FAILURE_DIR
        / (
            f"{trial_name}_{lineup_name}"
            f"_match_{match_number:03d}"
            f"_attempt_{attempt_number:03d}.log"
        )
    )

    log_path.write_text(
        "\n".join([
            f"Error: {error}",
            "",
            f"Workspace: {workspace}",
            "",
            "Command:",
            " ".join(command),
            "",
            "Simulation output:",
            output,
        ]),
        encoding="utf-8",
        errors="replace",
    )

    return log_path


# ============================================================
# Fitness
# ============================================================

def summarise_results(
    masses: list[float],
    ranks: list[int],
    failures: int,
    total_matches: int,
) -> dict[str, float]:
    if not ranks:
        return {
            "fitness": -999.0,
            "average_rank": 8.0,
            "median_mass": 0.0,
            "average_mass": 0.0,
            "win_rate": 0.0,
            "top_three_rate": 0.0,
            "failure_rate": 1.0,
        }

    successful = len(ranks)

    average_rank = statistics.fmean(ranks)
    median_mass = statistics.median(masses)
    average_mass = statistics.fmean(masses)

    win_rate = (
        sum(rank == 1 for rank in ranks)
        / successful
    )

    top_three_rate = (
        sum(rank <= 3 for rank in ranks)
        / successful
    )

    failure_rate = failures / total_matches

    rank_score = (8.0 - average_rank) / 7.0

    mass_score = min(
        math.log1p(median_mass)
        / math.log1p(50.0),
        1.5,
    )

    fitness = (
        0.90 * average_mass
        + 0.10 * median_mass
    )

    return {
        "fitness": float(fitness),
        "average_rank": float(average_rank),
        "median_mass": float(median_mass),
        "average_mass": float(average_mass),
        "win_rate": float(win_rate),
        "top_three_rate": float(top_three_rate),
        "failure_rate": float(failure_rate),
    }


# ============================================================
# Candidate evaluation
# ============================================================

def evaluate_lineup(
    config_path: Path,
    bot_arguments: list[str],
    match_count: int,
    lineup_name: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    masses: list[float] = []
    ranks: list[int] = []
    failures = 0

    trial_name = config_path.stem
    completed_slots = 0
    attempt_number = 0
    max_attempts = max(
        match_count,
        match_count * MAX_ATTEMPTS_MULTIPLIER,
    )

    while (
        completed_slots < match_count
        and attempt_number < max_attempts
    ):
        attempt_number += 1
        match_number = completed_slots + 1

        workspace = (
            WORKSPACE_DIR
            / trial_name
            / lineup_name
            / (
                f"match_{match_number:03d}"
                f"_attempt_{attempt_number:03d}"
            )
        )

        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=True)

        command = [
            "uv",
            "run",
            "simulation",
            "--headless",
            "--workspace",
            str(workspace.resolve()),
            *bot_arguments,
        ]

        output = ""

        try:
            return_code, output = run_simulation(
                command,
                environment,
            )

            timed_out_players = cumulative_timeout_players(
                output
            )

            if PLAYER_ID in timed_out_players:
                raise CandidateTimeoutError(
                    "Candidate player 0 received "
                    "CUMULATIVE_TIMEOUT."
                )

            opponent_timeouts = (
                timed_out_players - {PLAYER_ID}
            )
            if opponent_timeouts:
                raise OpponentTimeoutError(
                    "Opponent cumulative timeout: "
                    f"{sorted(opponent_timeouts)}"
                )

            mass, rank = parse_workspace_result(
                workspace=workspace,
                output=output,
                player_id=PLAYER_ID,
            )

            if return_code != 0:
                raise RuntimeError(
                    f"Simulation returned exit code "
                    f"{return_code}, despite producing a result."
                )

            masses.append(mass)
            ranks.append(rank)
            completed_slots += 1

            print(
                f"    [{lineup_name}] "
                f"Match {completed_slots}/{match_count}: "
                f"rank={rank}, mass={mass:.2f}"
            )

            if not KEEP_SUCCESSFUL_WORKSPACES:
                shutil.rmtree(
                    workspace,
                    ignore_errors=True,
                )

        except OpponentTimeoutError as error:
            # An opponent failure says nothing useful about the candidate.
            # Retry the same match slot without penalising its fitness.
            log_path = save_failure_log(
                trial_name=trial_name,
                lineup_name=lineup_name,
                match_number=match_number,
                attempt_number=attempt_number,
                command=command,
                workspace=workspace,
                output=output,
                error=error,
            )

            print(
                f"    [{lineup_name}] "
                f"Match {match_number}/{match_count}: "
                f"RETRY ({error})"
            )
            print(f"      Log: {log_path}")

        except Exception as error:
            # Candidate timeout, crash, parser failure, or a simulator failure:
            # count this as one failed evaluation slot.
            failures += 1
            completed_slots += 1

            log_path = save_failure_log(
                trial_name=trial_name,
                lineup_name=lineup_name,
                match_number=match_number,
                attempt_number=attempt_number,
                command=command,
                workspace=workspace,
                output=output,
                error=error,
            )

            print(
                f"    [{lineup_name}] "
                f"Match {completed_slots}/{match_count}: "
                f"FAILED: {error}"
            )
            print(f"      Failure log: {log_path}")

    # Avoid silently giving a candidate fewer evaluated matches when repeated
    # opponent failures exhaust the retry budget.
    missing_slots = match_count - completed_slots
    if missing_slots > 0:
        failures += missing_slots
        completed_slots += missing_slots

        print(
            f"    [{lineup_name}] Retry limit reached; "
            f"counting {missing_slots} missing match(es) "
            "as failures."
        )

    summary = summarise_results(
        masses=masses,
        ranks=ranks,
        failures=failures,
        total_matches=match_count,
    )

    return {
        **summary,
        "masses": masses,
        "ranks": ranks,
        "failures": failures,
    }


def evaluate(config_path: Path) -> dict[str, float]:
    environment = os.environ.copy()

    # Never let opponents inherit a BOT_CONFIG from the shell.
    environment.pop("BOT_CONFIG", None)

    # Only _tuning_candidate.py reads this variable and converts it to
    # BOT_CONFIG inside player 0's process.
    environment["CANDIDATE_CONFIG"] = str(
        config_path.resolve()
    )

    # Eight NumPy bots must not each create a pool of BLAS worker threads.
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    })

    strongest = evaluate_lineup(
        config_path=config_path,
        bot_arguments=STRONGEST_BOT_ARGUMENTS,
        match_count=STRONGEST_MATCHES,
        lineup_name="strongest",
        environment=environment,
    )

    mixed = evaluate_lineup(
        config_path=config_path,
        bot_arguments=MIXED_BOT_ARGUMENTS,
        match_count=MIXED_MATCHES,
        lineup_name="mixed",
        environment=environment,
    )

    overall_masses = (
        strongest["masses"]
        + mixed["masses"]
    )
    overall_ranks = (
        strongest["ranks"]
        + mixed["ranks"]
    )
    overall_failures = (
        strongest["failures"]
        + mixed["failures"]
    )

    overall = summarise_results(
        masses=overall_masses,
        ranks=overall_ranks,
        failures=overall_failures,
        total_matches=MATCHES_PER_TRIAL,
    )

    combined_fitness = (
        0.70 * strongest["fitness"]
        + 0.30 * mixed["fitness"]
    )

    return {
        "fitness": float(combined_fitness),

        "average_rank": overall["average_rank"],
        "median_mass": overall["median_mass"],
        "average_mass": overall["average_mass"],
        "win_rate": overall["win_rate"],
        "top_three_rate": overall["top_three_rate"],
        "failure_rate": overall["failure_rate"],

        "strongest_fitness": strongest["fitness"],
        "strongest_average_rank": strongest["average_rank"],
        "strongest_median_mass": strongest["median_mass"],
        "strongest_win_rate": strongest["win_rate"],
        "strongest_top_three_rate": strongest["top_three_rate"],
        "strongest_failure_rate": strongest["failure_rate"],

        "mixed_fitness": mixed["fitness"],
        "mixed_average_rank": mixed["average_rank"],
        "mixed_median_mass": mixed["median_mass"],
        "mixed_win_rate": mixed["win_rate"],
        "mixed_top_three_rate": mixed["top_three_rate"],
        "mixed_failure_rate": mixed["failure_rate"],
    }


# ============================================================
# Optuna objective
# ============================================================

def objective(trial: optuna.Trial) -> float:
    config = {
        "TURN_WEIGHT": trial.suggest_float(
            "TURN_WEIGHT",
            20.0,
            130.0,
        ),
        "FOOD_CLUSTER_WEIGHT": trial.suggest_float(
            "FOOD_CLUSTER_WEIGHT",
            0.15,
            1.10,
        ),
        "FOOD_DISTANCE_POWER": trial.suggest_float(
            "FOOD_DISTANCE_POWER",
            0.85,
            1.60,
        ),
        "DANGER_OVERRIDE_RANGE_MULT": trial.suggest_float(
            "DANGER_OVERRIDE_RANGE_MULT",
            5.5,
            11.0,
        ),
        "DANGER_DIRECT_RATIO": trial.suggest_float(
            "DANGER_DIRECT_RATIO",
            1.02,
            1.15,
        ),
        "CHASE_RANGE_MULT": trial.suggest_float(
            "CHASE_RANGE_MULT",
            3.5,
            8.0,
        ),
        "CHASE_LEAD_TICKS": trial.suggest_float(
            "CHASE_LEAD_TICKS",
            1.2,
            4.2,
        ),
        "CHASE_MIN_CLOSING_RATE": trial.suggest_float(
            "CHASE_MIN_CLOSING_RATE",
            -2.5,
            0.0,
        ),
        "CHASE_CLOSE_WEIGHT": trial.suggest_float(
            "CHASE_CLOSE_WEIGHT",
            10.0,
            60.0,
        ),
        "SPLIT_RANGE_SAFETY_MULT": trial.suggest_float(
            "SPLIT_RANGE_SAFETY_MULT",
            0.65,
            0.86,
        ),
        "SPLIT_TARGET_MIN_RADIUS_MULT": trial.suggest_float(
            "SPLIT_TARGET_MIN_RADIUS_MULT",
            0.08,
            0.30,
        ),
        "VIRUS_FARM_MAX_RADIUS": trial.suggest_float(
            "VIRUS_FARM_MAX_RADIUS",
            12.0,
            24.0,
        ),
    }

    config_path = (
        CONFIG_DIR
        / f"trial_{trial.number:04d}.json"
    )

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"\nTrial {trial.number}")
    print(json.dumps(config, indent=2))

    results = evaluate(config_path)

    for name, value in results.items():
        if name != "fitness":
            trial.set_user_attr(name, value)

    print(
        f"  Fitness={results['fitness']:.4f}, "
        f"average rank={results['average_rank']:.2f}, "
        f"median mass={results['median_mass']:.2f}, "
        f"failures={results['failure_rate']:.1%}"
    )

    print(
        f"    Strongest: "
        f"fitness={results['strongest_fitness']:.4f}, "
        f"avg rank={results['strongest_average_rank']:.2f}"
    )

    print(
        f"    Mixed:     "
        f"fitness={results['mixed_fitness']:.4f}, "
        f"avg rank={results['mixed_average_rank']:.2f}"
    )

    return results["fitness"]


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_candidate_wrapper()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{DATABASE_PATH}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=31,
        ),
    )

    # Queue known reference configurations only for a fresh study.
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_CONFIG)
        study.enqueue_trial(TRIAL_17_CONFIG)

    remaining_trials = max(
        0,
        NUMBER_OF_TRIALS - len(study.trials),
    )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
        )
    else:
        print(
            f"Study already contains "
            f"{len(study.trials)} trials; "
            "nothing left to run."
        )

    best_trial = study.best_trial

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"Trial:   {best_trial.number}")
    print(f"Fitness: {best_trial.value:.4f}")
    print(json.dumps(best_trial.params, indent=2))

    print("\nValidation reminder:")
    print(
        "Benchmark the top 3 candidates over a much larger "
        "number of games before choosing a final bot."
    )


if __name__ == "__main__":
    main()