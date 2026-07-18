#!/usr/bin/env python3
"""Resumable multi-fidelity optimiser for clean_bot.py.

Primary objective: expected final mass. Failed matches contribute zero mass.
Rank, median, win rate, and timeouts are recorded only as diagnostics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna


ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RESULT_TYPE_RE = re.compile(r"result_type=['\"]([^'\"]+)['\"]")
RANKING_RE = re.compile(r"ranking=(\[[^\]]*\])")
MASSES_RE = re.compile(r"final_masses=(\{[^}]*\})")

ORDER_PROFILES = {
    "escape_first": ["escape", "unstuck", "split", "chase", "virus", "food"],
    "split_first": ["split", "escape", "unstuck", "chase", "virus", "food"],
    "escape_then_split": ["escape", "split", "unstuck", "chase", "virus", "food"],
    "virus_before_chase": ["escape", "unstuck", "split", "virus", "chase", "food"],
}


@dataclass
class MatchSet:
    masses: list[float] = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)
    failures: int = 0
    elapsed_seconds: float = 0.0

    @property
    def scheduled(self) -> int:
        return len(self.masses) + self.failures

    def extend(self, other: "MatchSet") -> None:
        self.masses.extend(other.masses)
        self.ranks.extend(other.ranks)
        self.failures += other.failures
        self.elapsed_seconds += other.elapsed_seconds


def parse_outcome(output: str, player_id: int) -> tuple[float, int]:
    clean = ANSI_ESCAPE.sub("", output)
    lines = [
        line for line in clean.splitlines()
        if "match complete" in line and "ranking=" in line and "final_masses=" in line
    ]
    if not lines:
        raise ValueError("No completed-match outcome line found")
    line = lines[-1]
    result_match = RESULT_TYPE_RE.search(line)
    if result_match and result_match.group(1) != "SUCCESS":
        raise ValueError(f"Result was {result_match.group(1)!r}, not SUCCESS")
    ranking_match = RANKING_RE.search(line)
    masses_match = MASSES_RE.search(line)
    if ranking_match is None or masses_match is None:
        raise ValueError("Could not parse ranking or final_masses")
    ranking = [int(value) for value in ast.literal_eval(ranking_match.group(1))]
    masses = ast.literal_eval(masses_match.group(1))
    mass = masses.get(player_id, masses.get(str(player_id)))
    if player_id not in ranking or mass is None:
        raise ValueError(f"Player {player_id} missing from result")
    return float(mass), ranking.index(player_id) + 1


def extract_result(payload: Any, player_id: int) -> tuple[float, int] | None:
    if isinstance(payload, list):
        for item in reversed(payload):
            parsed = extract_result(item, player_id)
            if parsed is not None:
                return parsed
        return None
    if not isinstance(payload, dict):
        return None
    ranking = payload.get("ranking")
    masses = payload.get("final_masses")
    if ranking is not None and isinstance(masses, dict):
        ranking = [int(value) for value in ranking]
        mass = masses.get(player_id, masses.get(str(player_id)))
        if player_id in ranking and mass is not None:
            result_type = payload.get("result_type")
            if result_type is not None and str(result_type) != "SUCCESS":
                raise ValueError(f"Result was {result_type!r}, not SUCCESS")
            return float(mass), ranking.index(player_id) + 1
    for key in ("result", "match_result", "data", "payload", "output"):
        if key in payload:
            parsed = extract_result(payload[key], player_id)
            if parsed is not None:
                return parsed
    return None


def parse_workspace_result(workspace: Path, output: str, player_id: int) -> tuple[float, int]:
    # The normal simulator path prints the complete result to stdout. Parse it
    # immediately so successful matches do not spend two seconds polling the
    # workspace. JSON is a fallback for engine variants without that line.
    try:
        return parse_outcome(output, player_id)
    except ValueError:
        pass

    names = {"results.json", "result.json", "match_result.json", "match-results.json"}
    deadline = time.monotonic() + 2.0
    files: list[Path] = []
    while time.monotonic() < deadline:
        files = sorted(
            (path for path in workspace.rglob("*.json") if path.name in names),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if files:
            break
        time.sleep(0.1)
    for path in files:
        try:
            parsed = extract_result(json.loads(path.read_text(encoding="utf-8")), player_id)
            if parsed is not None:
                return parsed
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    raise ValueError("No completed-match outcome found in stdout or workspace")


def timed_out_players(output: str) -> set[int]:
    players: set[int] = set()
    for line in output.splitlines():
        upper = line.upper()
        if "CUMULATIVE_TIMEOUT" not in upper and "PLAYER_BANNED" not in upper:
            continue
        match = re.search(r"player(?:_id)?\s*[:=]\s*(\d+)", line, re.I)
        if match:
            players.add(int(match.group(1)))
    return players


def kill_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


def run_process(command: list[str], cwd: Path, environment: dict[str, str], timeout: float) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_group(process)
        stdout, stderr = process.communicate()
        raise RuntimeError(f"Simulation exceeded {timeout:.0f}s\n{stdout[-3000:]}\n{stderr[-3000:]}")
    finally:
        kill_group(process)
    return process.returncode, (stdout or "") + "\n" + (stderr or "")


def expected_mass(results: MatchSet) -> float:
    """Failures are zero-mass games, matching the actual tournament outcome."""
    return sum(results.masses) / max(results.scheduled, 1)


def summary(results: MatchSet) -> dict[str, float]:
    scheduled = max(results.scheduled, 1)
    successes = len(results.masses)
    ranks = results.ranks
    return {
        "expected_mass": expected_mass(results),
        "successful_average_mass": statistics.fmean(results.masses) if results.masses else 0.0,
        "median_mass": statistics.median(results.masses) if results.masses else 0.0,
        "maximum_mass": max(results.masses, default=0.0),
        "average_rank": statistics.fmean(ranks) if ranks else 8.0,
        "win_rate": sum(rank == 1 for rank in ranks) / scheduled,
        "top_three_rate": sum(rank <= 3 for rank in ranks) / scheduled,
        "failure_rate": results.failures / scheduled,
        "matches": float(results.scheduled),
        "elapsed_seconds": results.elapsed_seconds,
    }


def combined_fitness(strongest: MatchSet, mixed: MatchSet) -> float:
    # Average final mass is the objective. The lineup weighting is explicit so
    # changes to match counts do not silently change what the study optimises.
    return 0.75 * expected_mass(strongest) + 0.25 * expected_mass(mixed)


def suggest_config(trial: optuna.Trial) -> dict[str, Any]:
    order_profile = trial.suggest_categorical("ORDER_PROFILE", list(ORDER_PROFILES))
    return {
        "OVERRIDE_ORDER": ORDER_PROFILES[order_profile],
        "NUM_DIRECTIONS": trial.suggest_categorical("NUM_DIRECTIONS", [12, 16]),
        "MAX_ENEMIES": trial.suggest_categorical("MAX_ENEMIES", [18, 21, 24]),
        "SAFETY_OWN_PIECES": trial.suggest_categorical("SAFETY_OWN_PIECES", [2, 3]),
        "STEP_RADIUS_MULT": trial.suggest_float("STEP_RADIUS_MULT", 1.25, 1.75),

        "SPLIT_REACH_RELIABILITY": trial.suggest_float("SPLIT_REACH_RELIABILITY", 0.62, 0.90),
        "ESCAPE_DIRECT_RANGE_MULT": trial.suggest_float("ESCAPE_DIRECT_RANGE_MULT", 3.5, 8.0),
        "ESCAPE_SPLIT_REACH_BUFFER": trial.suggest_float("ESCAPE_SPLIT_REACH_BUFFER", 0.92, 1.16),
        "ESCAPE_MOVE_WEIGHT": trial.suggest_float("ESCAPE_MOVE_WEIGHT", 80.0, 320.0, log=True),
        "ESCAPE_WALL_WEIGHT": trial.suggest_float("ESCAPE_WALL_WEIGHT", 100.0, 650.0, log=True),
        "ESCAPE_VIRUS_SHIELD_WEIGHT": trial.suggest_float("ESCAPE_VIRUS_SHIELD_WEIGHT", 150.0, 1400.0, log=True),
        "ESCAPE_HARD_MARGIN": trial.suggest_float("ESCAPE_HARD_MARGIN", 0.0, 0.40),

        "SPLIT_CAPTURE_WEIGHT": trial.suggest_float("SPLIT_CAPTURE_WEIGHT", 3.0, 15.0, log=True),
        "SPLIT_DEPTH_COST": trial.suggest_float("SPLIT_DEPTH_COST", 0.5, 7.0, log=True),
        "SPLIT_FRAGMENT_COST": trial.suggest_float("SPLIT_FRAGMENT_COST", 0.02, 0.65, log=True),
        "SPLIT_EXTERNAL_RISK_WEIGHT": trial.suggest_float("SPLIT_EXTERNAL_RISK_WEIGHT", 0.15, 3.5, log=True),
        "SPLIT_MIN_UTILITY": trial.suggest_float("SPLIT_MIN_UTILITY", -2.0, 10.0),
        "SPLIT_LARGER_OWNER_RATIO": trial.suggest_float("SPLIT_LARGER_OWNER_RATIO", 0.95, 1.30),
        "SPLIT_PLAN_WAIT_TICKS": trial.suggest_int("SPLIT_PLAN_WAIT_TICKS", 1, 9),

        "CHASE_ACQUIRE_RANGE_MULT": trial.suggest_float("CHASE_ACQUIRE_RANGE_MULT", 4.0, 14.0),
        "CHASE_DIRECT_RANGE_MULT": trial.suggest_float("CHASE_DIRECT_RANGE_MULT", 1.5, 5.0),
        "CHASE_MAX_INTERCEPT_TICKS": trial.suggest_float("CHASE_MAX_INTERCEPT_TICKS", 5.0, 18.0),
        "CHASE_LOCK_TICKS": trial.suggest_int("CHASE_LOCK_TICKS", 30, 240, log=True),
        "CHASE_WALL_HORIZON": trial.suggest_float("CHASE_WALL_HORIZON", 14.0, 42.0),
        "CHASE_BLOCK_OFFSET_MULT": trial.suggest_float("CHASE_BLOCK_OFFSET_MULT", 0.25, 1.50),
        "CHASE_DISTANCE_WEIGHT": trial.suggest_float("CHASE_DISTANCE_WEIGHT", 0.05, 1.0, log=True),
        "CHASE_ETA_WEIGHT": trial.suggest_float("CHASE_ETA_WEIGHT", 0.25, 6.0, log=True),
        "CHASE_LOCK_BONUS": trial.suggest_float("CHASE_LOCK_BONUS", 4.0, 50.0, log=True),
        "CHASE_VELOCITY_NEW_WEIGHT": trial.suggest_float("CHASE_VELOCITY_NEW_WEIGHT", 0.20, 0.85),

        "VIRUS_ENEMY_RANGE_MULT": trial.suggest_float("VIRUS_ENEMY_RANGE_MULT", 2.0, 9.0),
        "VIRUS_MANUAL_SPLIT_TIME_BONUS": trial.suggest_float("VIRUS_MANUAL_SPLIT_TIME_BONUS", 0.5, 8.0, log=True),
        "VIRUS_MANUAL_SPLIT_RISK_COST": trial.suggest_float("VIRUS_MANUAL_SPLIT_RISK_COST", 0.25, 8.0, log=True),
        "VIRUS_MAX_FARM_MASS": trial.suggest_float("VIRUS_MAX_FARM_MASS", 45.0, 350.0, log=True),

        "FOOD_CLUSTER_RADIUS": trial.suggest_float("FOOD_CLUSTER_RADIUS", 2.5, 7.0),
        "FOOD_CLUSTER_WEIGHT": trial.suggest_float("FOOD_CLUSTER_WEIGHT", 0.25, 3.0, log=True),
        "FOOD_DISTANCE_POWER": trial.suggest_float("FOOD_DISTANCE_POWER", 0.75, 2.25),
        "FOOD_LOCK_TICKS": trial.suggest_int("FOOD_LOCK_TICKS", 3, 30),
        "ROAM_CENTER_WEIGHT": trial.suggest_float("ROAM_CENTER_WEIGHT", 0.15, 3.0, log=True),
        "ROAM_MOMENTUM_WEIGHT": trial.suggest_float("ROAM_MOMENTUM_WEIGHT", 0.05, 1.5, log=True),
    }


DEFAULT_TRIAL = {
    "ORDER_PROFILE": "escape_first",
    "NUM_DIRECTIONS": 16,
    "MAX_ENEMIES": 24,
    "SAFETY_OWN_PIECES": 3,
    "STEP_RADIUS_MULT": 1.50,
    "SPLIT_REACH_RELIABILITY": 0.76,
    "ESCAPE_DIRECT_RANGE_MULT": 6.0,
    "ESCAPE_SPLIT_REACH_BUFFER": 1.05,
    "ESCAPE_MOVE_WEIGHT": 180.0,
    "ESCAPE_WALL_WEIGHT": 300.0,
    "ESCAPE_VIRUS_SHIELD_WEIGHT": 650.0,
    "ESCAPE_HARD_MARGIN": 0.15,
    "SPLIT_CAPTURE_WEIGHT": 8.0,
    "SPLIT_DEPTH_COST": 2.5,
    "SPLIT_FRAGMENT_COST": 0.18,
    "SPLIT_EXTERNAL_RISK_WEIGHT": 1.25,
    "SPLIT_MIN_UTILITY": 2.0,
    "SPLIT_LARGER_OWNER_RATIO": 1.02,
    "SPLIT_PLAN_WAIT_TICKS": 5,
    "CHASE_ACQUIRE_RANGE_MULT": 8.0,
    "CHASE_DIRECT_RANGE_MULT": 3.0,
    "CHASE_MAX_INTERCEPT_TICKS": 12.0,
    "CHASE_LOCK_TICKS": 160,
    "CHASE_WALL_HORIZON": 30.0,
    "CHASE_BLOCK_OFFSET_MULT": 0.8,
    "CHASE_DISTANCE_WEIGHT": 0.30,
    "CHASE_ETA_WEIGHT": 2.0,
    "CHASE_LOCK_BONUS": 18.0,
    "CHASE_VELOCITY_NEW_WEIGHT": 0.50,
    "VIRUS_ENEMY_RANGE_MULT": 6.0,
    "VIRUS_MANUAL_SPLIT_TIME_BONUS": 3.0,
    "VIRUS_MANUAL_SPLIT_RISK_COST": 2.0,
    "VIRUS_MAX_FARM_MASS": 180.0,
    "FOOD_CLUSTER_RADIUS": 4.5,
    "FOOD_CLUSTER_WEIGHT": 1.2,
    "FOOD_DISTANCE_POWER": 1.45,
    "FOOD_LOCK_TICKS": 14,
    "ROAM_CENTER_WEIGHT": 1.0,
    "ROAM_MOMENTUM_WEIGHT": 0.35,
}


class Optimizer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.root.resolve()
        self.results = self.root / args.results_dir
        self.configs = self.results / "configs"
        self.workspaces = self.results / "workspaces"
        self.failures = self.results / "failures"
        for path in (self.results, self.configs, self.workspaces, self.failures):
            path.mkdir(parents=True, exist_ok=True)
        self.wrapper = self.root / "bots" / "_tuning_candidate.py"
        self.environment = os.environ.copy()
        self.environment.pop("BOT_CONFIG", None)
        self.environment.pop("BOT_TUNING_PATH", None)
        self.environment.update({
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        })
        self.ensure_wrapper()

    def ensure_wrapper(self) -> None:
        self.wrapper.parent.mkdir(parents=True, exist_ok=True)
        source = (
            "import os\n"
            "path = os.environ.get('CANDIDATE_CONFIG')\n"
            "if not path:\n    raise RuntimeError('CANDIDATE_CONFIG missing')\n"
            "os.environ['BOT_TUNING_PATH'] = path\n"
            f"from {self.args.candidate_module} import main\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        self.wrapper.write_text(source, encoding="utf-8")

    def save_failure(self, label: str, command: list[str], workspace: Path, output: str, error: Exception) -> None:
        path = self.failures / f"{label}.log"
        path.write_text(
            f"Error: {error}\nWorkspace: {workspace}\nCommand: {' '.join(command)}\n\n{output}",
            encoding="utf-8",
            errors="replace",
        )

    def evaluate_lineup(self, config_path: Path, lineup: str, count: int, label: str) -> MatchSet:
        result = MatchSet()
        if count <= 0:
            return result
        bot_arguments = self.args.strongest if lineup == "strongest" else self.args.mixed
        attempts = 0
        slot = 0
        max_attempts = count * self.args.max_attempt_multiplier
        while slot < count and attempts < max_attempts:
            attempts += 1
            workspace = self.workspaces / label / lineup / f"slot_{slot + 1:03d}_attempt_{attempts:03d}"
            shutil.rmtree(workspace, ignore_errors=True)
            workspace.mkdir(parents=True, exist_ok=True)
            command = [
                "uv", "run", "simulation", "--headless", "--workspace",
                str(workspace.resolve()), *bot_arguments,
            ]
            environment = self.environment.copy()
            environment["CANDIDATE_CONFIG"] = str(config_path.resolve())
            output = ""
            started = time.perf_counter()
            try:
                return_code, output = run_process(
                    command, self.root, environment, self.args.match_timeout
                )
                timeout_ids = timed_out_players(output)
                if self.args.player in timeout_ids:
                    raise RuntimeError("Candidate cumulative timeout")
                opponent_timeouts = timeout_ids - {self.args.player}
                if opponent_timeouts:
                    # Retry opponent failures without consuming a candidate slot.
                    raise ChildProcessError(f"Opponent timeout: {sorted(opponent_timeouts)}")
                mass, rank = parse_workspace_result(workspace, output, self.args.player)
                if return_code != 0:
                    raise RuntimeError(f"Simulation exited {return_code}")
                result.masses.append(mass)
                result.ranks.append(rank)
                slot += 1
                print(f"    {lineup} {slot}/{count}: mass={mass:.2f}, rank={rank}")
                if not self.args.keep_workspaces:
                    shutil.rmtree(workspace, ignore_errors=True)
            except ChildProcessError as error:
                self.save_failure(f"{label}_{lineup}_{attempts:04d}_retry", command, workspace, output, error)
                print(f"    {lineup} retry: {error}")
            except Exception as error:
                result.failures += 1
                slot += 1
                self.save_failure(f"{label}_{lineup}_{attempts:04d}", command, workspace, output, error)
                print(f"    {lineup} {slot}/{count}: FAILED ({error})")
            result.elapsed_seconds += time.perf_counter() - started
        if slot < count:
            result.failures += count - slot
        return result

    def write_config(self, label: str, config: dict[str, Any]) -> Path:
        path = self.configs / f"{label}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path

    def evaluate_stage(self, config_path: Path, label: str, strongest_count: int, mixed_count: int) -> tuple[MatchSet, MatchSet]:
        strongest = self.evaluate_lineup(config_path, "strongest", strongest_count, label)
        mixed = self.evaluate_lineup(config_path, "mixed", mixed_count, label)
        return strongest, mixed


def set_trial_attributes(trial: optuna.Trial, strongest: MatchSet, mixed: MatchSet, prefix: str = "") -> None:
    for name, value in summary(strongest).items():
        trial.set_user_attr(f"{prefix}strongest_{name}", value)
    for name, value in summary(mixed).items():
        trial.set_user_attr(f"{prefix}mixed_{name}", value)


def write_study_table(study: optuna.Study, path: Path) -> None:
    rows = []
    for trial in study.trials:
        rows.append({
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "expected_mass": trial.user_attrs.get("overall_expected_mass", ""),
            "failure_rate": trial.user_attrs.get("overall_failure_rate", ""),
            "params": json.dumps(trial.params, sort_keys=True),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["trial"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--search-fraction", type=float, default=0.76)
    parser.add_argument("--max-trials", type=int, default=10000)
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--candidate-module", default="my_bot")
    parser.add_argument("--results-dir", default="results/clean_tuning")
    parser.add_argument("--study-name", default="clean_bot_overnight_v1")
    parser.add_argument("--match-timeout", type=float, default=180.0)
    parser.add_argument("--max-attempt-multiplier", type=int, default=3)
    parser.add_argument("--stage1-strongest", type=int, default=4)
    parser.add_argument("--stage1-mixed", type=int, default=2)
    parser.add_argument("--stage2-strongest", type=int, default=8)
    parser.add_argument("--stage2-mixed", type=int, default=3)
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--final-strongest", type=int, default=30)
    parser.add_argument("--final-mixed", type=int, default=10)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument(
        "--strongest", nargs="+",
        default=["1:bots/_tuning_candidate.py", "7:bots/other_bots2.py"],
    )
    parser.add_argument(
        "--mixed", nargs="+",
        default=[
            "1:bots/_tuning_candidate.py", "2:bots/other_bot.py",
            "2:bots/other_bots.py", "3:bots/other_bots2.py",
        ],
    )
    args = parser.parse_args()
    if args.hours <= 0 or not 0.1 <= args.search_fraction <= 0.95:
        parser.error("Use --hours > 0 and --search-fraction between 0.1 and 0.95")

    optimizer = Optimizer(args)
    database = optimizer.results / "study.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{database}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=31, multivariate=True, group=True, n_startup_trials=14,
        ),
        pruner=optuna.pruners.PercentilePruner(
            percentile=50.0, n_startup_trials=12, n_warmup_steps=0,
        ),
    )
    if not study.trials:
        study.enqueue_trial(DEFAULT_TRIAL)

    overall_deadline = time.monotonic() + args.hours * 3600.0
    search_seconds = args.hours * 3600.0 * args.search_fraction

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial)
        label = f"trial_{trial.number:05d}"
        config_path = optimizer.write_config(label, config)
        print(f"\n{label}: order={trial.params['ORDER_PROFILE']}")

        strongest, mixed = optimizer.evaluate_stage(
            config_path, label + "_stage1",
            args.stage1_strongest, args.stage1_mixed,
        )
        stage1 = combined_fitness(strongest, mixed)
        trial.report(stage1, step=0)
        set_trial_attributes(trial, strongest, mixed, "stage1_")
        if trial.should_prune():
            print(f"  pruned after stage 1: expected mass={stage1:.3f}")
            raise optuna.TrialPruned()

        add_strongest, add_mixed = optimizer.evaluate_stage(
            config_path, label + "_stage2",
            args.stage2_strongest, args.stage2_mixed,
        )
        strongest.extend(add_strongest)
        mixed.extend(add_mixed)
        fitness = combined_fitness(strongest, mixed)
        trial.report(fitness, step=1)
        set_trial_attributes(trial, strongest, mixed)
        total = MatchSet(
            masses=strongest.masses + mixed.masses,
            ranks=strongest.ranks + mixed.ranks,
            failures=strongest.failures + mixed.failures,
            elapsed_seconds=strongest.elapsed_seconds + mixed.elapsed_seconds,
        )
        for name, value in summary(total).items():
            trial.set_user_attr(f"overall_{name}", value)
        print(f"  completed: expected mass={fitness:.3f}, failures={summary(total)['failure_rate']:.1%}")
        write_study_table(study, optimizer.results / "trials.csv")
        return fitness

    try:
        study.optimize(
            objective,
            n_trials=max(0, args.max_trials - len(study.trials)),
            timeout=search_seconds,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted; proceeding with completed trials.")

    completed = sorted(
        (trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
        key=lambda trial: trial.value if trial.value is not None else -float("inf"),
        reverse=True,
    )
    if not completed:
        print("No completed trials")
        return 1

    # Recreate exact configs from saved trial files; params contain ORDER_PROFILE
    # rather than the actual OVERRIDE_ORDER list.
    finalists = completed[:args.finalists]
    validation_rows = []
    for place, trial in enumerate(finalists, 1):
        if time.monotonic() >= overall_deadline:
            print("Overall time budget reached before all finalists were validated")
            break
        source = optimizer.configs / f"trial_{trial.number:05d}.json"
        label = f"finalist_{place}_trial_{trial.number:05d}"
        print(f"\nValidating {label}")
        strongest, mixed = optimizer.evaluate_stage(
            source, label, args.final_strongest, args.final_mixed,
        )
        fitness = combined_fitness(strongest, mixed)
        row = {
            "place": place,
            "trial": trial.number,
            "search_fitness": trial.value,
            "validation_expected_mass": fitness,
            "strongest_expected_mass": expected_mass(strongest),
            "mixed_expected_mass": expected_mass(mixed),
            "failures": strongest.failures + mixed.failures,
            "matches": strongest.scheduled + mixed.scheduled,
            "config": str(source),
        }
        validation_rows.append(row)

    validation_rows.sort(key=lambda row: row["validation_expected_mass"], reverse=True)
    validation_path = optimizer.results / "finalists.csv"
    if validation_rows:
        with validation_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=validation_rows[0].keys())
            writer.writeheader()
            writer.writerows(validation_rows)
        winner = validation_rows[0]
        winner_config = Path(winner["config"])
        shutil.copyfile(winner_config, optimizer.results / "best_validated_config.json")
        print("\nBEST VALIDATED CONFIG")
        print(json.dumps(json.loads(winner_config.read_text(encoding="utf-8")), indent=2))
        print(f"Expected mass: {winner['validation_expected_mass']:.3f}")
    else:
        best = completed[0]
        source = optimizer.configs / f"trial_{best.number:05d}.json"
        shutil.copyfile(source, optimizer.results / "best_search_config.json")

    write_study_table(study, optimizer.results / "trials.csv")
    print(f"\nResults: {optimizer.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())