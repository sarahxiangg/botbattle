#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import optuna

from benchmark import parse_outcome


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "trials"
RESULTS_DIR = ROOT / "results"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_PER_TRIAL = 4

SIMULATION_COMMAND = [
    "uv",
    "run",
    "simulation",
    "--headless",
    "1:bots/my_bot.py",
    "2:bots/other_bot.py",
    "2:bots/other_bots.py",
    "3:bots/other_bots2.py",
]


def evaluate(config_path: Path) -> dict[str, float]:
    masses: list[float] = []
    ranks: list[int] = []
    failures = 0

    environment = os.environ.copy()
    environment["BOT_CONFIG"] = str(config_path.resolve())

    for match_number in range(1, MATCHES_PER_TRIAL + 1):
        try:
            completed = subprocess.run(
                SIMULATION_COMMAND,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=150,
                check=False,
            )

            output = (
                (completed.stdout or "")
                + "\n"
                + (completed.stderr or "")
            )

            mass, rank = parse_outcome(output, player_id=0)
            masses.append(mass)
            ranks.append(rank)

            print(
                f"    Match {match_number}/{MATCHES_PER_TRIAL}: "
                f"rank={rank}, mass={mass:.2f}"
            )

        except Exception as error:
            failures += 1
            print(
                f"    Match {match_number}/{MATCHES_PER_TRIAL}: "
                f"FAILED: {error}"
            )

    if not ranks:
        return {
            "fitness": -999.0,
            "average_rank": 8.0,
            "median_mass": 0.0,
            "win_rate": 0.0,
            "top_three_rate": 0.0,
            "failure_rate": 1.0,
        }

    successful = len(ranks)

    average_rank = statistics.fmean(ranks)
    median_mass = statistics.median(masses)
    win_rate = sum(rank == 1 for rank in ranks) / successful
    top_three_rate = sum(rank <= 3 for rank in ranks) / successful
    failure_rate = failures / MATCHES_PER_TRIAL

    rank_score = (8.0 - average_rank) / 7.0
    mass_score = min(
        math.log1p(median_mass) / math.log1p(50.0),
        1.5,
    )

    fitness = (
        0.60 * rank_score
        + 0.25 * top_three_rate
        + 0.10 * win_rate
        + 0.05 * mass_score
        - 2.00 * failure_rate
    )

    return {
        "fitness": fitness,
        "average_rank": average_rank,
        "median_mass": median_mass,
        "win_rate": win_rate,
        "top_three_rate": top_three_rate,
        "failure_rate": failure_rate,
    }


def objective(trial: optuna.Trial) -> float:
    config = {
        "TURN_WEIGHT": trial.suggest_float(
            "TURN_WEIGHT", 40.0, 140.0
        ),
        "CHASE_RANGE_MULT": trial.suggest_float(
            "CHASE_RANGE_MULT", 4.5, 9.0
        ),
        "CHASE_LEAD_TICKS": trial.suggest_float(
            "CHASE_LEAD_TICKS", 1.5, 4.0
        ),
        "CHASE_CLOSE_WEIGHT": trial.suggest_float(
            "CHASE_CLOSE_WEIGHT", 12.0, 40.0
        ),
        "SPLIT_RANGE_SAFETY_MULT": trial.suggest_float(
            "SPLIT_RANGE_SAFETY_MULT", 0.65, 0.85
        ),
        "VIRUS_FARM_MAX_RADIUS": trial.suggest_float(
            "VIRUS_FARM_MAX_RADIUS", 12.0, 24.0
        ),
    }

    config_path = CONFIG_DIR / f"trial_{trial.number:04d}.json"
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
        f"median mass={results['median_mass']:.2f}"
    )

    return results["fitness"]


database_path = RESULTS_DIR / "optuna.db"

study = optuna.create_study(
    study_name="bot_tuning_v1",
    storage=f"sqlite:///{database_path}",
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=31),
)

study.optimize(objective, n_trials=3)

print("\nBEST RESULT")
print(f"Fitness: {study.best_value:.4f}")
print(json.dumps(study.best_params, indent=2))