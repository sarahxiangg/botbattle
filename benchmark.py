#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RESULT_TYPE_RE = re.compile(r"result_type=['\"]([^'\"]+)['\"]")
RANKING_RE = re.compile(r"ranking=(\[[^\]]*\])")
MASSES_RE = re.compile(r"final_masses=(\{[^}]*\})")


def parse_outcome(output: str, player_id: int) -> tuple[float, int]:
    clean = ANSI_ESCAPE.sub("", output)
    lines = [
        line for line in clean.splitlines()
        if "match complete" in line
        and "ranking=" in line
        and "final_masses=" in line
    ]
    if not lines:
        raise ValueError("No completed-match outcome line found.")

    line = lines[-1]

    result_match = RESULT_TYPE_RE.search(line)
    if result_match and result_match.group(1) != "SUCCESS":
        raise ValueError(f"Result was {result_match.group(1)!r}, not SUCCESS.")

    ranking_match = RANKING_RE.search(line)
    masses_match = MASSES_RE.search(line)
    if ranking_match is None or masses_match is None:
        raise ValueError("Could not parse ranking or final_masses.")

    ranking = ast.literal_eval(ranking_match.group(1))
    masses = ast.literal_eval(masses_match.group(1))

    if player_id not in ranking:
        raise ValueError(f"Player {player_id} missing from ranking.")
    if player_id not in masses:
        raise ValueError(f"Player {player_id} missing from final_masses.")

    return float(masses[player_id]), ranking.index(player_id) + 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated bot simulations and benchmark one player."
    )
    parser.add_argument("-n", "--runs", type=int, default=200)
    parser.add_argument("-p", "--player", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results.csv"),
    )
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be greater than zero.")

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Supply the simulation command after --.")

    masses: list[float] = []
    ranks: list[int] = []
    rows: list[dict[str, object]] = []
    failures = 0
    benchmark_start = time.perf_counter()

    print(f"Running {args.runs} simulations for player {args.player}")
    print("Command:", " ".join(command))
    print()

    try:
        for run_number in range(1, args.runs + 1):
            run_start = time.perf_counter()
            output = ""

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                output = (completed.stdout or "") + "\n" + (completed.stderr or "")
                mass, rank = parse_outcome(output, args.player)
                elapsed = time.perf_counter() - run_start

                masses.append(mass)
                ranks.append(rank)
                rows.append({
                    "run": run_number,
                    "success": True,
                    "final_mass": mass,
                    "rank": rank,
                    "elapsed_seconds": round(elapsed, 6),
                    "return_code": completed.returncode,
                    "error": "",
                })

                print(
                    f"\rRun {run_number:>4}/{args.runs} | "
                    f"mass {mass:>9.3f} | rank {rank} | "
                    f"ok {len(masses)} | failed {failures}",
                    end="",
                    flush=True,
                )

            except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
                failures += 1
                elapsed = time.perf_counter() - run_start
                rows.append({
                    "run": run_number,
                    "success": False,
                    "final_mass": "",
                    "rank": "",
                    "elapsed_seconds": round(elapsed, 6),
                    "return_code": "",
                    "error": str(exc),
                })

                print(
                    f"\rRun {run_number:>4}/{args.runs} | FAILED | "
                    f"ok {len(masses)} | failed {failures}",
                    end="",
                    flush=True,
                )

                if args.show_failures:
                    print(f"\nFailure on run {run_number}: {exc}")
                    if output:
                        print(output.rstrip())
                    print()

    except KeyboardInterrupt:
        print("\nInterrupted; saving collected results.")

    print("\n")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run",
                "success",
                "final_mass",
                "rank",
                "elapsed_seconds",
                "return_code",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if not masses:
        print("No successful outcomes were parsed.")
        print(f"CSV saved to: {args.csv.resolve()}")
        return 1

    elapsed_total = time.perf_counter() - benchmark_start
    rank_counts = Counter(ranks)
    successful = len(masses)
    wins = rank_counts[1]
    top_three = sum(count for rank, count in rank_counts.items() if rank <= 3)

    print("=" * 58)
    print(f"PLAYER {args.player} RESULTS")
    print("=" * 58)
    print(f"Successful runs:       {successful}/{len(rows)}")
    print(f"Failed runs:           {failures}")
    print()
    print(f"Average final mass:    {statistics.fmean(masses):.4f}")
    print(f"Median final mass:     {statistics.median(masses):.4f}")
    print(
        f"Mass standard dev.:    "
        f"{statistics.stdev(masses) if successful > 1 else 0.0:.4f}"
    )
    print(f"Minimum final mass:    {min(masses):.4f}")
    print(f"Maximum final mass:    {max(masses):.4f}")
    print()
    print(f"Average rank:          {statistics.fmean(ranks):.4f}")
    print(f"Median rank:           {statistics.median(ranks):.2f}")
    print(f"Wins:                  {wins}/{successful} ({wins / successful:.1%})")
    print(
        f"Top-three finishes:    {top_three}/{successful} "
        f"({top_three / successful:.1%})"
    )
    distribution = ", ".join(
        f"{rank}: {rank_counts[rank]}" for rank in sorted(rank_counts)
    )
    print(f"Rank distribution:     {distribution}")
    print()
    print(f"Total elapsed:         {elapsed_total:.2f} seconds")
    print(f"Average per run:       {elapsed_total / max(len(rows), 1):.2f} seconds")
    print(f"Per-run CSV:           {args.csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())