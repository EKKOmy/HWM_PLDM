#!/usr/bin/env python3
"""Compare planning summary JSON files without loading models or pickle files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DIFFICULTIES = ("medium", "hard")
PREFIXES = ("d4rl", "l2_d4rl")
HISTORICAL_RATES = {"flat": (0.675, 0.45), "hwm": (0.80, 0.625)}


@dataclass(frozen=True)
class Measurement:
    success_rate: float
    num_tasks: Optional[int] = None
    successes: Optional[int] = None
    elapsed_seconds: Optional[float] = None


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    prefix: str
    measurements: Dict[str, Measurement]


def _number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def _integer(value: object, key: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def load_run(spec: str, metric_prefix: str = "auto") -> Run:
    """Read a LABEL=path specification; success rates are fractions, not percents."""
    label, separator, filename = spec.partition("=")
    if not separator or not label.strip() or not filename.strip():
        raise ValueError("--run must have the form LABEL=summary.json")
    if metric_prefix not in ("auto",) + PREFIXES:
        raise ValueError(f"Unsupported metric prefix: {metric_prefix}")
    path = Path(filename).expanduser()
    try:
        with path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: summary must be a JSON object")

    prefix = metric_prefix
    if prefix == "auto":
        has_l2 = any(
            f"l2_d4rl_{difficulty}_planning_success_rate" in summary
            for difficulty in DIFFICULTIES
        )
        prefix = "l2_d4rl" if has_l2 else "d4rl"

    measurements = {}
    for difficulty in DIFFICULTIES:
        stem = f"{prefix}_{difficulty}_planning_"
        key = stem + "success_rate"
        if key not in summary:
            continue
        try:
            rate = _number(summary[key], key)
            if not 0 <= rate <= 1:
                raise ValueError(f"{key} must be in [0, 1], not percent units")

            count_key = stem + "num_tasks"
            count = None
            successes = None
            if count_key in summary:
                count = _integer(summary[count_key], count_key, 1)
                successes_key = stem + "successes"
                if successes_key in summary:
                    successes = _integer(summary[successes_key], successes_key, 0)
                    if successes > count:
                        raise ValueError(f"{successes_key} cannot exceed {count_key}")

            elapsed_key = stem + "elapsed_seconds"
            elapsed = None
            if elapsed_key in summary:
                elapsed = _number(summary[elapsed_key], elapsed_key)
                if elapsed < 0:
                    raise ValueError(f"{elapsed_key} must be >= 0")
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        measurements[difficulty] = Measurement(rate, count, successes, elapsed)

    if not measurements:
        raise ValueError(
            f"{path}: no planning success-rate metrics for prefix {prefix!r}; "
            "expected {d4rl,l2_d4rl}_{medium,hard}_planning_success_rate"
        )
    return Run(label.strip(), path, prefix, measurements)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    lines = [" | ".join(value.ljust(w) for value, w in zip(headers, widths))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(value.ljust(w) for value, w in zip(row, widths)) for row in rows
    )
    return "\n".join(lines)


def format_report(runs: Sequence[Run], historical: bool = False) -> str:
    if not runs:
        raise ValueError("At least one measured run is required")
    include_counts = any(
        metric.num_tasks is not None
        for run in runs
        for metric in run.measurements.values()
    )
    include_elapsed = any(
        metric.elapsed_seconds is not None
        for run in runs
        for metric in run.measurements.values()
    )
    headers = ["Run", "Difficulty", "Success", "Delta (pp)", "Metric prefix"]
    if include_counts:
        headers.append("Successes / tasks")
    if include_elapsed:
        headers.append("Elapsed (s)")
    rows = []
    baseline = runs[0]
    for run in runs:
        for difficulty in DIFFICULTIES:
            metric = run.measurements.get(difficulty)
            reference = baseline.measurements.get(difficulty)
            success = "NA" if metric is None else f"{metric.success_rate * 100:.2f}%"
            delta = "NA"
            if metric is not None and reference is not None:
                delta = f"{(metric.success_rate - reference.success_rate) * 100:+.2f}"
            row = [run.label, difficulty, success, delta, run.prefix]
            if include_counts:
                count = "NA"
                if metric is not None and metric.num_tasks is not None:
                    numerator = "NA" if metric.successes is None else str(metric.successes)
                    count = f"{numerator} / {metric.num_tasks}"
                row.append(count)
            if include_elapsed:
                row.append(
                    f"{metric.elapsed_seconds:.3f}"
                    if metric is not None and metric.elapsed_seconds is not None
                    else "NA"
                )
            rows.append(row)

    lines = [
        f"Measured summaries; percentage-point differences relative to {baseline.label!r}.",
        "Differences are descriptive; identical tasks/seeds are not verified.",
        _table(headers, rows),
        "NA = not recorded. Counts are never inferred from success rates.",
    ]
    lines.extend(f"Source [{run.label}]: {run.path}" for run in runs)
    if historical:
        historical_rows = [
            [label, f"{rates[0] * 100:.2f}%", f"{rates[1] * 100:.2f}%"]
            for label, rates in HISTORICAL_RATES.items()
        ]
        lines.extend(
            [
                "",
                "历史参考(未配对) / Historical reference (unpaired)",
                "User-reported values; not re-measured. Task counts and timings unknown.",
                _table(["Reference", "Medium", "Hard"], historical_rows),
            ]
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=summary.json",
        help="Measured summary; repeat this option. The first run is the baseline.",
    )
    parser.add_argument(
        "--metric-prefix", choices=("auto",) + PREFIXES, default="auto",
        help="Auto selects l2_d4rl for the whole run when present, else d4rl.",
    )
    parser.add_argument(
        "--historical", action="store_true",
        help="Append the user-reported flat/HWM historical references (unpaired).",
    )
    args = parser.parse_args(argv)
    try:
        runs: List[Run] = [load_run(spec, args.metric_prefix) for spec in args.run]
        if len({run.label for run in runs}) != len(runs):
            raise ValueError("Run labels must be unique")
        print(format_report(runs, args.historical))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
