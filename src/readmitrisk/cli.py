"""Command-line entry point for ReadmitRisk.

Each Makefile target maps to a subcommand here:

    make generate -> readmitrisk generate
    make cohort   -> readmitrisk cohort
    make train    -> readmitrisk train
    make eval     -> readmitrisk eval     (the eval gate; non-zero exit on failure)
    make fairness -> readmitrisk fairness

Imports of phase-specific modules are deferred into each handler so that early phases
run even before later modules exist, and so a missing heavy dependency in one command
never blocks the others.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def cmd_generate(args: argparse.Namespace) -> int:
    from .duckdb_util import raw_counts
    from .generate import run_generation
    from .paths import get_paths

    cfg = load_config()
    _print_header("ReadmitRisk — synthetic data generation")
    print(
        f"backend={args.backend or cfg.generation.backend}  "
        f"population={cfg.generation.population}  seed={cfg.generation.seed}"
    )
    summary = run_generation(cfg, force_backend=args.backend)
    print("\nGeneration summary:")
    print(json.dumps(summary, indent=2))

    # DuckDB smoke test: prove the output is readable and print counts.
    counts = raw_counts(get_paths().raw)
    print("\nDuckDB smoke counts:")
    for k, v in counts.items():
        print(f"  {k:<22} {v:>10,}")
    if not summary.get("tables_present"):
        print("ERROR: not all required tables were produced.", file=sys.stderr)
        return 1
    return 0


def cmd_smoke(_args: argparse.Namespace) -> int:
    from .duckdb_util import raw_counts
    from .paths import get_paths

    _print_header("ReadmitRisk — DuckDB smoke test")
    counts = raw_counts(get_paths().raw)
    for k, v in counts.items():
        print(f"  {k:<22} {v:>10,}")
    return 0


def cmd_cohort(args: argparse.Namespace) -> int:
    from .cohort import build_cohort

    _print_header("ReadmitRisk — cohort construction (DuckDB SQL)")
    result = build_cohort(write=True, use_sample=args.sample)
    print(result.describe())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .models.pipeline import train_models

    _print_header("ReadmitRisk — model training (Cox PH + RSF)")
    report = train_models(use_sample=args.sample)
    print(report.summary())
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation.evaluate import run_eval

    _print_header("ReadmitRisk — evaluation (survival metrics + gate)")
    passed = run_eval(use_sample=args.sample)
    return 0 if passed else 1


def cmd_fairness(args: argparse.Namespace) -> int:
    from .fairness.audit import run_fairness

    _print_header("ReadmitRisk — fairness audit (subgroup C-index + calibration)")
    run_fairness(use_sample=args.sample)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readmitrisk", description="ReadmitRisk pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic EHR data")
    g.add_argument("--backend", choices=["synthea", "fallback", "auto"], default=None)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("smoke", help="DuckDB smoke test over raw data")
    s.set_defaults(func=cmd_smoke)

    for name, fn, helptext in (
        ("cohort", cmd_cohort, "build the time-to-event cohort via DuckDB SQL"),
        ("train", cmd_train, "fit Cox PH + Random Survival Forest"),
        ("eval", cmd_eval, "evaluate survival metrics and enforce the gate"),
        ("fairness", cmd_fairness, "run the subgroup fairness audit"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument(
            "--sample",
            action="store_true",
            help="use the small committed cached sample instead of full generated data",
        )
        p.set_defaults(func=fn)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
