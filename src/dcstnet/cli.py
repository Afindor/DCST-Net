from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from .pipeline import (
    prepare_context,
    run_all,
    run_evaluation,
    run_materialization,
    run_phase_a,
    run_phase_b,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcst", description="DCST-Net experiment and replay pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("prepare", "validate inputs and construct split metadata"),
        ("train-phase-a", "train the booking-pressure residual forecaster"),
        ("materialize", "write the detached Phase-A trend cache"),
        ("train-phase-b", "train route, calendar, and fusion components"),
        ("evaluate", "run leakage-safe test replay"),
        ("run", "execute the complete two-phase pipeline"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--config", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        result = run_all(arguments.config)
    else:
        context = prepare_context(arguments.config)
        actions: dict[str, Callable[[], object]] = {
            "prepare": lambda: {
                "train": len(context.data.train),
                "validation": len(context.data.validation),
                "test": len(context.data.test),
            },
            "train-phase-a": lambda: run_phase_a(context),
            "materialize": lambda: {"cached_samples": len(run_materialization(context).values)},
            "train-phase-b": lambda: run_phase_b(context),
            "evaluate": lambda: run_evaluation(context),
        }
        result = actions[arguments.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
