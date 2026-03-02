"""CLI: Run baseline + ablation experiments."""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Literal

from src.config import Settings
from src.evaluation.report import save_report
from src.evaluation.runner import EvaluationResults, EvaluationRunner, Variant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and ablation experiments")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=Path("data/kb/concepts.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["no_rag", "no_kgda", "no_raa_ground"],
        help="Variants to run",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    settings = Settings()
    runner = EvaluationRunner(
        settings=settings,
        kb_path=args.kb_path,
        repetitions=args.repetitions,
    )

    results = EvaluationResults()
    persona_levels = ["L1", "L3"]

    for variant in args.variants:
        for level in persona_levels:
            for rep in range(args.repetitions):
                print(f"Running: variant={variant}, level={level}, rep={rep}")
                persona = runner._build_persona(level)
                session_result = await runner.run_session(variant, persona)
                session_result.repetition = rep
                results.sessions.append(session_result)

    save_report(results, args.output_dir)
    print(f"Baseline/ablation results saved to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
