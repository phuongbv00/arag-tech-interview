"""CLI: Run the full evaluation pipeline."""

import argparse
import asyncio
import logging
from pathlib import Path

from src.config import Settings
from src.evaluation.report import save_report
from src.evaluation.runner import EvaluationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=Path("data/kb/concepts.json"),
        help="Path to concepts JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for evaluation reports",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per condition",
    )
    args = parser.parse_args()

    settings = Settings()

    runner = EvaluationRunner(
        settings=settings,
        kb_path=args.kb_path,
        repetitions=args.repetitions,
    )

    print("Starting full evaluation pipeline...")
    results = await runner.run_full_evaluation()

    print(f"Saving reports to {args.output_dir}...")
    save_report(results, args.output_dir)

    print("Evaluation complete.")
    print(f"  - Markdown report: {args.output_dir / 'evaluation_report.md'}")
    print(f"  - JSON report: {args.output_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
