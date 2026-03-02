"""Generate markdown/JSON result tables from evaluation results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.runner import EvaluationResults, SessionResult


def _aggregate_metrics(sessions: list[SessionResult]) -> dict[str, dict[str, float]]:
    """Aggregate metrics across sessions, computing mean and std."""
    metric_values: dict[str, list[float]] = defaultdict(list)

    for session in sessions:
        for key, value in session.custom_metrics.items():
            metric_values[key].append(value)
        for key, value in session.ragas_metrics.items():
            metric_values[f"ragas_{key}"].append(value)

    aggregated: dict[str, dict[str, float]] = {}
    for key, values in metric_values.items():
        mean = sum(values) / len(values) if values else 0.0
        variance = (
            sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
        )
        std = variance**0.5
        aggregated[key] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(values)}

    return aggregated


def generate_markdown_report(results: EvaluationResults) -> str:
    """Generate a markdown report from evaluation results."""
    lines: list[str] = ["# Evaluation Results\n"]

    # Group by variant and persona level
    groups: dict[str, dict[str, list[SessionResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for session in results.sessions:
        groups[session.variant][session.persona_level].append(session)

    # Summary table
    lines.append("## Summary\n")
    lines.append(
        "| Variant | Level | Gap Coverage | Precision | Recall | "
        "Hallucination | Citation Rate |"
    )
    lines.append("|---------|-------|-------------|-----------|--------|" "--------------|--------------|")

    for variant in sorted(groups.keys()):
        for level in sorted(groups[variant].keys()):
            sessions = groups[variant][level]
            metrics = _aggregate_metrics(sessions)

            gcr = metrics.get("gap_coverage_rate", {}).get("mean", 0.0)
            prec = metrics.get("element_precision", {}).get("mean", 0.0)
            rec = metrics.get("element_recall", {}).get("mean", 0.0)
            hall = metrics.get("hallucination_rate", {}).get("mean", 0.0)
            cit = metrics.get("evidence_citation_rate", {}).get("mean", 0.0)

            lines.append(
                f"| {variant} | {level} | {gcr:.4f} | {prec:.4f} | "
                f"{rec:.4f} | {hall:.4f} | {cit:.4f} |"
            )

    lines.append("")

    # Detailed per-variant sections
    for variant in sorted(groups.keys()):
        lines.append(f"## {variant}\n")
        for level in sorted(groups[variant].keys()):
            sessions = groups[variant][level]
            metrics = _aggregate_metrics(sessions)

            lines.append(f"### {level} (n={len(sessions)})\n")
            lines.append("| Metric | Mean | Std |")
            lines.append("|--------|------|-----|")
            for metric_name, stats in sorted(metrics.items()):
                lines.append(
                    f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} |"
                )
            lines.append("")

    return "\n".join(lines)


def generate_json_report(results: EvaluationResults) -> dict:
    """Generate a JSON-serializable report from evaluation results."""
    groups: dict[str, dict[str, list[SessionResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for session in results.sessions:
        groups[session.variant][session.persona_level].append(session)

    report: dict = {}
    for variant in groups:
        report[variant] = {}
        for level in groups[variant]:
            sessions = groups[variant][level]
            metrics = _aggregate_metrics(sessions)
            report[variant][level] = {
                "n_sessions": len(sessions),
                "metrics": metrics,
            }

    return report


def save_report(
    results: EvaluationResults,
    output_dir: Path,
) -> None:
    """Save both markdown and JSON reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_report = generate_markdown_report(results)
    (output_dir / "evaluation_report.md").write_text(md_report)

    json_report = generate_json_report(results)
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(json_report, indent=2)
    )
