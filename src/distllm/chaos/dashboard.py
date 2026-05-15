"""CLI dashboard for chaos engineering results."""

from typing import List, Optional

from distllm.chaos.injector import ChaosEvent
from distllm.chaos.resilience import ResilienceScore


def _render_table(headers: List[str], rows: List[List[str]], widths: Optional[List[int]] = None) -> str:
    """Render a simple text table."""
    if widths is None:
        widths = [max(len(str(h)), max((len(str(row[i])) for row in rows), default=0)) for i, h in enumerate(headers)]

    def format_row(values):
        return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))

    lines = [format_row(headers)]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def render_scenario_summary(results: List, scores: List[ResilienceScore]) -> str:
    """Render a summary of scenario results."""
    headers = ["Scenario", "Steps", "Failed", "Duration", "Recovery", "Score", "Grade"]
    rows = []
    for i, result in enumerate(results):
        score = scores[i] if i < len(scores) else ResilienceScore(0, 0, 0, 0)
        rows.append([
            result.scenario_name,
            str(result.steps_executed),
            str(result.steps_failed),
            f"{result.total_duration_s:.1f}s",
            f"{result.actual_recovery_time_s:.1f}s",
            f"{score.overall:.0f}",
            score.grade,
        ])
    return _render_table(headers, rows)


def render_events(events: List[ChaosEvent]) -> str:
    """Render a list of chaos events."""
    headers = ["Type", "Node", "Result", "Duration"]
    rows = []
    for event in events[-20:]:  # Last 20 events
        rows.append([
            event.event_type,
            event.node_id,
            event.result,
            f"{event.duration_s:.2f}s",
        ])
    return _render_table(headers, rows)


def render_resilience_summary(scores: List[ResilienceScore]) -> str:
    """Render overall resilience summary."""
    if not scores:
        return "No chaos scenarios have been executed yet."

    avg_score = sum(s.overall for s in scores) / len(scores)
    best = max(scores, key=lambda s: s.overall)
    worst = min(scores, key=lambda s: s.overall)

    lines = [
        "=== Chaos Engineering Resilience Summary ===",
        "",
        f"Scenarios executed: {len(scores)}",
        f"Average resilience score: {avg_score:.1f}/100",
        f"Best scenario: {best.scenario_name} ({best.overall:.0f}, grade {best.grade})",
        f"Worst scenario: {worst.scenario_name} ({worst.overall:.0f}, grade {worst.grade})",
    ]
    return "\n".join(lines)
