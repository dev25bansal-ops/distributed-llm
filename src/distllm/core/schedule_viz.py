"""Schedule visualizer — ASCII and HTML timeline output.

Produces visual representations of scheduling decisions for debugging
and analysis.  Supports both terminal (ASCII) and browser (HTML) output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScheduleSnapshot:
    """A single scheduling iteration snapshot."""
    iteration: int
    timestamp: float
    active: list[dict]  # [{request_id, status, priority, tokens}]
    pending: list[dict]
    preempted: list[str]
    budget: dict  # {max_prefill, max_decode, max_batch}
    batch_seqs: list[str]  # request_ids in this batch
    prefill_tokens: int = 0
    decode_tokens: int = 0


class ScheduleVisualizer:
    """Captures and visualizes scheduling decisions.

    Usage::

        viz = ScheduleVisualizer()
        # In the scheduler loop:
        viz.capture(scheduler)
        # Output:
        print(viz.to_ascii())
        viz.to_html("schedule.html")
    """

    def __init__(self, max_history: int = 100):
        self._history: list[ScheduleSnapshot] = []
        self._max_history = max_history

    def capture(self, scheduler: Any) -> None:
        """Capture a snapshot of the current scheduler state."""
        snapshot = ScheduleSnapshot(
            iteration=getattr(scheduler, '_iteration_count', len(self._history)),
            timestamp=time.time(),
            active=[
                {
                    "request_id": rid,
                    "status": seq.status.value if hasattr(seq.status, 'value') else str(seq.status),
                    "priority": seq.priority,
                    "tokens": seq.total_len,
                    "generated": len(seq.generated_tokens),
                }
                for rid, seq in getattr(scheduler, 'active', {}).items()
            ],
            pending=[
                {
                    "request_id": seq.request_id,
                    "priority": seq.priority,
                    "tokens": seq.total_len,
                }
                for _, _, seq in getattr(scheduler, '_pending_heap', [])
            ],
            preempted=list(getattr(scheduler, '_preempted', {}).keys()),
            budget={
                "max_batch_size": getattr(scheduler, 'max_batch_size', 0),
                "max_tokens": getattr(scheduler, 'max_tokens_per_batch', 0),
            },
            batch_seqs=[],  # Populated by caller
            prefill_tokens=getattr(scheduler, '_total_prefill_tokens', 0),
            decode_tokens=getattr(scheduler, '_total_decode_tokens', 0),
        )
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def to_ascii(self, last_n: int = 20) -> str:
        """Render recent scheduling history as ASCII timeline.

        Each row shows one iteration with active sequences as colored blocks.
        """
        if not self._history:
            return "No scheduling history captured."

        lines = []
        lines.append("Schedule Timeline (most recent first)")
        lines.append("=" * 80)

        snapshots = self._history[-last_n:]
        for snap in reversed(snapshots):
            # Header line
            active_count = len(snap.active)
            pending_count = len(snap.pending)
            preempted_count = len(snap.preempted)
            lines.append(
                f"  iter={snap.iteration:4d}  "
                f"active={active_count:2d}  "
                f"pending={pending_count:3d}  "
                f"preempted={preempted_count:1d}  "
                f"prefill={snap.prefill_tokens:6d}  "
                f"decode={snap.decode_tokens:6d}"
            )

            # Active sequences bar
            if snap.active:
                bar_parts = []
                for seq in snap.active:
                    status = seq["status"]
                    pri = seq["priority"]
                    rid = seq["request_id"][:8]
                    if status == "prefilling":
                        bar_parts.append(f"\033[43m{rid}\033[0m")  # Yellow
                    elif status == "decoding":
                        bar_parts.append(f"\033[42m{rid}\033[0m")  # Green
                    else:
                        bar_parts.append(f"\033[47m{rid}\033[0m")  # White
                lines.append(f"    [{' '.join(bar_parts)}]")

            # Pending queue
            if snap.pending:
                pri_counts: dict[int, int] = {}
                for p in snap.pending:
                    pri = p["priority"]
                    pri_counts[pri] = pri_counts.get(pri, 0) + 1
                pri_str = " ".join(f"P{p}:{c}" for p, c in sorted(pri_counts.items()))
                lines.append(f"    pending: {pri_str}")

        lines.append("")
        lines.append("Legend: \033[43m████\033[0m=prefill  \033[42m████\033[0m=decode  \033[47m████\033[0m=other")
        return "\n".join(lines)

    def to_html(self, output_path: str | None = None) -> str:
        """Render scheduling history as an interactive HTML timeline.

        Args:
            output_path: If provided, writes HTML to this file.

        Returns:
            HTML string.
        """
        if not self._history:
            html = "<html><body><p>No scheduling history captured.</p></body></html>"
        else:
            rows = []
            for snap in reversed(self._history[-50:]):
                active_cells = []
                for seq in snap.active:
                    status = seq["status"]
                    rid = seq["request_id"][:12]
                    pri = seq["priority"]
                    color = {
                        "prefilling": "#f59e0b",
                        "decoding": "#10b981",
                        "pending": "#6b7280",
                        "preempted": "#ef4444",
                    }.get(status, "#9ca3af")
                    active_cells.append(
                        f'<span style="background:{color};color:#fff;'
                        f'padding:2px 6px;margin:1px;border-radius:3px;'
                        f'font-size:11px;" title="{rid} pri={pri}">{rid}</span>'
                    )

                rows.append(f"""
                <tr>
                    <td>{snap.iteration}</td>
                    <td>{len(snap.active)}</td>
                    <td>{len(snap.pending)}</td>
                    <td>{len(snap.preempted)}</td>
                    <td>{snap.prefill_tokens}</td>
                    <td>{snap.decode_tokens}</td>
                    <td>{' '.join(active_cells) if active_cells else '<em>idle</em>'}</td>
                </tr>""")

            table_rows = "\n".join(rows)
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>DistLLM Schedule Visualizer</title>
    <style>
        body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
        h1 {{ color: #00d4ff; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
        th {{ background: #16213e; color: #00d4ff; }}
        tr:hover {{ background: #16213e; }}
        .legend {{ margin: 10px 0; font-size: 12px; }}
        .legend span {{ padding: 2px 8px; margin-right: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>DistLLM Schedule Timeline</h1>
    <div class="legend">
        <span style="background:#f59e0b;color:#fff;">prefill</span>
        <span style="background:#10b981;color:#fff;">decode</span>
        <span style="background:#ef4444;color:#fff;">preempted</span>
        <span style="background:#6b7280;color:#fff;">pending</span>
    </div>
    <table>
        <tr>
            <th>Iteration</th><th>Active</th><th>Pending</th>
            <th>Preempted</th><th>Prefill Tok</th><th>Decode Tok</th>
            <th>Sequences</th>
        </tr>
        {table_rows}
    </table>
    <p style="color:#666;font-size:11px;">Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>"""

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    def stats(self) -> dict:
        """Get summary statistics from captured history."""
        if not self._history:
            return {"snapshots": 0}

        total_active = sum(len(s.active) for s in self._history)
        total_pending = sum(len(s.pending) for s in self._history)
        return {
            "snapshots": len(self._history),
            "avg_active": round(total_active / len(self._history), 1),
            "avg_pending": round(total_pending / len(self._history), 1),
            "max_active": max(len(s.active) for s in self._history),
            "max_pending": max(len(s.pending) for s in self._history),
        }
