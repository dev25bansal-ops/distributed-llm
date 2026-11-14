"""Render a .pptx to PNGs via PowerPoint COM so the deck can be looked at.

A generated deck that has never been seen is not a finished deck: text can
overflow its box, a colour can be unreadable, a shape can sit off-slide. This
exports one PNG per slide into the target directory.

Run: python scripts/render_deck.py <deck.pptx> <outdir>
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    deck = Path(sys.argv[1]).resolve()
    outdir = Path(sys.argv[2]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    import win32com.client  # noqa: PLC0415

    app = win32com.client.Dispatch("PowerPoint.Application")
    # PowerPoint refuses WithWindow=False for some operations; it is fine to
    # run windowed here because this is a short, local export.
    pres = app.Presentations.Open(str(deck), WithWindow=True)
    try:
        for i, slide in enumerate(pres.Slides, start=1):
            dest = outdir / f"slide-{i:02d}.png"
            slide.Export(str(dest), "PNG", 1600, 900)
            print(f"slide-{i:02d}.png")
    finally:
        pres.Close()
        app.Quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())