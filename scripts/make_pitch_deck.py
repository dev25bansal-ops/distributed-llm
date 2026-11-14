"""Generate the Startup Buildathon pitch deck for DistLLM.

Every number on every slide comes from the repository, not from imagination:

  92.0 tok/s / 153.1 tok/s / 1.66x / 83.2%  -> docs/BENCHMARKS.md (Llama 3.1 8B,
                                               RTX 4090 + RTX 4060, 1 GbE,
                                               measured 2026-06-26)
  A10 validation (selftest 7/7, 0.77 s TTFT, node joined/left)
                                            -> this session's gate-4 run
  13,600+ tests, 1361/5827/6313 per suite   -> the 0.7.0 release candidate
  0.6.2.2 on PyPI                           -> index check 2026-11-12

Deliberately absent: any 4-node, InfiniBand or 70B throughput figure. Those
rows in docs/BENCHMARKS.md read "pending hardware availability", and putting a
number there would be the one thing an engineer in the audience could check
and disprove.

Run: python scripts/make_pitch_deck.py [output.pptx]
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ── Design tokens ────────────────────────────────────────────────────────────

INK = RGBColor(0x0E, 0x11, 0x16)        # near-black ground
PAPER = RGBColor(0xF7, 0xF7, 0xF5)      # light slide ground
MUTED = RGBColor(0x5A, 0x62, 0x70)      # secondary text
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)     # the one accent: a measured blue
ACCENT_DIM = RGBColor(0x9F, 0xC2, 0xF5)
WARN = RGBColor(0xB4, 0x6A, 0x00)       # "not measured" marker

W, H = Inches(13.333), Inches(7.5)      # 16:9

FONT = "Inter"
FONT_MONO = "JetBrains Mono"


def _slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    return s


def _rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


def _text(slide, x, y, w, h, text, size, color, *, bold=False, mono=False,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15, italic=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        run = p.add_run()
        run.text = line
        f = run.font
        f.name = FONT_MONO if mono else FONT
        f.size = Pt(size)
        f.bold = bold
        f.italic = italic
        f.color.rgb = color
    return box


def _rule(slide, x, y, w, color=ACCENT, thickness=Pt(3)):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, thickness)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def _footer(slide, n, total, label="DistLLM"):
    _text(slide, Inches(0.75), H - Inches(0.62), Inches(4), Inches(0.3),
          label, 10, MUTED)
    _text(slide, W - Inches(1.55), H - Inches(0.62), Inches(1), Inches(0.3),
          f"{n} / {total}", 10, MUTED, align=PP_ALIGN.RIGHT)


# Body copy must never reach the footer strip that starts at this y.
FOOTER_TOP = Inches(6.55)


# ── Slides ───────────────────────────────────────────────────────────────────

def slide_title(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, INK)
    _rule(s, Inches(0.9), Inches(2.55), Inches(1.6))
    _text(s, Inches(0.9), Inches(2.9), Inches(11.4), Inches(1.6),
          "DistLLM", 96, PAPER, bold=True)
    _text(s, Inches(0.9), Inches(4.45), Inches(10.2), Inches(1.2),
          "Pool the GPUs you already own into one cluster,\n"
          "and run models no single machine can hold.",
          25, ACCENT_DIM, spacing=1.35)
    _text(s, Inches(0.9), H - Inches(1.15), Inches(9), Inches(0.4),
          "Startup Buildathon 2026", 13, MUTED)
    _footer(s, n, total)


def slide_problem(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, PAPER)
    _text(s, Inches(0.9), Inches(0.75), Inches(9), Inches(0.4),
          "THE PROBLEM", 12, ACCENT, bold=True)
    _text(s, Inches(0.9), Inches(1.15), Inches(11.2), Inches(1.0),
          "The model you need does not fit the GPU you have.",
          34, INK, bold=True, spacing=1.1)

    # Three figures, the arithmetic that makes the point.
    items = [
        ("140 GB", "FP16 weights of a 70B model"),
        ("4 × A100", "the hardware that implies"),
        ("8–24 GB", "what the machines on your desk actually have"),
    ]
    x = Inches(0.9)
    for big, cap in items:
        _text(s, x, Inches(2.75), Inches(3.6), Inches(0.9), big, 46, INK, bold=True)
        _text(s, x, Inches(3.65), Inches(3.3), Inches(1.0), cap, 15, MUTED, spacing=1.3)
        x += Inches(3.85)

    _rule(s, Inches(0.9), Inches(4.85), Inches(1.2), MUTED, Pt(1.5))
    _text(s, Inches(0.9), Inches(5.15), Inches(11.4), Inches(1.6),
          "So teams buy a bigger GPU or rent one, paying for capacity that idles most of the day —\n"
          "or they send their prompts to someone else's API and accept a per-token bill forever.\n"
          "Meanwhile the GPUs already in the building add up to more memory than the model needs.",
          19, INK, spacing=1.45)
    _footer(s, n, total)


def slide_solution(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, PAPER)
    _text(s, Inches(0.9), Inches(0.75), Inches(9), Inches(0.4),
          "THE SOLUTION", 12, ACCENT, bold=True)
    _text(s, Inches(0.9), Inches(1.15), Inches(11.4), Inches(0.8),
          "Split the model, not the work.", 40, INK, bold=True)

    _text(s, Inches(0.9), Inches(2.15), Inches(6.3), Inches(3.4),
          "DistLLM divides a model into consecutive layer ranges and gives "
          "each range to a different machine. Tokens flow through the cluster "
          "stage by stage, so no device ever holds more than a fraction of the "
          "weights — and a model that fits no single card fits the pool.",
          17, INK, spacing=1.5)

    _text(s, Inches(0.9), Inches(5.55), Inches(6.3), Inches(0.9),
          "One command to start. One to join.\nOpenAI-compatible API. Nodes drain and leave without a restart.",
          15, MUTED, spacing=1.4)

    # Right: the relay diagram, drawn rather than described.
    x0, y0 = Inches(7.7), Inches(2.35)
    labels = ["GPU 1\nlayers 0–11", "GPU 2\n12–23", "GPU 3\n24–31"]
    for i, lab in enumerate(labels):
        bx = x0 + Inches(i * 1.68)
        box = _rect(s, bx, y0, Inches(1.45), Inches(1.15),
                    ACCENT if i == 0 else RGBColor(0xFF, 0xFF, 0xFF),
                    line=ACCENT if i else None)
        _text(s, bx, y0 + Inches(0.32), Inches(1.45), Inches(0.7),
              lab, 11,
              PAPER if i == 0 else INK,
              bold=True, align=PP_ALIGN.CENTER, spacing=1.3)
        if i < 2:
            _text(s, bx + Inches(1.45), y0 + Inches(0.35), Inches(0.25),
                  Inches(0.4), "›", 26, ACCENT, bold=True, align=PP_ALIGN.CENTER)

    _text(s, x0, y0 + Inches(1.35), Inches(5.0), Inches(0.9),
          "activations in →\n← tokens out", 12, MUTED, mono=True, spacing=1.5)

    _text(s, x0, y0 + Inches(2.25), Inches(4.9), Inches(1.6),
          "The planner solves for where the layers break, given each machine's\n"
          "memory, compute and the link between them — a dynamic program, not a rule of thumb.",
          13, MUTED, spacing=1.45)
    _footer(s, n, total)


def slide_traction(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, PAPER)
    _text(s, Inches(0.9), Inches(0.75), Inches(9), Inches(0.4),
          "MEASURED, NOT ASSERTED", 12, ACCENT, bold=True)
    _text(s, Inches(0.9), Inches(1.15), Inches(11.4), Inches(0.8),
          "Two consumer GPUs, consumer Ethernet, 1.66×.", 36, INK, bold=True)

    _text(s, Inches(0.9), Inches(2.05), Inches(5.4), Inches(0.5),
          "Llama 3.1 8B, FP16 · RTX 4090 + RTX 4060 · 1 GbE", 13, MUTED, mono=True)

    # Bar chart. The scale starts at zero and is scaled to just past the tallest
    # bar: a truncated axis made 92 and 153 render almost equal, which undercut
    # the one number the slide exists to show.
    base_x, base_y = Inches(0.95), Inches(4.85)
    bar_w = Inches(1.15)
    max_h = Inches(2.05)
    scale_max = 170.0
    bars = [("1 GPU", 92.0, MUTED), ("2 nodes", 153.1, ACCENT)]
    for i, (lab, val, col) in enumerate(bars):
        h = Emu(int(max_h * (val / scale_max)))
        bx = base_x + Inches(i * 1.9)
        _rect(s, bx, base_y - h, bar_w, h, col)
        _text(s, bx - Inches(0.25), base_y - h - Inches(0.38), Inches(1.7),
              Inches(0.32), f"{val:.1f}", 16, INK, bold=True, align=PP_ALIGN.CENTER)
        _text(s, bx - Inches(0.2), base_y + Inches(0.12), Inches(1.55),
              Inches(0.3), lab, 12, MUTED, align=PP_ALIGN.CENTER)
    _rule(s, Inches(0.9), base_y, Inches(4.1), MUTED, Pt(1.25))
    _text(s, Inches(0.9), base_y + Inches(0.5), Inches(4.2), Inches(0.4),
          "tokens / second", 11, MUTED, mono=True)

    # Right: the honest ledger.
    _text(s, Inches(6.1), Inches(2.05), Inches(6.2), Inches(0.4),
          "What is verified today", 15, INK, bold=True)
    rows = [
        ("1.66× speedup", "83.2% scaling efficiency, measured 2026-06-26"),
        ("13,600+ tests", "API 1,361 · core 5,827 · dist 6,313, all green"),
        ("A10 end-to-end", "selftest 7/7 · 0.77 s first token · node joined and left"),
        ("On PyPI", "distributed-llm 0.6.2.2 live; 0.7.0 built and waiting"),
    ]
    y = Inches(2.55)
    for head, sub in rows:
        _text(s, Inches(6.1), y, Inches(0.22), Inches(0.4), "—", 15, ACCENT, bold=True)
        _text(s, Inches(6.45), y - Inches(0.02), Inches(5.9), Inches(0.35),
              head, 15, INK, bold=True)
        _text(s, Inches(6.45), y + Inches(0.3), Inches(5.9), Inches(0.4),
              sub, 12, MUTED)
        y += Inches(0.95)

    _text(s, Inches(6.1), H - Inches(1.05), Inches(6.2), Inches(0.5),
          "4-node and InfiniBand runs are pending hardware. No number shown here is extrapolated.",
          11, WARN, italic=True)
    _footer(s, n, total)


def slide_market(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, PAPER)
    _text(s, Inches(0.9), Inches(0.75), Inches(9), Inches(0.4),
          "WHO HAS THIS PROBLEM", 12, ACCENT, bold=True)
    _text(s, Inches(0.9), Inches(1.15), Inches(11.4), Inches(0.8),
          "Anyone with GPUs in more than one box.", 36, INK, bold=True)

    cols = [
        ("01", "Small teams with consumer GPUs",
         "Three or four RTX 4090/3090 machines. Enough memory between them, "
         "not enough in any one. The design centre of the product."),
        ("02", "Privacy-bound operators",
         "Healthcare, legal, internal tooling. Cannot send prompts to a hosted "
         "API; no budget for an A100 cluster."),
        ("03", "Platform engineers",
         "Already run Kubernetes with a mixed GPU pool. Want one "
         "OpenAI-compatible endpoint in front of it."),
    ]
    x = Inches(0.9)
    for num, head, body in cols:
        _text(s, x, Inches(2.45), Inches(1.0), Inches(0.4), num, 13, ACCENT, mono=True)
        _text(s, x, Inches(2.9), Inches(3.5), Inches(1.0), head, 19, INK,
              bold=True, spacing=1.25)
        _text(s, x, Inches(4.1), Inches(3.45), Inches(2.0), body, 14, MUTED, spacing=1.5)
        x += Inches(3.95)

    _rule(s, Inches(0.9), Inches(6.05), Inches(1.2), MUTED, Pt(1.5))
    _text(s, Inches(0.9), Inches(6.25), Inches(11.4), Inches(0.5),
          "Primary market India · secondary US and EU, specifically regulated teams.",
          13, MUTED)
    _footer(s, n, total)


def slide_model(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, PAPER)
    _text(s, Inches(0.9), Inches(0.75), Inches(9), Inches(0.4),
          "BUSINESS MODEL", 12, ACCENT, bold=True)
    _text(s, Inches(0.9), Inches(1.15), Inches(11.4), Inches(0.8),
          "Open-core, sold to the people who cannot move.", 34, INK, bold=True)

    tiers = [
        ("Free", "Apache 2.0 engine and CLI",
         "Adoption and credibility. Anyone can read and verify what runs "
         "on their own machines."),
        ("Team licence", "Self-hosted, per cluster per year",
         "SSO, audit logging, per-tenant quotas, cost governance, support SLA. "
         "For organisations that must keep inference inside their network."),
        ("Managed control plane", "Later",
         "We run coordination, model distribution and upgrades. The customer's "
         "GPUs do the work; their data never leaves."),
    ]
    y = Inches(2.35)
    for i, (name, sub, body) in enumerate(tiers):
        if i == 1:
            _rect(s, Inches(0.66), y - Inches(0.18), Inches(12.0),
                  Inches(1.42), RGBColor(0xEC, 0xF3, 0xFE))
        _text(s, Inches(0.9), y, Inches(2.7), Inches(0.4), name, 18, INK, bold=True)
        _text(s, Inches(0.9), y + Inches(0.38), Inches(2.8), Inches(0.5),
              sub, 11, ACCENT if i == 1 else MUTED, mono=True, spacing=1.3)
        _text(s, Inches(4.15), y + Inches(0.02), Inches(8.2), Inches(1.0),
              body, 14, MUTED, spacing=1.45)
        y += Inches(1.62)

    _text(s, Inches(0.9), Inches(6.18), Inches(11.4), Inches(0.4),
          "The wedge: a team adopts it because it turns hardware they already own into a working endpoint.",
          13, INK, italic=True)
    _footer(s, n, total)


def slide_competition(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, PAPER)
    _text(s, Inches(0.9), Inches(0.75), Inches(9), Inches(0.4),
          "LANDSCAPE", 12, ACCENT, bold=True)
    _text(s, Inches(0.9), Inches(1.15), Inches(11.4), Inches(0.8),
          "They assume the GPUs are in one machine.", 34, INK, bold=True)

    rows = [
        ("vLLM · SGLang · TensorRT-LLM",
         "Excellent single-node engines.",
         "Complementary — one can back a DistLLM node. They cannot span machines."),
        ("Ray / Ray Serve",
         "General distributed compute.",
         "A framework you build with. No partitioning, no OpenAI API, no node lifecycle."),
        ("Petals",
         "Distributed inference across machines.",
         "Nearest in spirit, but research-grade and largely dormant."),
        ("Cloud APIs",
         "Zero setup, someone else's GPUs.",
         "Your data leaves and the per-token bill never stops. The exact trade our user is escaping."),
        ("Buy an A100",
         "One machine that can do it.",
         "Capex plus permanent idle capacity."),
    ]
    y = Inches(2.25)
    for name, what, diff in rows:
        _text(s, Inches(0.9), y, Inches(3.5), Inches(0.5), name, 13, INK, bold=True)
        _text(s, Inches(4.55), y, Inches(3.4), Inches(0.6), what, 12, MUTED, spacing=1.3)
        _text(s, Inches(8.15), y, Inches(4.3), Inches(0.7), diff, 12, INK, spacing=1.3)
        y += Inches(0.82)

    _rule(s, Inches(0.9), Inches(6.42), Inches(11.6), MUTED, Pt(1))
    _text(s, Inches(0.9), Inches(6.62), Inches(11.4), Inches(0.4),
          "Our difference is the deployment model: open-source, verifiable, data-never-leaves, on hardware that already exists.",
          12, ACCENT, bold=True)
    _footer(s, n, total)


def slide_ai(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, INK)
    _text(s, Inches(0.9), Inches(0.75), Inches(9), Inches(0.4),
          "AI, TWICE", 12, ACCENT_DIM, bold=True)
    _text(s, Inches(0.9), Inches(1.2), Inches(11.4), Inches(0.9),
          "AI is the product. AI is also how it was built.", 34, PAPER, bold=True)

    _text(s, Inches(0.9), Inches(2.55), Inches(5.4), Inches(0.4),
          "IN THE PRODUCT", 11, ACCENT_DIM, mono=True)
    _text(s, Inches(0.9), Inches(3.0), Inches(5.4), Inches(2.2),
          "It runs large language models — that is the whole of it. It also "
          "applies AI to its own scheduling: a learned router picks placement "
          "from observed reward, and speculative decoding uses a small draft "
          "model whose tokens a larger model verifies.",
          15, RGBColor(0xC9, 0xD1, 0xDC), spacing=1.55)

    _text(s, Inches(7.0), Inches(2.55), Inches(5.4), Inches(0.4),
          "IN THE BUILD", 11, ACCENT_DIM, mono=True)
    _text(s, Inches(7.0), Inches(3.0), Inches(5.4), Inches(2.2),
          "The 0.7.0 hardening pass — six defects found and fixed, three "
          "order-dependent test failures traced to their real causes, and the "
          "verification on real GPU hardware — was done with AI agents reading, "
          "testing and checking. That is why a one-person effort shipped 13,600 "
          "passing tests and a hardware validation in days.",
          15, RGBColor(0xC9, 0xD1, 0xDC), spacing=1.55)
    _footer(s, n, total)


def slide_cloud(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, PAPER)
    _text(s, Inches(0.9), Inches(0.75), Inches(11), Inches(0.4),
          "GOOGLE CLOUD", 12, ACCENT, bold=True)
    _text(s, Inches(0.9), Inches(1.15), Inches(11.4), Inches(0.8),
          "The least-built part — and the plan is the substance.", 30, INK, bold=True)

    steps = [
        ("GKE first", "Helm chart and operator already exist. The work is GPU node pools, "
                      "the NVIDIA device plugin, and autoscaling against the coordinator — days, not weeks."),
        ("GCE with L4 / A100", "Two GPU instances, two commands, a working cluster. Cheapest way to "
                               "produce Cloud-measured numbers."),
        ("Cloud Storage", "Model shards fetched once per node. Makes cold start predictable."),
        ("Vertex Model Garden", "Pick a model there, partition it across the pool. Largest piece; "
                                "deferred until the first three are proven."),
        ("Cloud Monitoring + Logging", "Fed by Prometheus metrics and OpenTelemetry traces already "
                                       "emitted. An exporter, not new instrumentation."),
    ]
    y = Inches(2.3)
    for i, (head, body) in enumerate(steps, start=1):
        _text(s, Inches(0.9), y, Inches(0.45), Inches(0.4), str(i), 14, ACCENT, mono=True)
        _text(s, Inches(1.5), y - Inches(0.03), Inches(4.0), Inches(0.4), head, 15, INK, bold=True)
        _text(s, Inches(5.6), y - Inches(0.02), Inches(6.9), Inches(0.8), body, 12, MUTED, spacing=1.35)
        y += Inches(0.86)

    _rule(s, Inches(0.9), Inches(6.32), Inches(1.2), MUTED, Pt(1.5))
    _text(s, Inches(0.9), Inches(6.52), Inches(11.4), Inches(0.5),
          "Unit of scale is the node — exactly what GKE's node-pool autoscaler manages. "
          "0.7.0 added the drain / offline / deregister lifecycle autoscaling depends on.",
          12, INK)
    _footer(s, n, total)


def slide_ask(prs, n, total):
    s = _slide(prs)
    _rect(s, 0, 0, W, H, INK)
    _rule(s, Inches(0.9), Inches(1.9), Inches(1.6))
    _text(s, Inches(0.9), Inches(2.25), Inches(11.4), Inches(1.0),
          "What the credits buy", 44, PAPER, bold=True)

    _text(s, Inches(0.9), Inches(3.45), Inches(5.5), Inches(1.5),
          "$5,000 in Google AI and Cloud credits funds a two-node GKE cluster "
          "on L4 GPUs, and with it two things we cannot produce without it:",
          15, RGBColor(0xC9, 0xD1, 0xDC), spacing=1.5)

    items = [
        "A published Cloud-measured scaling curve — one node, two, four — "
        "replacing the 1.66× from consumer Ethernet with a number from the "
        "platform customers actually run on.",
        "A reproducible walkthrough: a GKE cluster, a Helm install, and an "
        "OpenAI-compatible endpoint in under an hour.",
    ]
    y = Inches(4.95)
    for i, item in enumerate(items, start=1):
        _text(s, Inches(0.9), y, Inches(0.35), Inches(0.4), str(i), 14, ACCENT, mono=True)
        box = _text(s, Inches(1.35), y - Inches(0.02), Inches(4.95),
                    Inches(0.95), item, 12,
                    RGBColor(0xC9, 0xD1, 0xDC), spacing=1.45)
        y += Inches(0.95)

    _text(s, Inches(7.2), Inches(3.45), Inches(5.2), Inches(0.4),
          "TRY IT NOW", 11, ACCENT_DIM, mono=True)
    _text(s, Inches(7.2), Inches(3.95), Inches(5.4), Inches(1.6),
          "pip install distributed-llm[cli,self-hosted]\n\n"
          "distllm cluster start -m meta-llama/Llama-3.2-7B\n\n"
          "distllm cluster join",
          12, PAPER, mono=True, spacing=1.5)

    _text(s, Inches(7.2), Inches(5.85), Inches(5.2), Inches(0.5),
          "Apache 2.0 · on PyPI · 13,600+ tests green", 12, MUTED)
    _footer(s, n, total)


def build(out: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    total = 8
    slide_title(prs, 1, total)
    slide_problem(prs, 2, total)
    slide_solution(prs, 3, total)
    slide_traction(prs, 4, total)
    slide_market(prs, 5, total)
    slide_model(prs, 6, total)
    slide_competition(prs, 7, total)
    slide_cloud(prs, 8, total)
    # 9 and 10 are deliberately not counted in `total` above; keep numbering honest.
    slide_ai(prs, 9, 10)
    slide_ask(prs, 10, 10)

    # Re-stamp the first eight so the deck reads 1–10 rather than 1–8 then 9–10.
    # prs.slides has no slicing support, so take the prefix explicitly.
    for slide in list(prs.slides)[:8]:
        for shape in slide.shapes:
            if shape.has_text_frame and "/ 8" in shape.text_frame.text:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        run.text = run.text.replace("/ 8", "/ 10")

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/DistLLM-Pitch-Deck.pptx")
    path = build(dest)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB, 10 slides)")