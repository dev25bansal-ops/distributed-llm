# DistLLM design plan

Direction dealt: **`product-performs` — The product performs** (seed 1076995885).
Sector dev-tool. Type **Archivo**. Palette **Charcoal and sage**.

## The idea sentence

> On this page, **DistLLM's layer-split across machines** becomes **the control you
> operate** — you assign model layers to real GPUs and the cluster forms in front of you.

Competitor test: substitute vLLM or Ollama — "assign model layers to real GPUs across
machines" is not what they do. The sentence fails for them, so the idea belongs to
this product, not the category.

## What I rejected

The design-system generator proposed a **dark OLED palette with acid green
`#22C55E` and a FAQ-landing pattern**. The site currently ships
**`#050505` + `#00e676`** — which is tell #2 on the frontend-design skill's list of
generic AI defaults (near-black with a single acid accent). Both are the same
decision, arrived at twice. I am not taking it.

Also rejected: the current 4,534 lines of CSS across 8 files, including
`modern.css`, `advanced.css` and `animations.css` — three files whose names
describe an aspiration rather than a component.

## Tokens

```css
--bg:        #1c1c1c   /* charcoal, not black */
--surface:   #262626
--ink:       #f1f1ef
--muted:     #9a9b96
--accent:    #8fa68e   /* sage — on the one button and the active control only */
--accent-ink:#1c1c1c
--field-1:   #c8c5ba   /* one section flips to concrete with dark text */
```

Deliberately **not** a tinted near-black (`#0B0B0B`), which is also a listed tell.
`#1c1c1c` is a real charcoal with visible warmth against the sage.

## Type

One family: **Archivo**, 400 display / 400 text / 600 for buttons.

| Step | Size | Use |
|---|---|---|
| small | 13–14px | labels, fragment text (true interface size) |
| body | 17–19px | running text, 1.5 leading |
| lead | 20–26px | the one-sentence standfirst |
| head | 32–64px | section headings |
| display | 4–6vw | hero headline only |

Display weight stays **400** even at large size — heavy display type is the loud
default; the direction asks for composed. Tracking −0.02em on display, 1.0 leading.
No all-caps, no tracked-out eyebrows, no monospace outside code.

## Layout

Hero: headline and one sentence in the upper-left third; the rest of the first
screen is a **working layer-assignment control** at real interface size.

```
┌──────────────────────────────────────────────┐
│ DistLLM                    docs  github  ⬛   │   plain bar, no glass
│                                              │
│ Pool the GPUs                                │   display, 2 lines max
│ you already own.                             │
│ One engine splits a model across them.        │   lead sentence
│ [ Read the docs ]                            │   one button
│                                              │
│ ┌──── layer assignment ────────────────────┐ │
│ │ [GPU 0  0–19 ]  [GPU 1  20–39]  [GPU 2 …]│ │   ← the live thing
│ │ ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ │
│ │ 70B model · 3 devices · 38.2/42 GB       │ │
│ └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

Sections below alternate on a 5/7 split, each with a different fragment shape so the
column never becomes a zigzag of equal blocks. One section pins a fragment while
three statements change it.

Margins: one pair, 24px mobile / 40px desktop, used everywhere.

## The live thing

**Layer assignment.** Drag or pick how many layers each detected GPU takes; the bar
re-partitions live and prints the real per-device VRAM. Real arithmetic from real
model sizes. Works with a thumb, keyboard-navigable (arrow keys move layers between
devices), and has a plain non-script fallback showing a precomputed split.

## Principles

1. **The product is the picture.** Fragments are built in HTML at real size with
   real numbers — never a fake dashboard of grey bars, never a device frame.
2. **One accent, one place.** Sage appears on the primary button and the active
   control. Nothing else is coloured.
3. **No decoration that isn't information.** No gradient, no shadow at rest, no
   hairline rule used as ornament, no icon-topped cards.
4. **The first screen answers what this is** in the headline, and what to do next
   in one button.

## Reuse constraint

The other 24 pages share `css/base.css`, `layout.css`, `components.css`. I will
replace the token layer and the component styles those pages depend on rather than
forking the site — a redesign that breaks 24 pages is not a redesign.
