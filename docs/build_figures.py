# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Draws the README's figures as SVG, in a light and a dark variant each.

  docs/assets/board-{light,dark}.svg         the denoising board's scores,
                                             one bar per system, read from the
                                             board that
                                             bench/scripts/leaderboard.py
                                             --json-out writes: numbers are
                                             generated, never typed
  docs/assets/board-restoration-{light,dark}.svg  the restoration board's,
                                             the same way
  docs/assets/architecture-{light,dark}.svg  the network, from the STFT to the
                                             isolated voice, with the two
                                             checkpoints' serving modes

    python docs/build_figures.py [--leaderboard <board.json>]
                                 [--out docs/assets]

Without --leaderboard only the architecture figures are drawn.

The README shows each pair through a <picture> element, so GitHub picks the
variant that matches the reader's theme.
"""

import argparse
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Instrument Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Geist Mono, SFMono-Regular, Menlo, Consolas, monospace"
# The brand's paper and dark layouts; the two bar hues passed the colour checks
# (lightness band, chroma, colour-vision separation, contrast) in each theme.
PALETTES = {
    # GitHub's light and dark canvases; the text and bar colours are the
    # tokens of inworld.ai: its ink scale, its teal accent, its gray scale.
    "light": {
        "surface": "#FFFFFF",
        "text": "#151613",
        "body": "#57534B",
        "muted": "#69665E",
        "rule": "#D7D2CC",
        "ours": "#02793D",
        "others": "#9A9089",
    },
    "dark": {
        "surface": "#0D1117",
        "text": "#EEEAE0",
        "body": "#B4AFA3",
        "muted": "#89857C",
        "rule": "#30363D",
        "ours": "#1A9B58",
        "others": "#9A9089",
    },
}
OURS_PREFIX = "voice_isolator"
BAR = 22  # px; thinner than the slot, the slot's rest is air
SLOT = 36


def _e(text: str) -> str:
    return html.escape(str(text), quote=True)


def _text(x, y, s, size, fill, weight="normal", anchor="start", mono=False):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else FONT}"'
        f' font-size="{size}" font-weight="{weight}" fill="{fill}"'
        f' text-anchor="{anchor}">{_e(s)}</text>')


def board_rows(board: dict, which: str | None = None) -> list:
    """The full-band tier in board order: (short label, score, ours).

    The score is the composite, or the score of the board ``which`` when the
    registry publishes boards side by side (then in that board's own order).
    The short label is the roster's display name before its parenthesis; the
    table carries the full names.
    """
    order = board["order"]
    boards = board.get("boards") or {}
    first = board.get("ranking_board") or (next(iter(boards))
                                           if boards else None)
    if which and which != first and which in order:
        # the board's own systems, in its order
        names = list(order[which])
    else:
        names = list(order["full"])
        if which and which != first:
            # a board JSON without that board's order: every system by that
            # score
            names = sorted(
                (n for n in names
                 if isinstance((board["systems"][n].get("board_scores") or {}
                               ).get(which), (int, float))),
                key=lambda n: -board["systems"][n]["board_scores"][which])
    rows = []
    for name in names:
        system = board["systems"][name]
        value = (system.get("board_scores") or
                 {}).get(which) if which else system.get("composite")
        if not isinstance(value, (int, float)):
            continue
        label = system.get("label", name).split(" (")[0]
        # the mark after a name: the streaming probe says yes
        if (system.get("streaming") or {}).get("verdict") == "yes":
            label += "*"
        rows.append((label, float(value), name.startswith(OURS_PREFIX)))
    return rows


def board_svg(board: dict, mode: str, which: str | None = None) -> str:
    """One horizontal bar per ranked system, the score at its tip: the
    composite, or the score of the board ``which``."""
    p = PALETTES[mode]
    rows = board_rows(board, which)
    what = f"{which.capitalize()} score" if which else "Composite score"
    left, right, top = 330.0, 84.0, 74.0
    width = 1040.0
    plot_w = width - left - right
    height = top + SLOT * len(rows) + 66
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}"'
        f' height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}"'
        f' role="img" aria-label="Inworld VIBE full-band tier, {what.lower()}'
        f' per system">',
        f'<rect width="100%" height="100%" fill="{p["surface"]}"/>',
        _text(
            28, 36, f"{what}, higher is better. Gaps under "
            f"{board['tie_bands']['composite']:g} are ties.", 16, p["body"]),
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot_w * tick
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}"'
                   f' y2="{top + SLOT * len(rows)}" stroke="{p["rule"]}"'
                   ' stroke-width="1"/>')
        out.append(
            _text(x,
                  top + SLOT * len(rows) + 18,
                  f"{tick:g}",
                  13.5,
                  p["muted"],
                  anchor="middle",
                  mono=True))
    for i, (label, value, ours) in enumerate(rows):
        y = top + SLOT * i + (SLOT - BAR) / 2
        w = max(plot_w * value, 4.0)
        fill = p["ours"] if ours else p["others"]
        # square at the baseline, 4 px rounded at the data end
        out.append(
            f'<path d="M{left:.1f},{y:.1f} h{w - 4:.1f} a4,4 0 0 1 4,4'
            f' v{BAR - 8} a4,4 0 0 1 -4,4 h-{w - 4:.1f} z" fill="{fill}"/>')
        out.append(
            _text(left - 14, y + BAR - 5, label, 16, p["text"], anchor="end"))
        out.append(
            _text(left + w + 8,
                  y + BAR - 5,
                  f"{value:.3f}",
                  15,
                  p["text"],
                  "600",
                  mono=True))
    if any(label.endswith("*") for label, _, _ in rows):
        # the one note: the mark after a name means streaming
        out.append(_text(28, height - 22, "* streaming", 13.5, p["muted"]))
    out.append(
        _text(width - 28,
              height - 22,
              f"Inworld VIBE v0 · set build {board['set_id'][:16]}…",
              13,
              p["muted"],
              anchor="end",
              mono=True))
    out.append("</svg>")
    return "\n".join(out)


ARCH_DESC = (
    "Input x(t), 48 kHz mono. STFT with N = 2048 and hop H = 480 (10 ms) gives"
    " X in C^(F x T), F = 1025 bins. Band split into B = 46 bands, 100 Hz to"
    " 2 kHz wide, each projected to D = 512, gives Z in R^(B x T x D). Seven"
    " dual-path blocks, alternating A, M, A, M, A, M, A: block A is time"
    " attention (rotary positions, exclusive self-attention, 8 heads) then"
    " band attention; block M is time Mamba (state 16, expand 2, conv 4) then"
    " band Mamba; the time path runs along time within each band, the band"
    " path across bands within each frame; every sub-layer has a residual"
    " connection. Per-band linear heads produce a bounded complex mask M and"
    " a bounded residual R, both in C^(F x T). Output spectrum"
    " X_hat = X + M (.) X + R, from the input spectrum X through a skip"
    " connection; iSTFT gives the isolated voice s_hat(t). The realtime"
    " checkpoint runs the same blocks causally with a look-ahead of 0, 2 or 4"
    " frames as a learned embedding; the offline checkpoint is two-sided and"
    " renders 8 s chunks.")
# The figure's words, in a small markup: *x* italic, **x** bold, ^{x} raised.
ARCH_STAGES = (  # title, parameter lines, role
    ("STFT", ("*N* = 2048", "*H* = 480 (10 ms)", "*F* = 1025 bins"), "dsp"),
    ("Band split", ("*B* = 46 bands", "100 Hz–2 kHz wide",
                    "linear to *D* = 512"), "net"),
    ("7 dual-path blocks", ("A M A M A M A", "attention, Mamba"), "net"),
    ("Per-band heads", ("mask *M*, bounded", "residual *R*, |*R*| ≤ 0.5"),
     "head"),
    ("iSTFT", (), "dsp"),
)
ARCH_SHAPES = ("*X* ∈ ℂ^{*F*×*T*}", "*Z* ∈ ℝ^{*B*×*T*×*D*}",
               "*Z*′ ∈ ℝ^{*B*×*T*×*D*}")
ARCH_SUM = "*X\u0302* = *X* + *M* ⊙ *X* + *R*"
ARCH_SKIP = "skip: the input spectrum *X*"
ARCH_BLOCK_TYPES = (  # tag, kind, (sub-layer, its axis) twice, settings
    ("A", "attention", (("time attention", "along *T*, in each band"),
                        ("band attention", "across *B*, in each frame")),
     "rotary positions · exclusive self-attention · 8 heads"),
    ("M", "Mamba", (("time Mamba", "along *T*, in each band"),
                    ("band Mamba", "across *B*, in each frame")),
     "selective scan · state 16 · expand 2 · conv 4"),
)
ARCH_NOTES = (
    ("Realtime checkpoint", "the same blocks run causally, with a look-ahead"
     " *ℓ* ∈ {0, 2, 4} frames as a learned embedding."),
    ("Offline checkpoint", "two-sided, 8 s chunks."),
)
ARCH_WIDTH = 1040  # the board's width, so both figures display at one scale
ARCH_MARGIN = 24
# GitHub's canvases. Fills say a stage's role: fixed signal processing (dsp),
# learned layers (net), the heads. Body text keeps at least 7.5:1 contrast on
# every fill, captions at least 5.5:1.
ARCH_PALETTES = {
    "light": {
        "bg": "#FFFFFF",
        "ink": "#1F2328",
        "ink2": "#3D444D",
        "ink3": "#59636E",
        "line": "#59636E",
        "panel": "#F6F8FA",
        "dsp": "#DCE8F6",
        "net": "#D7EEDF",
        "head": "#F8E5C2",
    },
    "dark": {
        "bg": "#0D1117",
        "ink": "#F0F6FC",
        "ink2": "#D1D7E0",
        "ink3": "#A6AFBA",
        "line": "#A6AFBA",
        "panel": "#161B22",
        "dsp": "#1C3554",
        "net": "#17442F",
        "head": "#4A3511",
    },
}
# Advance widths in 1/1000 em of " " to "~", regular and bold: for each glyph
# the wider of Helvetica and Helvetica Neue (Arial shares Helvetica's
# metrics), so a box sized from them holds its text in whichever font of the
# stack the reader has.
_ADVANCE = {
    False:
        (278, 278, 426, 556, 556, 1000, 667, 278, 333, 333, 389, 600, 278, 389,
         278, 333, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278,
         600, 600, 600, 556, 1015, 667, 685, 722, 722, 667, 611, 778, 722, 278,
         519, 667, 556, 871, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944,
         667, 667, 611, 278, 333, 278, 600, 556, 222, 556, 593, 537, 593, 556,
         296, 574, 556, 222, 222, 519, 222, 853, 556, 574, 593, 593, 333, 500,
         315, 556, 500, 759, 518, 500, 500, 334, 260, 334, 600),
    True: (278, 333, 481, 556, 556, 1000, 722, 278, 333, 333, 407, 600, 278,
           407, 278, 389, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333,
           333, 600, 600, 600, 611, 975, 722, 722, 741, 741, 667, 611, 778, 741,
           296, 556, 722, 611, 907, 741, 778, 667, 778, 722, 667, 611, 741, 667,
           944, 667, 667, 648, 333, 389, 333, 600, 556, 278, 574, 611, 574, 611,
           574, 352, 611, 611, 278, 278, 574, 278, 907, 611, 611, 611, 611, 389,
           556, 370, 611, 556, 815, 556, 556, 519, 389, 280, 389, 600),
}
_ADVANCE_SYMBOLS = {
    "×": 600,
    "–": 556,
    "·": 278,
    "′": 333,
    "≤": 600,
    "∈": 667,
    "ℂ": 722,
    "ℝ": 722,
    "⊙": 778,
    "ℓ": 444,
    "ŝ": 556,
    "\u0302": 0,  # the hat of X-hat, a combining mark
}
_SCRIPT = 0.72  # a superscript's size relative to its text
_LINE_PITCH = 21


def _runs(markup: str) -> list:
    """Splits figure markup into (text, italic, bold, raised) runs."""
    runs, text, italic, bold, raised, i = [], "", False, False, False, 0
    while i < len(markup):
        if markup.startswith("**", i):
            runs.append((text, italic, bold, raised))
            text, bold, i = "", not bold, i + 2
        elif markup[i] == "*":
            runs.append((text, italic, bold, raised))
            text, italic, i = "", not italic, i + 1
        elif markup.startswith("^{", i):
            runs.append((text, italic, bold, raised))
            text, raised, i = "", True, i + 2
        elif markup[i] == "}" and raised:
            runs.append((text, italic, bold, raised))
            text, raised, i = "", False, i + 1
        else:
            text, i = text + markup[i], i + 1
    runs.append((text, italic, bold, raised))
    return [run for run in runs if run[0]]


def _text_width(markup: str, size: float, bold: bool = False) -> float:
    """The width ``markup`` takes at ``size`` px."""
    total = 0.0
    for text, _, run_bold, raised in _runs(markup):
        advances = _ADVANCE[bold != run_bold]
        em = sum(
            advances[ord(ch) -
                     32] if " " <= ch <= "~" else _ADVANCE_SYMBOLS.get(ch, 600)
            for ch in text)
        total += em / 1000 * size * (_SCRIPT if raised else 1)
    return total


def _rich(x, y, markup, size, fill, bold=False, anchor="middle"):
    """A text element from markup: italic, bold and raised runs as tspans."""
    spans, shift = [], 0.0
    for text, italic, run_bold, raised in _runs(markup):
        attrs = []
        target = -0.4 * size if raised else 0.0
        if target != shift:
            attrs.append(f'dy="{target - shift:.2f}"')
            shift = target
        if raised:
            attrs.append(f'font-size="{size * _SCRIPT:.2f}"')
        if italic:
            attrs.append('font-style="italic"')
        if run_bold:  # **x** flips the element's weight
            attrs.append(f'font-weight="{400 if bold else 600}"')
        body = html.escape(text, quote=False)
        spans.append(
            f"<tspan {' '.join(attrs)}>{body}</tspan>" if attrs else body)
    weight = ' font-weight="600"' if bold else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}"'
            f' font-size="{size}" fill="{fill}"{weight}'
            f' text-anchor="{anchor}">{"".join(spans)}</text>')


def _rounded(points, radius):
    """Path data through ``points``, the corners rounded by ``radius``."""
    d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for (x0, y0), (x1, y1), (x2, y2) in zip(points, points[1:], points[2:]):
        before = math.hypot(x1 - x0, y1 - y0)
        after = math.hypot(x2 - x1, y2 - y1)
        r = min(radius, before / 2, after / 2)
        ax, ay = x1 - (x1 - x0) / before * r, y1 - (y1 - y0) / before * r
        bx, by = x1 + (x2 - x1) / after * r, y1 + (y2 - y1) / after * r
        d += f" L{ax:.1f},{ay:.1f} Q{x1:.1f},{y1:.1f} {bx:.1f},{by:.1f}"
    return d + f" L{points[-1][0]:.1f},{points[-1][1]:.1f}"


def _arrow(points, color, dash=None, radius=10.0):
    """A line through ``points`` whose last point is the tip of its head."""
    length, width = 8.5, 6.5
    (x1, y1), (x2, y2) = points[-2], points[-1]
    cos, sin = math.cos(math.atan2(y2 - y1, x2 - x1)), math.sin(
        math.atan2(y2 - y1, x2 - x1))
    # the line stops inside the head, so its end never shows past the tip
    body = list(
        points[:-1]) + [(x2 - 0.62 * length * cos, y2 - 0.62 * length * sin)]
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    head = [(0, 0), (-length, width / 2), (-0.72 * length, 0),
            (-length, -width / 2)]
    tip = " L".join(f"{x2 + u * cos - v * sin:.1f},{y2 + u * sin + v * cos:.1f}"
                    for u, v in head)
    return (f'<path d="{_rounded(body, radius)}" fill="none" stroke="{color}"'
            f' stroke-width="1.4" stroke-linecap="round"'
            f' stroke-linejoin="round"{dashes}/>\n'
            f'<path d="M{tip} Z" fill="{color}"/>')


def _sum_node(x, y, r, palette):
    """The sum: a circle with a plus, drawn rather than typeset."""
    k = r * 0.52
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{palette["bg"]}"'
            f' stroke="{palette["line"]}" stroke-width="1.4"/>\n'
            f'<path d="M{x - k:.1f},{y:.1f} H{x + k:.1f} M{x:.1f},{y - k:.1f}'
            f' V{y + k:.1f}" stroke="{palette["line"]}" stroke-width="1.4"'
            ' stroke-linecap="round"/>')


def _dot(x, y, r, palette):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"'
            f' fill="{palette["line"]}"/>')


def _stage(x, top, w, h, title, lines, fill, palette):
    """A filled box with its title and parameters centred in it."""
    out = [
        f'<rect x="{x:.1f}" y="{top}" width="{w:.1f}" height="{h}" rx="8"'
        f' fill="{fill}"/>'
    ]
    y = top + h / 2 - _LINE_PITCH * len(lines) / 2 + 5
    out.append(_rich(x + w / 2, y, title, 17.5, palette["ink"], bold=True))
    for k, line in enumerate(lines):
        out.append(
            _rich(x + w / 2, y + 24 + _LINE_PITCH * k, line, 15,
                  palette["ink2"]))
    return out


def _block_rows(x, y, palette):
    """The two block types as rows of residual sub-layers; (x, y) is the left
    edge and the first row's centre line."""
    out = []
    box_h = 52
    box_w = max(
        max(_text_width(name, 15.5, bold=True), _text_width(axis, 13.5))
        for _, _, sublayers, _ in ARCH_BLOCK_TYPES
        for name, axis in sublayers) + 30
    for row, (tag, kind, sublayers, settings) in enumerate(ARCH_BLOCK_TYPES):
        mid = y + 78 * row
        out.append(_rich(x + 18, mid + 2, tag, 21, palette["ink"], bold=True))
        out.append(_rich(x + 18, mid + 20, kind, 12.5, palette["ink3"]))
        bx = x + 84
        out.append(_arrow([(x + 48, mid), (bx, mid)], palette["line"]))
        for k, (name, axis) in enumerate(sublayers):
            out.append(f'<rect x="{bx:.1f}" y="{mid - box_h / 2:.1f}"'
                       f' width="{box_w:.1f}" height="{box_h}" rx="7"'
                       f' fill="{palette["net"]}"/>')
            out.append(
                _rich(bx + box_w / 2,
                      mid - 3,
                      name,
                      15.5,
                      palette["ink"],
                      bold=True))
            out.append(
                _rich(bx + box_w / 2, mid + 16, axis, 13.5, palette["ink2"]))
            px = bx + box_w + 28
            out.append(
                _arrow([(bx + box_w, mid), (px - 11, mid)], palette["line"]))
            # the residual: the sub-layer's input goes around it to the sum
            fork = bx - 18
            over = mid - box_h / 2 - 13
            out.append(_dot(fork, mid, 2.6, palette))
            out.append(
                _arrow([(fork, mid), (fork, over), (px, over), (px, mid - 11)],
                       palette["line"],
                       radius=7))
            out.append(_sum_node(px, mid, 11, palette))
            nxt = px + 11 + (34 if k == 0 else 30)
            out.append(_arrow([(px + 11, mid), (nxt, mid)], palette["line"]))
            bx = nxt
        out.append(
            _rich(bx + 12,
                  mid + 5,
                  settings,
                  14.5,
                  palette["ink2"],
                  anchor="start"))
    return out


def architecture_svg(mode: str) -> str:
    """The network as a layer diagram: the pipeline with its tensor shapes,
    then the insides of the two block types, then the two checkpoints."""
    p = ARCH_PALETTES[mode]
    left, right = ARCH_MARGIN, ARCH_WIDTH - ARCH_MARGIN
    top, h, stub, sum_r = 66, 116, 54, 13
    mid = top + h / 2
    widths = [
        _text_width(title, 17.5, bold=True)
        if not lines else max([_text_width(title, 17.5, bold=True)] +
                              [_text_width(line, 15)
                               for line in lines])
        for title, lines, _ in ARCH_STAGES
    ]
    widths = [w + 24 for w in widths]
    widths[-1] = max(widths[-1], 72)
    # input stub, four stages, the sum, the iSTFT, output stub; the stubs sit
    # flush against their stages and the other gaps share what is left
    items = [stub] + widths[:4] + [2 * sum_r, widths[4], stub]
    gap = (right - left - sum(items)) / 5
    xs, x = [], left
    for i, w in enumerate(items):
        xs.append(x)
        x += w + (0 if i in (0, 6) else gap)
    out = []
    for x0, symbol, dx in ((xs[0], "*x*(*t*)", -4), (xs[7], "*ŝ*(*t*)", 2)):
        out.append(_arrow([(x0, mid), (x0 + stub, mid)], p["line"]))
        out.append(_rich(x0 + stub / 2 + dx, mid - 10, symbol, 19, p["ink"]))
        out.append(
            _rich(x0 + stub / 2 + dx, mid + 24, "48 kHz", 13.5, p["ink3"]))
    for (title, lines, role), j in zip(ARCH_STAGES, (1, 2, 3, 4, 6)):
        out.extend(_stage(xs[j], top, items[j], h, title, lines, p[role], p))
    sx = xs[5] + sum_r
    ends = [(xs[a] + items[a], xs[a + 1]) for a in range(1, 6)]
    ends[3] = (ends[3][0], sx - sum_r)
    ends[4] = (sx + sum_r, ends[4][1])
    for k, (x1, x2) in enumerate(ends):
        out.append(_arrow([(x1 + 1, mid), (x2 - 1, mid)], p["line"]))
        if k < len(ARCH_SHAPES):
            out.append(
                _rich((x1 + x2) / 2, top - 16, ARCH_SHAPES[k], 16, p["ink"]))
    out.append(_sum_node(sx, mid, sum_r, p))
    out.append(_rich(sx, top - 16, ARCH_SUM, 16, p["ink"]))
    # the skip: X leaves the STFT's arrow and reaches the sum unchanged
    fork = sum(ends[0]) / 2
    under = top + h + 24
    out.append(_dot(fork, mid, 3.4, p))
    out.append(
        _arrow([(fork, mid), (fork, under), (sx, under), (sx, mid + sum_r)],
               p["line"],
               dash="5 4"))
    out.append(_rich((fork + sx) / 2, under + 20, ARCH_SKIP, 13.5, p["ink3"]))
    # the block, opened
    panel = under + 50
    out.append(f'<rect x="{left}" y="{panel}" width="{right - left}"'
               f' height="214" rx="12" fill="{p["panel"]}"/>')
    heading = "Inside each dual-path block"
    out.append(
        _rich(left + 22, panel + 32, heading, 15.5, p["ink"], True, "start"))
    out.append(
        _rich(left + 30 + _text_width(heading, 15.5, bold=True),
              panel + 32,
              "· every sub-layer is residual",
              15.5,
              p["ink3"],
              anchor="start"))
    out.extend(_block_rows(left + 22, panel + 98, p))
    y = panel + 214 + 36
    for k, (lead, rest) in enumerate(ARCH_NOTES):
        out.append(
            _rich(left,
                  y + 23 * k,
                  f"**{lead}:** {rest}",
                  14.5,
                  p["ink2"],
                  anchor="start"))
    height = y + 23 * (len(ARCH_NOTES) - 1) + 22
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ARCH_WIDTH}"'
        f' height="{height:.0f}" viewBox="0 0 {ARCH_WIDTH} {height:.0f}"'
        ' role="img" aria-label="Inworld Voice Isolator architecture">',
        f"<desc>{html.escape(ARCH_DESC)}</desc>",
        f'<rect width="100%" height="100%" fill="{p["bg"]}"/>',
    ] + out + ["</svg>"])


def write(path: str, text: str) -> None:
    """Writes one figure and says so."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--leaderboard",
        default=None,
        help=("the board bench/scripts/leaderboard.py --json-out wrote; without"
              " it only the architecture figures are drawn"),
    )
    ap.add_argument("--out", default=os.path.join(HERE, "assets"))
    a = ap.parse_args()
    board = None
    if a.leaderboard:
        with open(a.leaderboard, encoding="utf-8") as fh:
            board = json.load(fh)
    os.makedirs(a.out, exist_ok=True)
    boards = list((board or {}).get("boards") or {})
    first = (board or
             {}).get("ranking_board") or (boards[0] if boards else None)
    for mode in ("light", "dark"):
        if board is not None:
            # the first board is the headline chart; every other board gets
            # its own
            write(os.path.join(a.out, f"board-{mode}.svg"),
                  board_svg(board, mode, first))
            for other in boards:
                if other != first:
                    write(os.path.join(a.out, f"board-{other}-{mode}.svg"),
                          board_svg(board, mode, other))
        write(os.path.join(a.out, f"architecture-{mode}.svg"),
              architecture_svg(mode))


if __name__ == "__main__":
    main()
