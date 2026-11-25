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
"""The board as markdown tables, for the README and the page's sources."""

import anchors
import ranking
import scorefiles
import streamview

SHORT = {
    "suppression": "supp",
    "preservation": "pres",
    "fidelity": "fidel",
    "quality": "qual",
}


def _marked(stream, name):
    """The mark after a system's name: the streaming probe says yes."""
    return STREAM_MARK[streamview.streaming_cell(stream, name) == "yes"]


def _view(board):
    """Returns (score column, parts, axes inside) of one board's view.

    Without boards the composite's: its column, every part, no filter.
    """
    boards = ranking.BOARDS or {}
    if not boards:
        return "score", list(ranking.PARTS or {}), None
    shown = board or ranking.ranking_board()
    parts = list(boards[shown])
    return shown, parts, [a for p in parts for a in ranking.PARTS[p]]


def _markdown_rows(scores,
                   axis_scores,
                   comp,
                   rows,
                   tiers,
                   axes,
                   sfx,
                   stream=None,
                   precision=None,
                   board=None):
    """Returns one tier of one board's view as a markdown table.

    The board's score, the parts it reads and their axes, the axes scored
    beside every score, then the diagnostics.
    """
    column, parts, inside = _view(board)
    names = [
        a for a in ranking.axis_order(axes)
        if inside is None or a in inside or a in ranking.OUTSIDE
    ]
    head = (["#", "system", "precision", column] + parts +
            [SHORT.get(a, a) for a in names] + [
                "supp dB" + (" (0–8 kHz)" if sfx else ""),
                "pauses dB",
                "del %",
                "laughs lost %",
                "NISQA",
            ])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, n in enumerate(rows, 1):
        d = scores[n]
        scored = anchors.part_scores(axis_scores[n], ranking.PARTS)
        # a view without the recordings has none
        lost = scorefiles.get(d, "nonverbal@+0", "nv_lost_pct")
        value = (ranking.board_scores(axis_scores[n])[column]
                 if ranking.BOARDS else comp[n])
        cells = ([
            str(i),
            tiers[n][1] + _marked(stream, n),
            (precision or {}).get(n, "–"),
            f"**{value:.3f}**",
        ] + [f"{scored[k]:.2f}" for k in parts] + [
            f"{axis_scores[n].get(k, ranking.NAN):.2f}" for k in names
        ] + [
            f"{scorefiles.get(d, 'babble@+4', 'babble_resid_db' + sfx):.1f}",
            f"{scorefiles.get(d, '_real_cleanup', 'pause_level_db'):.1f}",
            f"{scorefiles.get(d, '_deletion', 'pct'):.1f}",
            "–" if lost != lost else f"{lost:.1f}",
            f"{scorefiles.get(d, 'babble@+4', 'nisqa_mos'):.2f}",
        ])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_table(scores,
                   axis_scores,
                   comp,
                   order,
                   tiers,
                   stream=None,
                   precision=None):
    """Full-band tier as a GitHub-flavoured markdown table: the view of the
    board that orders the table."""
    return _with_streaming_note(
        _markdown_rows(scores, axis_scores, comp, order["full"], tiers,
                       ranking.AXES_FULL, "", stream, precision), stream,
        order["full"])


def board_lead(board):
    """One line saying which systems a board ranks, and by what."""
    if board == ranking.restoration_board():
        return (f"**{board.capitalize()}**: the systems that restore above the"
                f" input's band (more than {ranking.ASSIGN['above_db']:g} dB"
                f" on the band-limited rows), ordered by their {board} score.")
    return (f"**{board.capitalize()}**: the systems that add nothing above the"
            f" input's band, ordered by their {board} score.")


def markdown_boards(scores,
                    axis_scores,
                    comp,
                    order,
                    tiers,
                    stream=None,
                    precision=None):
    """Returns every board's view of the full-band tier (for the README), each
    after its lead, with the streaming note once; without boards, the one
    table."""
    if not ranking.BOARDS:
        return markdown_table(scores, axis_scores, comp, order, tiers, stream,
                              precision)
    blocks, shown = [], []
    for board in ranking.BOARDS:
        rows = (order["full"]
                if board == ranking.ranking_board() else order.get(board, []))
        blocks.append(
            board_lead(board) + "\n\n" +
            _markdown_rows(scores, axis_scores, comp, rows, tiers,
                           ranking.AXES_FULL, "", stream, precision, board))
        shown += rows
    return _with_streaming_note("\n\n".join(blocks), stream, shown)


def _with_streaming_note(table, stream, rows):
    """The table, with the note on the mark when any row carries it."""
    if any(streamview.streaming_cell(stream, n) == "yes" for n in rows):
        return table + "\n\n" + STREAMING_NOTE
    return table


# the mark after a system's name: the streaming probe says yes
STREAM_MARK = {True: "\\*", False: ""}
STREAMING_NOTE = "\\* streaming: " + streamview.rule_sentence() + "."

NB_LEAD = ("The 16 kHz tier, scored inside 0–8 kHz on the same items and not"
           " comparable to the full-band tier's boards above: suppression,"
           " speaker similarity and passthrough are read on the band-limited"
           " signals, and there is no"
           " high-band term.")


def markdown_table_nb(scores,
                      axis_scores,
                      comp,
                      order,
                      tiers,
                      stream=None,
                      precision=None):
    """Returns the 16 kHz tier as a markdown table, empty without entrants."""
    rows = order.get("nb", [])
    if not rows:
        return ""
    return _with_streaming_note(
        _markdown_rows(scores, axis_scores, comp, rows, tiers, ranking.AXES_NB,
                       "_nb", stream, precision), stream, rows)


def markdown_outside(axis_scores, order, tiers, axes=None):
    """Returns one markdown table per axis, in the axis's own order.

    The axes scored beside the composite (so what the composite does not read is
    published all the same) and any named in ``axes``, for a section that lists
    the board by one axis.
    """
    blocks = []
    for axis in list(ranking.OUTSIDE) + [
            a for a in axes or () if a not in ranking.OUTSIDE
    ]:
        # every ranked full-band system, whichever board ranks it
        ranked = [n for key in order if key != "nb" for n in order[key]]
        rows = [
            n for n in ranked
            if axis_scores[n].get(axis, ranking.NAN) == axis_scores[n].get(
                axis, ranking.NAN)
        ]
        rows.sort(key=lambda n, axis=axis: -axis_scores[n][axis])
        lines = [f"| # | system | {axis} |", "|---|---|---|"]
        lines += [
            f"| {i} | {tiers[n][1]} | {axis_scores[n][axis]:.2f} |"
            for i, n in enumerate(rows, 1)
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
