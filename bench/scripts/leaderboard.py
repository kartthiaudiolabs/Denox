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
"""Ranked leaderboard for Inworld VIBE, the voice-isolation benchmark.

Ranking is the opinionated part, so the rules are stated here and printed with
the table rather than buried in a weighted sum nobody can audit.

TIERS. The goal is a voice isolator that is excellent across the whole 0-48 kHz
band. A 16 kHz-native model emits nothing above 8 kHz, so it cannot meet that
goal however well it suppresses; those are ranked in a separate tier. Plain
DSP, the music-separation reference and the unprocessed input are reference
rows.

AXES. Eleven (bench/axes.json states the axes, their rows, the parts, the
boards and the rule that assigns each system a board), because the target is a
model with no weak side and an axis added on one side must not take weight
from another:
  removes
    suppression   babble_supp_db @+4 / @+0 / @-5, si_sdr on ambience, stationary
                  noise and traffic @-5
    music         si_sdr on music@-6
    cleanup       what is left in the pauses of the real recordings and the
                  same-room floors
    events        the same-room transient events left
    voices        si_sdr on the competing talker @+5
  keeps
    preservation  real-file deleted-rate, voice-kept P10 on babble@+4, the vocal
                  sounds outside speech lost under a bed (scorekept.py) and the
                  voice's level while same-room noise goes
    fidelity      harm a listener could notice on clean speech: the quiet tenth
                  of the voice, level, the 8-24 kHz band (invented high
                  frequency costs as much as lost), speaker similarity, NISQA
                  and the magnitude spectrum, each against the untouched
                  recording, and the vocal sounds outside speech lost on clean
                  recordings.
                  Clean si_sdr is printed, not scored: it measures identity of
                  the waveform, not harm.
  sounds
    timbre        spk_sim over twelve noise families at a moderate level; gaps
                  under its tie band are ties
    quality       NISQA v2 MOS on babble@+4 and on the real recordings
  restores
    room          recordings in real rooms against the dry voice: spk_sim and
                  NISQA. A system that leaves the room as recorded scores zero.
  beside both scores
    band          the energy above a band-limited input's band and the speaker
                  kept while it does

BOARDS. Two scores, each the mean of its parts: denoising (removes, keeps,
sounds) and restoration (restores). A system that restores above the input's
band is ranked on restoration, every other system on denoising; every system is
scored on both.

NORMALISATION. Each sub-metric closes the gap between versioned, fixed anchors
derived from INPUT, REF, or declared constants. The calibration and primary
deletion cohort belong to the set, never to the competitor roster. Sub-metrics
are averaged within an axis, axes within a part, parts into the score.

NO HARD GATE. Deletion rides in the preservation axis, is printed as its own
column, and a sensitivity line shows the order under a strict deletion gate.

Systems not listed in roster.py can be added with --tiers extra.json
({"name": ["full"|"nb"|"ref", "label"], ...}).
"""

import argparse
import dataclasses
import json
import math
import os

import anchors
import common
import contracts
import ranking
import roster
import scorefiles
import scorestream
import streamview
import tables


@dataclasses.dataclass
class Board:
    """One ranked board: the scores, how they rank, and what is said beside."""
    scores: dict
    tiers: dict
    anchored: dict
    axis_scores: dict
    comp: dict
    order: dict
    stream: dict
    partial: dict
    grouped: dict
    missing: dict
    precision: dict


WIDTH = 38  # the system column of the printed board


def parse_args() -> argparse.Namespace:
    """Returns the command line."""
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=os.environ.get("BENCH_RESULTS",
                               os.path.join(os.path.dirname(here), "results")),
    )
    ap.add_argument("--json-out", default="")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument(
        "--outside-out",
        default="",
        help=(
            "markdown tables of the axes scored beside the composite, in their"
            " order"),
    )
    ap.add_argument(
        "--axis-tables",
        default="",
        help=("comma-separated axes to add to --outside-out in their own order "
              "(e.g. room, the Real rooms section)"),
    )
    ap.add_argument(
        "--streaming-out",
        default="",
        help="markdown table of the streaming probe behind the yes/no column",
    )
    ap.add_argument("--tiers",
                    default="",
                    help="JSON file with extra {name: [tier, label]}")
    ap.add_argument(
        "--anchors",
        default=str(anchors.DEFAULT),
        help="versioned anchor definitions",
    )
    ap.add_argument(
        "--calibration",
        default="",
        help="frozen set calibration (default: results/anchors.resolved.json)",
    )
    ap.add_argument("--strict-delete", type=float, default=3.0)
    ap.add_argument("--title", default="INWORLD VIBE BOARDS")
    ap.add_argument(
        "--view",
        default="all",
        choices=sorted(ranking.AXES_VIEWS) + list(common.PANELS),
        help=("all: every condition; core: the multilingual half only "
              "(rank results scored with BENCH_SUBSET=core)"),
    )
    return ap.parse_args()


def board_scores(a: argparse.Namespace, tiers: dict) -> tuple[dict, dict]:
    """Returns the scores a board may rank and its frozen calibration.

    A result nobody registered and a filtered scoring pass are left out, each
    with a line saying so.

    Raises:
        SystemExit: The scores mix scorer implementations, renders, subsets or
            builds of the set, or are sealed.
    """
    scores = scorefiles.load(a.results)
    unregistered = sorted(n for n in scores if n not in tiers)
    # a result file nobody registered: a stray sweep, never a row of a board
    if unregistered:
        print("IGNORED (no tier registered; pass --tiers to rank them): " +
              ", ".join(unregistered))
        scores = {n: d for n, d in scores.items() if n not in unregistered}
    filtered = scorefiles.partial_passes(scores)
    if filtered:
        print("NOT RANKED - a filtered scoring pass, never a board row: " +
              ", ".join(f"{n} ({v})" for n, v in sorted(filtered.items())))
        scores = {n: d for n, d in scores.items() if n not in filtered}
    problem = scorefiles.scorer_problem(scores) or scorefiles.render_problem(
        scores, a.results)
    if problem:
        raise SystemExit(problem)
    set_ids = {n: scores[n].get("_set_id") for n in scores}
    subsets = {n: scores[n].get("_subset", "all") for n in scores}
    if len(set(subsets.values())) > 1:
        raise SystemExit("systems were scored on different subsets of the set: "
                         f"{subsets}; one results directory holds one subset.")
    if a.view == "core" and set(subsets.values()) != {"core"}:
        raise SystemExit(
            "--view core ranks results scored with BENCH_SUBSET=core; these"
            f" were scored on {set(subsets.values())}")
    if len({v for v in set_ids.values() if v}) > 1:
        raise SystemExit(
            "systems were scored on different eval sets (set_id):"
            f" { {n: (v or 'none')[:12] for n, v in set_ids.items()} };"
            " re-score before ranking.")
    anchored = anchors.load(
        a.calibration or os.path.join(a.results, anchors.RESOLVED),
        anchors.definitions(a.anchors),
    )
    if anchored.get("sealed") or any(
            score.get("_sealed") for score in scores.values()):
        raise SystemExit("sealed results cannot be exported as a leaderboard")
    return scores, anchored


def condition_profiles(scores: dict, order: dict) -> dict:
    """Returns the ranked systems' conditions grouped by task contract.

    Raises:
        SystemExit: A condition family has no task contract.
    """
    conditions = {
        c for rows in order.values() for n in rows
        for c, v in scores[n].items()
        if isinstance(v, dict) and not c.startswith("_")
    }
    try:
        return contracts.by_profile(conditions)
    except KeyError as family:
        raise SystemExit(
            f"condition family {family} has no task contract: add it to a"
            " profile in contracts.py") from None


def load_precision(results: str, scores: dict) -> dict:
    """Returns the arithmetic each system's wrapper declares it ran in.

    From the stamped metadata beside the scores, else from the wrapper's own
    description; a dash where neither says.
    """
    out = {}
    for name in scores:
        value = None
        for path in (os.path.join(results, f"{name}.meta.json"),
                     os.path.join(common.BENCH_DIR, "wrappers", "meta",
                                  f"{name}.json")):
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    value = json.load(fh).get("precision")
                if value:
                    break
        out[name] = str(value) if value else "–"
    return out


# what each key of a board's order names when printed
VIEW_NAMES = {
    "full": "denoising, full-band tier",
    "nb": "denoising, 16 kHz tier",
    "restoration": "restoration, full-band tier",
}


def header(axis_names: list, part_names: list) -> str:
    """Returns the column header of the printed board."""
    scores = ([f"{b:>{max(len(b), 5)}s}" for b in ranking.BOARDS]
              if ranking.BOARDS else [f"{'SCORE':>5s}"])
    return (f"{'#':>2}  {'system':{WIDTH}s} {'strm':>4s} {'prec':>5s}"
            f" {' '.join(scores)} | " +
            "".join(f"{k:>{max(len(k), 4)}s} " for k in part_names) +
            ("| " if part_names else "") +
            " ".join(f"{k[:4]:>4s}" for k in axis_names) +
            f" | {'supp':>5s} {'del%':>5s} {'nv%':>5s} {'clean':>5s}"
            f" {'HFerr':>5s} {'NISQA':>5s}")


def present(value: float) -> bool:
    """Returns whether a score is there: NaN marks one that is not."""
    return not math.isnan(value)


def cell(value: float) -> str:
    """Returns an axis score as a four-wide cell, a dash where there is none."""
    return f"{value:4.2f}" if present(value) else "   -"


def diagnostics(d: dict, sfx: str = "", high_band: bool = True) -> str:
    """Returns the diagnostic columns of one row of the printed board."""
    supp = scorefiles.get(d, "babble@+4", "babble_resid_db" + sfx)
    deleted = scorefiles.get(d, "_deletion", "pct")
    lost = scorefiles.get(d, "nonverbal@+0", "nv_lost_pct")
    clean = scorefiles.get(d, "clean", "si_sdr" + sfx)
    hf = scorefiles.get(d, "clean", "_hf_err")
    nisqa = scorefiles.get(d, "babble@+4", "nisqa_mos")
    return (f" | {supp:5.1f} {deleted:5.1f} {lost:5.1f} {clean:5.1f}" +
            (f" {hf:5.1f}" if high_band else "    n/a") + f" {nisqa:5.2f}")


def print_tiers(board: Board, title: str) -> None:
    """Prints the board's rules, its ranked tiers and its reference rows."""
    scores, tiers, axis_scores = board.scores, board.tiers, board.axis_scores
    entrants = [n for n in scores if tiers[n][0] in ranking.AXES_BY_TIER]
    axis_names = ranking.axis_order()
    part_names = list(ranking.PARTS or {})
    hdr = header(axis_names, part_names)
    print("=" * len(hdr))
    print(title)
    print(f"SCORE = {ranking.composite_rule()} (0-1, higher better);"
          f" {len(entrants)} entrants")
    print(f"strm (streaming probe): {streamview.rule_sentence()}; - not probed;"
          " * after a name: the probe says yes")
    print(f"Fixed anchors {board.anchored['anchors_version']}; calibration"
          f" {board.anchored['calibration_sha256'][:12]}; complete frozen"
          " deletion cohort")
    print("=" * len(hdr))
    views = [("full", "FULL-BAND TIER - eligible for the 0-48 kHz goal" +
              (f"; {ranking.ranking_board()}: the systems that add nothing"
               " above the input's band" if ranking.BOARDS else ""))]
    views += [(b, f"{b.upper()} - the systems that restore above the input's"
               f" band (more than {ranking.ASSIGN['above_db']:g} dB on the"
               " band-limited rows)" if ranking.ASSIGN else b.upper())
              for b in (ranking.BOARDS or {})
              if b != ranking.ranking_board()]
    views.append(("nb", "16 kHz TIER - band-limited by construction (scored"
                  " inside 0-8 kHz)"))
    for tier, tier_title in views:
        sfx = "_nb" if tier == "nb" else ""
        print(f"\n{tier_title}\n{hdr}")
        for i, n in enumerate(board.order[tier], 1):
            scored = anchors.part_scores(axis_scores[n], ranking.PARTS)
            row = axis_scores[n]
            axes = " ".join(cell(row.get(k, math.nan)) for k in axis_names)
            label = tiers[n][1] + ("*" if streamview.streaming_cell(
                board.stream, n) == "yes" else "")
            scored_boards = ranking.board_scores(row)
            shown = (" ".join(f"{scored_boards[b]:>{max(len(b), 5)}.3f}"
                              for b in ranking.BOARDS)
                     if ranking.BOARDS else f"{board.comp[n]:5.3f}")
            print(f"{i:2d}. {label:{WIDTH}s}"
                  f" {streamview.streaming_cell(board.stream, n):>4s}"
                  f" {board.precision.get(n, '–'):>5s}"
                  f" {shown} | " + "".join(f"{scored[k]:>{max(len(k), 4)}.2f} "
                                           for k in part_names) +
                  ("| " if part_names else "") + axes +
                  diagnostics(scores[n], sfx, tier != "nb"))
    print("\nREFERENCE ROWS (not ranked)")
    for n in [x for x in scores if tiers[x][0] == "ref"]:
        print(f"    {tiers[n][1]:{WIDTH}s} {'':4s} {'':5s} | " + " " *
              (5 * len(part_names)) + ("| " if part_names else "") + " " *
              (5 * len(axis_names) - 1) + diagnostics(scores[n]))


def print_ties(board: Board) -> None:
    """Prints the ties of each tier and the axes scored beside the composite."""
    tiers, axis_scores = board.tiers, board.axis_scores
    for tier, rows in board.order.items():
        if not rows:
            continue
        for cluster in ranking.tie_clusters(rows, board.comp):
            print(f"\nTIED ({VIEW_NAMES.get(tier, tier)}, composite gaps under"
                  f" {ranking.TIE_BAND:.2f}): " + ", ".join(
                      f"#{rows.index(n) + 1} {tiers[n][1]}" for n in cluster))
            print(
                "    Not close for the same reasons - compare the axis columns,"
                " not the rank.")
        for axis, labels in ranking.axis_ties(axis_scores, rows, tiers).items():
            print(f"\nTIED ON {axis.upper()} ({VIEW_NAMES.get(tier, tier)},"
                  " within"
                  f" {ranking.TIE_BANDS[axis]:.2f} of the best): " +
                  ", ".join(labels))
        for axis in ranking.OUTSIDE:
            scored = sorted(
                (n for n in rows
                 if present(axis_scores[n].get(axis, math.nan))),
                key=lambda n, axis=axis: -axis_scores[n][axis],
            )
            print("\nBESIDE " +
                  ("BOTH SCORES" if ranking.BOARDS else "THE COMPOSITE") +
                  f" ({VIEW_NAMES.get(tier, tier)}) - {axis}, scored and shown,"
                  " not read by the score: " +
                  " > ".join(f"{tiers[n][1]} {axis_scores[n][axis]:.2f}"
                             for n in scored))


def print_notes(board: Board, strict_delete: float) -> None:
    """Prints the streaming probe, the strict-deletion order and the contracts.

    It ends with what could not be ranked or rendered.
    """
    scores, tiers, order = board.scores, board.tiers, board.order
    if board.stream:
        print("\nSTREAMING PROBE - measured, not declared; a screening of"
              " look-ahead and state, not of delivery or throughput:")
        for n in [x for t in order for x in order[t] if x in board.stream]:
            print(f"    {tiers[n][1]:{WIDTH}s}"
                  f" {scorestream.describe(board.stream[n])}")
    strict = [
        n for n in order["full"]
        if scorefiles.get(scores[n], "_deletion", "pct") <= strict_delete
    ]
    print("\nSENSITIVITY - the denoising order of the full-band tier among"
          " systems that also keep"
          f" deleted-rate <= {strict_delete:.0f}%:")
    print("    " + " > ".join(tiers[n][1] for n in strict))
    print("    excluded by that gate: " +
          ", ".join(f"{tiers[n][1]}"
                    f" ({scorefiles.get(scores[n], '_deletion', 'pct'):.1f}%)"
                    for n in order["full"]
                    if n not in strict))
    print("\nTASK CONTRACTS - what a row holds a system to;"
          f" {contracts.axes_statement()}")
    scored = {
        c for axes in ranking.AXES_BY_TIER.values() for rows in axes.values()
        for c, _, _ in rows
    }
    for name, listed in board.grouped.items():
        print(f"    {name:13s}{contracts.PROFILES[name]}")
        print(f"    {'':13s}{len(listed)} conditions" +
              ("" if name == "denoise" else ": " +
               ", ".join(c + ("" if c in scored else " (diagnostic)")
                         for c in listed)))
    print("    undone before scoring: " + "; ".join(
        f"{what} ({bound})" for what, bound, _ in contracts.CONVENTIONS))
    for tier, systems in board.partial.items():
        print(f"\nNOT RANKED ({tier}) - incomplete or invalid metrics (missing"
              f" axes / < {ranking.MIN_ITEMS} items / level calibration"
              " saturated):")
        for n, reasons in systems.items():
            print(f"    {tiers[n][1]:{WIDTH}s} {'; '.join(reasons)}")
    if board.missing:
        print("\nMISSING OUTPUTS - items a system did not return (its means"
              " exclude them):")
        for n, m in board.missing.items():
            print(f"    {tiers[n][1]:{WIDTH}s} " +
                  ", ".join(f"{c} {k}" for c, k in m.items()))


def payload(board: Board, view: str) -> dict:
    """Returns the board as ``--json-out`` writes it."""
    anchored, scores = board.anchored, board.scores
    axis_scores = board.axis_scores
    return {
        "set_id":
            anchored["set_id"],
        "anchors_version":
            anchored["anchors_version"],
        "axes_version":
            ranking.AXES_REGISTRY["version"],
        "axes_sha256":
            anchors.digest(ranking.AXES_REGISTRY),
        "anchors": {
            k: v for k, v in anchored.items() if k != "deletion_cohort"
        },
        "axes":
            ranking.axis_order(),
        "parts":
            ranking.PARTS,
        "outside":
            ranking.OUTSIDE,
        "boards":
            ranking.BOARDS,
        "ranking_board":
            ranking.ranking_board(),
        # the registry's rule that decides which board ranks a system
        "assignment":
            ranking.ASSIGN,
        "composite_rule":
            ranking.composite_rule(),
        "streaming_rule":
            streamview.rule_sentence(),
        "tie_bands":
            dict(ranking.TIE_BANDS, composite=ranking.TIE_BAND),
        "tie_groups": {
            t: ranking.tie_clusters(rows, board.comp)
            for t, rows in board.order.items()
        },
        "contracts": {
            "profiles": contracts.PROFILES,
            "conditions": board.grouped,
            "axes": {
                k: list(v) if v else None for k, v in contracts.AXES.items()
            },
            "axes_statement": contracts.axes_statement(),
            "conventions": [list(row) for row in contracts.CONVENTIONS],
        },
        "view":
            view,
        "subset":
            next(iter(score.get("_subset", "all") for score in scores.values()),
                 "all"),
        "incomplete":
            board.partial,
        "missing_outputs":
            board.missing,
        "systems": {
            n: {
                "tier":
                    board.tiers[n][0],
                # the board the registry's rule assigns (None: no
                # band-limited rows), and the board whose score the composite
                # is (None: not ranked)
                "board":
                    ranking.board_of(scores[n]),
                "composite_board": (ranking.board_of(scores[n]) if board.comp[n]
                                    == board.comp[n] else None),
                "label":
                    board.tiers[n][1],
                "composite":
                    board.comp[n],
                "board_scores":
                    ranking.board_scores(axis_scores[n]),
                "part_scores":
                    anchors.part_scores(axis_scores[n], ranking.PARTS),
                "axis_scores":
                    axis_scores[n],
                "streaming": (streamview.streaming_summary(board.stream[n])
                              if n in board.stream else None),
                "precision":
                    board.precision.get(n),
                "band_above_input_db":
                    scorefiles.band_above_input_db(scores[n]),
                "metrics": {
                    k: ({
                        kk: vv for kk, vv in v.items() if kk != "items"
                    } if k == "_deletion" else v) for k, v in scores[n].items()
                },
            } for n in scores
        },
        "order":
            board.order,
    }


def write(path: str, text: str) -> None:
    """Writes one output file and says so."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path}")


def write_tables(board: Board, a: argparse.Namespace) -> None:
    """Writes the markdown tables the command line asks for."""
    scores, tiers, order = board.scores, board.tiers, board.order
    if a.outside_out:
        write(
            a.outside_out,
            tables.markdown_outside(board.axis_scores, order, tiers,
                                    [x for x in a.axis_tables.split(",") if x])
            + "\n")
    if a.streaming_out:
        write(a.streaming_out,
              streamview.markdown_streaming(board.stream, order, tiers) + "\n")
    if a.markdown_out:
        text = tables.markdown_boards(scores, board.axis_scores, board.comp,
                                      order, tiers, board.stream,
                                      board.precision) + "\n"
        nb = tables.markdown_table_nb(scores, board.axis_scores, board.comp,
                                      order, tiers, board.stream,
                                      board.precision)
        # the 16 kHz tier follows the full-band table after a blank line, with
        # its lead
        if nb:
            text += "\n" + tables.NB_LEAD + "\n\n" + nb + "\n"
        write(a.markdown_out, text)


def main() -> None:
    a = parse_args()
    ranking.select_view(a.view)
    tiers = roster.tiers(a.tiers)
    scores, anchored = board_scores(a, tiers)
    axis_scores, comp, order = ranking.rank(scores, tiers, anchored)
    board = Board(
        scores=scores,
        tiers=tiers,
        anchored=anchored,
        axis_scores=axis_scores,
        comp=comp,
        order=order,
        stream=streamview.load_streaming(a.results, scores),
        partial=ranking.incomplete(scores, axis_scores, comp, tiers),
        grouped=condition_profiles(scores, order),
        missing=ranking.missing_outputs(scores),
        precision=load_precision(a.results, scores),
    )
    print_tiers(board, a.title)
    print_ties(board)
    print_notes(board, a.strict_delete)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(scorefiles.finite(payload(board, a.view)),
                      fh,
                      indent=1,
                      allow_nan=False)
        print(f"\nwrote {a.json_out}")
    write_tables(board, a)


if __name__ == "__main__":
    main()
