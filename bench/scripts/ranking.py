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
"""Axes, composite and order: how a board scores and ranks its systems.

The axes, their parts and tie bands come from the axis registry
(``bench/axes.json``, admission.py). Each sub-metric is normalised between the
frozen anchors (anchors.py) and averaged within its axis, the axes within a
part and the parts into the composite. ``select_view`` points the module's
axis tables at one view; ``rank`` and the rest read them.
"""

import math

import admission
import anchors
import common
import compare
import contracts
import roster
import scorefiles

AXES_REGISTRY = admission.load()

AXES_VIEWS = admission.replay(AXES_REGISTRY)

AXES_FULL = {k: list(v) for k, v in AXES_VIEWS["all"]["full"].items()}

AXES_NB = {k: list(v) for k, v in AXES_VIEWS["all"]["nb"].items()}

AXES_BY_TIER = {"full": AXES_FULL, "nb": AXES_NB}

# how the axes combine; None: the plain mean of the axes
PARTS = admission.parts(AXES_REGISTRY)

# axes scored and shown beside the composite, not in it
OUTSIDE = admission.outside(AXES_REGISTRY)

# the scores published side by side, {board: [parts]}; None: one composite of
# every part
BOARDS = admission.boards(AXES_REGISTRY)

# which board ranks a system (admission.assignment); None: every system ranks
# on the first board
ASSIGN = admission.assignment(AXES_REGISTRY)

# {axis: gap under which two systems tie on it}
TIE_BANDS = admission.tie_bands(AXES_REGISTRY)

# the axes score what contracts.AXES states: no other row enters an axis
# except as a stated change there
for _view in AXES_VIEWS.values():
    for _axes in _view.values():
        contracts.check_axes(_axes)


def axis_order(axes=None):
    """Returns the axes in the order of the composite's parts.

    Without parts, the registry's own order; the axes scored beside the
    composite come last.
    """
    axes = AXES_FULL if axes is None else axes
    inside = [a for a in axes if a not in OUTSIDE]
    if PARTS:
        inside = [a for part in PARTS.values() for a in part if a in axes]
    return inside + [a for a in axes if a in OUTSIDE]


def ranking_board():
    """Returns the board whose score orders the table: the first declared.

    None when the registry publishes one composite.
    """
    return next(iter(BOARDS)) if BOARDS else None


def restoration_board():
    """Returns the board the registry's assignment sends the systems that
    restore above the input's band to; None without such a rule."""
    return ASSIGN["board"] if ASSIGN else None


def assigned(d: dict, rule: dict | None = None):
    """Returns the board an assignment rule sends a system to, or None.

    ``rule`` is the registry's (``admission.assignment``) by default. None when
    the system lacks the diagnostic the rule reads (no band-limited rows):
    missing evidence assigns no board. Without a rule every system goes to the
    first board.
    """
    return scorefiles.assigned_board(d, ASSIGN if rule is None else rule,
                                     ranking_board())


def board_of(d: dict):
    """Returns the board a system is ranked on, or None.

    None when the registry publishes one composite, and when the system lacks
    the band-limited rows the assignment reads (``assigned``). Every system is
    scored on every board.
    """
    return assigned(d) if BOARDS else None


def ranks_tier(board, tier: str) -> bool:
    """Whether ``board`` ranks the systems of ``tier``: the first board ranks
    every tier, any other board the full-band tier (``admission.assignment``).
    """
    return board == ranking_board() or tier == "full"


def board_scores(scores: dict) -> dict:
    """Returns {board: the mean of its parts' scores} for one system.

    Empty without boards; a board is nan where one of its parts is.
    """
    if not BOARDS:
        return {}
    parts = anchors.part_scores(scores, PARTS)
    out = {}
    for name, members in BOARDS.items():
        values = [parts.get(part, NAN) for part in members]
        out[name] = (sum(values) / len(values) if values and all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                     else NAN)
    return out


def composite_rule():
    """Returns how the axes combine, and which sit beside the composite."""
    if not PARTS:
        rule = f"mean of {len(AXES_FULL) - len(OUTSIDE)} equally weighted axes"
    elif BOARDS:
        parts = {
            name: f"{name} ({', '.join(axes)})" for name, axes in PARTS.items()
        }
        rule = (
            "two scores, each ranking the systems of its own task and"
            " each the mean of its parts, each part the mean of its axes: " +
            "; ".join(f"{board} = {' + '.join(parts[p] for p in members)}"
                      for board, members in BOARDS.items()))
        if ASSIGN:
            rule += (f"; a system that restores above the input's band (more"
                     f" than {ASSIGN['above_db']:g} dB on the band-limited"
                     f" rows) is ranked on {ASSIGN['board']}, every other on"
                     f" {ASSIGN['otherwise']}, one without those rows on"
                     " neither")
        else:
            rule += f"; the table is ordered by {ranking_board()}"
    else:
        rule = (
            f"mean of {len(PARTS)} equally weighted parts, each the mean of its"
            " axes: " + "; ".join(
                f"{name} ({', '.join(axes)})" for name, axes in PARTS.items()))
    if OUTSIDE:
        rule += (f"; scored beside both, read by neither: {', '.join(OUTSIDE)}"
                 if BOARDS else
                 f"; scored beside it, not in it: {', '.join(OUTSIDE)}")
    return rule


def select_view(view):
    """Points the module's axis tables at one view, in place.

    ``rank`` and the rest read the module's tables. A development panel
    (``common.PANELS``) reads the whole-set axes.
    """
    axes_view = "all" if view in common.PANELS else view
    if axes_view not in AXES_VIEWS:
        raise SystemExit(
            f"--view must be one of"
            f" {sorted(AXES_VIEWS) + list(common.PANELS)}, not {view!r}")
    for tier, axes in AXES_VIEWS[axes_view].items():
        AXES_BY_TIER[tier].clear()
        AXES_BY_TIER[tier].update({k: list(v) for k, v in axes.items()})
    return view


NAN = float("nan")

TIE_BAND = 0.03

MIN_ITEMS = scorefiles.MIN_ITEMS  # fewer scored items cannot feed an axis


def whole_counts(entry: list, count: int) -> bool:
    """Returns whether one recording's [deleted, segments] fit its cohort.

    Both must be integers, the segments the cohort's count and the deleted ones
    between none and all of them.
    """
    # bool is an int subclass, and a flag is not a count
    integers = all(
        isinstance(v, int) and not isinstance(v, bool) for v in entry)
    return (len(entry) == 2 and integers and entry[1] == count and
            0 <= entry[0] <= count)


def pool_deletion(scores: dict, cohort: dict) -> set:
    """Pools each system's deletion on the frozen INPUT cohort.

    Every expected segment is required.

    A missing recording or changed segment denominator makes that system
    incomplete. It cannot shrink another system's evaluation population.
    """
    for score in scores.values():
        data = score["_deletion"]
        items = data.get("items") or {}
        complete = bool(cohort) and all(
            uid in items and whole_counts(items[uid], count)
            for uid, count in cohort.items())
        if complete:
            deleted = sum(items[uid][0] for uid in sorted(cohort))
            total = sum(cohort.values())
            data.update(
                pct=100.0 * deleted / total,
                n_segments=total,
                cohort_items=len(cohort),
                complete=True,
            )
        else:
            data.update(pct=NAN, n_segments=0, cohort_items=0, complete=False)
    return set(cohort)


def missing_outputs(scores):
    """Returns {system: {condition: items not returned}}.

    Every system whose metrics report items it failed to render: a system does
    not improve its means by failing on its hardest items, so the board says so
    next to the rank.
    """
    out = {}
    for n, d in scores.items():
        m = {
            c: int(v["n_missing"]) for c, v in d.items() if
            isinstance(v, dict) and not c.startswith("_") and v.get("n_missing")
        }
        if m:
            out[n] = m
    return out


def _covered(scores, n, cond, key):
    """Returns whether a row may feed an axis.

    It may with a finite value from at least ``MIN_ITEMS`` items that carry the
    metric and no item the system or its sidecar failed to return: a missing
    item is never a smaller mean, it is an incomplete row. A pooled row (one
    named with a leading underscore) answers to the same rule through the
    counts its sidecar states (the deletion row's are the segments of its
    frozen cohort), and to its own completeness flag where it has one.
    """
    v = scorefiles.get(scores[n], cond, key)
    data = scores[n].get(cond, {})
    # the items behind the row, or for the deletion row the segments of its
    # frozen cohort
    count = data.get("metric_counts",
                     {}).get(key, data.get("n", data.get("n_segments", 0)))
    missing = data.get("n_missing", 0) or data.get("metric_missing", {}).get(
        key, 0)
    return (math.isfinite(v) and bool(data.get("complete", True)) and
            count >= MIN_ITEMS and not missing)


def rank(scores: dict,
         tiers: dict | None = None,
         calibration: dict | None = None):
    """Score and order every system. Returns (axis_scores, composite, order).

    The caller supplies a frozen calibration. Neither endpoints nor the
    deletion cohort are inferred from the systems being ranked. ``order``
    holds each tier's systems by their score on the board that orders the
    table and, under every other board's name, the full-band systems that
    board ranks (``board_of``); the composite is each system's score on its
    own board.
    """
    tiers = tiers or roster.TIERS
    if calibration is None:
        raise ValueError(
            "a frozen anchors.resolved.json calibration is required")
    anchors.validate_scores(calibration, scores)
    if calibration.get("axes_sha256") != anchors.digest(AXES_BY_TIER):
        raise ValueError("axis membership differs from the frozen calibration")
    untiered = sorted(n for n in scores if n not in tiers and n != "INPUT")
    # a result file nobody registered: a stray sweep, never a row of the
    # published board
    if untiered:
        print("IGNORED (no tier registered; pass --tiers to rank them):"
              f" {', '.join(untiered)}")
        scores = {n: v for n, v in scores.items() if n not in untiered}
    tier_of = {n: tiers.get(n, ("ref", n))[0] for n in scores}
    pool_deletion(scores, calibration["deletion_cohort"])
    axis_scores = {}
    for tier, axes in AXES_BY_TIER.items():
        members = [n for n in scores if tier_of[n] == tier]
        for n in members:
            out = {}
            for axis, subs in axes.items():
                vs = []
                for cond, key, sign in subs:
                    if not _covered(scores, n, cond, key):
                        vs = []
                        break
                    point = calibration["points"][anchors.row_key(cond, key)]
                    if point["direction"] != sign:
                        raise ValueError(
                            f"axis/anchor direction mismatch: {cond}/{key}")
                    vs.append(
                        anchors.normalize(scorefiles.get(scores[n], cond, key),
                                          point))
                out[axis] = sum(vs) / len(vs) if vs else NAN
            axis_scores[n] = out
    for n in scores:
        axis_scores.setdefault(n, {})

    def composite(n):
        """Returns the composite, only when every axis is present.

        ``anchors.composite`` over the axis scores. Partial entrants are not
        ranked, nor is a system whose level calibration saturated: its
        calibrated metrics are not valid.
        """
        if level_saturated(scores[n]):
            return NAN
        if not BOARDS:
            return anchors.composite(axis_scores[n], PARTS, OUTSIDE)
        # each board needs its own axes and no other's: a system is ranked on
        # the board of its task (board_of) by that board's score; a system the
        # rule cannot place, or a board that does not rank its tier, gives no
        # composite, and incomplete() says why
        board = board_of(scores[n])
        if board is None or not ranks_tier(board, tier_of[n]):
            return NAN
        return board_scores(axis_scores[n])[board]

    comp = {n: composite(n) for n in scores}
    first = ranking_board()
    order = {
        t:
            sorted(
                (n for n in scores if tier_of[n] == t and comp[n] == comp[n] and
                 board_of(scores[n]) == first),
                key=lambda n: -comp[n],
            ) for t in AXES_BY_TIER
    }
    # every other board ranks its own full-band systems by its own score
    for board in BOARDS or ():
        if board != first:
            order[board] = sorted(
                (n for n in scores if tier_of[n] == "full" and
                 comp[n] == comp[n] and board_of(scores[n]) == board),
                key=lambda n: -comp[n],
            )
    return axis_scores, comp, order


def level_saturated(scores: dict) -> bool:
    """Returns whether the scorer's level calibration hit its bound.

    The one rule shared with the paired comparisons and the panel report
    (``compare.level_saturated``).
    """
    return compare.level_saturated(scores.get("_level_cal_db"))


def incomplete(scores, axis_scores, comp, tiers):
    """Returns {tier: {system: [reasons]}} for the entrants not ranked.

    The reasons are the axes they miss, or a level calibration that saturated.
    """
    out = {}
    for n in scores:
        tier = tiers.get(n, ("ref", n))[0]
        if tier in AXES_BY_TIER and comp[n] != comp[n]:
            reasons = []
            if level_saturated(scores[n]):
                reasons.append(
                    f"level calibration saturated at"
                    f" {common.MAX_LEVEL_CAL_DB:g} dB (the output sits"
                    f" {scores[n]['_level_cal_db']:+.1f} dB or"
                    " further from the input: a scale of its own, so no"
                    " calibrated metric is valid)")
            board = board_of(scores[n]) if BOARDS else None
            if BOARDS and board is None:
                reasons.append("no board: the band-limited rows (" + ", ".join(
                    c for c, _ in scorefiles.BAND_LIMITED_ROWS) +
                               ") that assign one are missing")
            elif BOARDS and not ranks_tier(board, tier):
                reasons.append(
                    f"assigned to {board}, which ranks the full-band tier"
                    " only")
            needed = (set(a for part in BOARDS[board] for a in PARTS[part])
                      if board else None)
            missing = [
                ax for ax, v in axis_scores[n].items()
                if v != v and (needed is None or ax in needed)
            ]
            if missing:
                reasons.append("missing " + ", ".join(missing))
            out.setdefault(tier, {})[n] = reasons or ["missing all"]
    return out


def tie_clusters(rows, comp):
    """Returns the groups of tied systems among the ordered ``rows``.

    Top down, a group holds the systems under ``TIE_BAND`` below its best
    composite, so every pair inside it is a tie; only groups of two or more are
    returned.
    """
    clusters, current = [], []
    for n in rows:
        if current and comp[current[0]] - comp[n] < TIE_BAND:
            current.append(n)
            continue
        if len(current) > 1:
            clusters.append(current)
        current = [n]
    if len(current) > 1:
        clusters.append(current)
    return clusters


def axis_ties(axis_scores, rows, tiers):
    """Returns {axis: labels} of the systems tied with an axis's best.

    For every axis that declares a tie band and has more than one ranked system
    within that band of its best score.
    """
    out = {}
    for axis, band in TIE_BANDS.items():
        scored = {n: axis_scores[n].get(axis, NAN) for n in rows}
        scored = {n: v for n, v in scored.items() if v == v}
        if not scored:
            continue
        best = max(scored.values())
        tied = [
            n for n in sorted(scored, key=lambda n, scored=scored: -scored[n])
            if best - scored[n] < band
        ]
        if len(tied) > 1:
            out[axis] = [tiers[n][1] for n in tied]
    return out
