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
"""Paired comparisons with foreground/bed source-clustered uncertainty.

Resample each source factor, multiply its cluster frequencies, and keep all
repeated ladder observations together. Intervals describe this frozen panel;
they do not correct repeated adaptive selection against the same panel.
"""

import argparse
import json
import math
import pathlib
from typing import Callable, Iterator

import numpy as np

import anchors
import common
import levels
import observations
import scorefiles

IDENTITY = ("set_id", "subset", "anchors_version", "anchors_sha256")
GROUPS = ("speech_source", "bed_source")
CALIBRATED = frozenset((
    "kept_mean",
    "kept_p10",
    "kept_nb_mean",
    "kept_q1",
    "kept_q2",
    "kept_q3",
    "kept_coh_mean",
    "kept_coh_p10",
    "kept_coh_nb_mean",
    "kept_coh_nb_p10",
))


SIDECAR_KEYS = {  # what each sidecar's per-item records add to a system's
    "nisqa": (
        "nisqa_mos",
        "nisqa_noisiness",
        "nisqa_discontinuity",
        "nisqa_coloration",
        "nisqa_loudness",
    ),
    "realx": ("primary_drops", "primary_segments"),
    "kept": (
        "mag_snr",
        "mag_snr_nb",
        "nv_blocks",
        "nv_events",
        "nv_lost",
        "nv_lost_pct",
    ),
    "residual": (
        "pause_resid_db",
        "pause_resid_db_nb",
        "pause_level_db",
        "floor_db",
        "floor_worst_db",
        "babble_resid_db",
        "babble_resid_db_nb",
    ),
    "events": (
        "event_resid_db",
        "event_resid_raw_db",
        "floor_resid_db",
        "floor_resid_raw_db",
        "voice_level_db",
        "voice_pair_db",
        "voice_kept_db",
    ),
}
# board value: 100 * sum / sum over the items
POOLED = {"nv_lost_pct": ("nv_lost", "nv_blocks")}
# the pooled rows: the equal mean of the named families' means
# (scorefiles.real_quality), or the mean over every item of the named
# conditions that carries the row (scoreresidual.aggregate,
# scoreevents.aggregate)
REAL_FAMILIES = ("real_dirty", "real_multi", "real_noisy", "real_wild")
# the pause rows pool every real family (scoreresidual.REAL); the quality
# pooling keeps its four (scorefiles.REAL_FAMILIES)
CLEANUP_FAMILIES = REAL_FAMILIES + ("real_everyday",)
BLOCK_CONDS = ("coroom@pause", "coroom@speech", "coroom@floor", "coroom@both")
POOLED_ROWS = {
    "_real_quality": ("families", REAL_FAMILIES),
    "_real_cleanup": ("items", CLEANUP_FAMILIES),
    "_coroom": ("items", BLOCK_CONDS),
}


def residual_db(energies: list) -> float:
    """Returns one item's bounded residual (scoreresidual.babble_metrics)."""
    # pylint: disable-next=import-outside-toplevel
    import scoreresidual  # imported here: it loads the audio stack

    return scoreresidual.babble_metrics(energies)


def level_saturated(level_cal_db) -> bool:
    """Returns whether a system's level calibration hit the scorer's bound.

    The bound is common.MAX_LEVEL_CAL_DB.

    The scorer aligns a system's level to the input by at most that much; an
    output further off is at a scale of its own, every calibrated metric of it
    carries the remainder, and no ranking or paired comparison may read it. The
    leaderboard, the paired comparisons and the panel report all decide through
    this one rule.
    """
    return (isinstance(level_cal_db,
                       (int, float)) and math.isfinite(level_cal_db) and
            abs(level_cal_db) >= common.MAX_LEVEL_CAL_DB)


def saturated_systems(*metadata: dict) -> list:
    """Returns the names of the systems whose level calibration saturated.

    The names are taken from the given per-item headers.
    """
    return [
        m.get("system", "?")
        for m in metadata
        if level_saturated(m.get("level_cal_db"))
    ]


def same_audio(base: dict, records: list[dict], what: str) -> None:
    """Refuse sidecar records not measured on the audio of the base records.

    Every per-item record carries the identity of what it read (``identity``:
    the output samples, the mixture and reference files, the item's semantics).
    A sidecar record measured the same audio when its output, mixture and
    reference equal the base record's and its semantics agree on every field the
    base states; a sidecar may add inputs of its own (the primary-speaker
    segments it scored). Anything else measured other audio, an earlier render,
    a changed reference, and must never be merged with it. ``base`` maps item
    ids to base records; records of items the base does not hold are an
    inventory error.
    """
    for record in records:
        mine = base.get(record["id"])
        if mine is None:
            raise ValueError(
                f"{what}: {record['id']} is not in the base records")
        theirs, ours = record.get("identity") or {}, mine.get("identity") or {}
        same = all(
            theirs.get(k) == ours.get(k) for k in ("output", "mix", "ref"))
        stated = ours.get("semantics") or {}
        same = same and all((theirs.get("semantics") or {}).get(k) == v
                            for k, v in stated.items())
        if not same or not ours:
            raise ValueError(
                f"{what}: {record['id']} was measured on other audio than the"
                " base scores (output, mixture, reference or semantics"
                " differ); re-run it")


def load_system(results: str | pathlib.Path,
                system: str) -> tuple[dict, list[dict]]:
    """Loads a system's observations and the sidecars that match them.

    Foreign identities and measurements of other audio are never accepted.
    """
    root = pathlib.Path(results)
    metadata, rows = observations.read(root / f"{system}.items.jsonl")
    if metadata["system"] != system:
        raise ValueError("per-item system identity differs from filename")
    by_id = {r["id"]: r for r in rows}
    metadata["sidecar_scorers"] = {}
    for suffix, keys in SIDECAR_KEYS.items():
        path = root / f"{system}.{suffix}.items.jsonl"
        if not path.exists():
            continue
        side, extra = observations.read(path)
        if any(side.get(k) != metadata.get(k) for k in (*IDENTITY, "system")):
            raise ValueError(f"{suffix} per-item identity mismatch")
        if not side.get("scorer_sha256"):
            raise ValueError(f"{suffix} per-item scorer identity is missing")
        metadata["sidecar_scorers"][suffix] = side["scorer_sha256"]
        same_audio(by_id, extra, f"{system}.{suffix}")
        for record in extra:
            base = by_id.get(record["id"])
            if base is None or any(
                    base.get(k) != record.get(k)
                    for k in (*GROUPS, "condition")):
                raise ValueError(
                    "sidecar observations do not match the base inventory")
            for key in keys:
                base["metrics"].pop(key, None)
            if record["output_status"] == "ok":
                base["metrics"].update(record["metrics"])
                if "nv_blocks" in record["metrics"]:
                    base["metrics"].update(
                        lost_counts(record,
                                    metadata.get("level_cal_db") or 0.0))
    return metadata, rows


def lost_counts(record: dict, level_cal_db: float) -> dict:
    """Returns a kept record's lost blocks, read at ``level_cal_db``.

    The sidecar stores the counts it read at the calibration of its own pass. A
    pairing reads the blocks' raw levels again at the base scores' calibration,
    as it reads deleted speech, so a count never comes from another gain than
    the scores it is paired with.

    Raises:
        ValueError: The record keeps no block levels to read again.
    """
    kept_db = (record.get("measurements") or {}).get("nv_kept_db")
    if kept_db is None:
        raise ValueError(f"{record['id']}: a kept record without its block"
                         " levels: re-run scorekept.py")
    gone = int(np.sum(levels.lost(kept_db, level_cal_db)))
    counts = {"nv_lost": gone, "nv_blocks": len(kept_db)}
    if kept_db:
        counts["nv_lost_pct"] = 100.0 * gone / len(kept_db)
    return counts


def aligned(left: tuple,
            right: tuple) -> tuple[dict, dict, list[dict], list[dict]]:
    """Pair identical inventories.

    Silent intersections and group changes are refused.
    """
    lm, lr = left
    rm, rr = right
    if any(lm.get(k) != rm.get(k) for k in IDENTITY):
        raise ValueError(
            "paired results have different set, subset, or anchor identities")
    if (not lm.get("scorer_sha256") or
            lm.get("scorer_sha256") != rm.get("scorer_sha256") or
            lm.get("sidecar_scorers", {}) != rm.get("sidecar_scorers", {})):
        raise ValueError(
            "paired results were produced by different reference scorers")
    a, b = {r["id"]: r for r in lr}, {r["id"]: r for r in rr}
    if set(a) != set(b):
        raise ValueError(
            "paired result inventories differ; intersections are not"
            " comparisons")
    ids = sorted(a)
    for uid in ids:
        if any(a[uid].get(k) != b[uid].get(k)
               for k in (*GROUPS, "condition", "language")):
            raise ValueError("paired source groups or condition labels differ")
    return lm, rm, [a[i] for i in ids], [b[i] for i in ids]


def weights(groups: list[list],
            n_boot: int = 2000,
            seed: int = 0) -> Iterator[np.ndarray]:
    """Yield deterministic product-multinomial weights.

    The weights are for crossed source factors.

    None means that an observation has no draw from that factor (clean audio
    has no noise bed). Identical factor partitions are resampled only once.
    """
    if n_boot < 100 or not groups or not groups[0]:
        raise ValueError(
            "at least 100 bootstrap replicates and nonempty groups are required"
        )
    n = len(groups[0])
    factors, seen = [], set()
    for group in groups:
        if len(group) != n:
            raise ValueError("cluster vectors have different lengths")
        labels = sorted({str(v) for v in group if v is not None})
        mapping = {v: i for i, v in enumerate(labels)}
        indices = np.array(
            [mapping[str(v)] if v is not None else -1 for v in group])
        partition = tuple(indices.tolist())
        if partition in seen or len(labels) <= 1:
            continue
        seen.add(partition)
        factors.append((indices, len(labels)))
    if not factors:
        raise ValueError(
            "uncertainty requires at least two independent source groups")
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        result = np.ones(n, dtype=float)
        for indices, count in factors:
            frequency = rng.multinomial(count, np.full(count, 1 / count))
            present = indices >= 0
            result[present] *= frequency[indices[present]]
        yield result


def interval(reducer: Callable,
             groups: list[list],
             n_boot: int = 2000,
             seed: int = 0) -> dict:
    """Evaluate a paired effect and its percentile interval.

    The interval is taken under source resampling.
    """
    estimate = float(reducer(np.ones(len(groups[0]))))
    if not math.isfinite(estimate):
        raise ValueError("paired effect is not finite")
    values = [float(reducer(w)) for w in weights(groups, n_boot, seed)]
    finite = np.array([v for v in values if math.isfinite(v)])
    if len(finite) < 0.95 * n_boot:
        raise ValueError(
            "too many empty bootstrap cells; increase independent source"
            " coverage")
    low, high = np.quantile(finite, [0.025, 0.975]).tolist()
    return {
        "delta":
            estimate,
        "ci95": [low, high],
        "excludes_zero":
            low > 0 or high < 0,
        "n_boot":
            n_boot,
        "n_valid_boot":
            len(finite),
        "seed":
            seed,
        "cluster_counts": [
            len({v for v in g if v is not None}) for g in groups
        ],
    }


def value(record: dict, metric: str, metadata: dict) -> float:
    """Reads one item's value of a row, under the board's level convention.

    As ``levels.py`` states it: the system's output gain, ``level_cal_db`` of
    its records, is undone wherever the board undoes it, and the level and
    high-band errors are the item's absolute errors.
    """
    if record.get("output_status") != "ok":
        return float("nan")
    metrics = record.get("metrics", {})
    cal = metadata.get("level_cal_db") or 0.0
    raw = metrics.get(metric)
    if metric == "_level_err":
        gain = metrics.get("out_in_rms_db")
        raw = (levels.level_error_db(gain) if isinstance(gain,
                                                         (int,
                                                          float)) else None)
    elif metric == "_hf_err":
        bands = metrics.get("band_kept_db")
        raw = levels.hf_error_db(bands, cal) if bands else None
    elif metric == "band_added_db":
        # derived from the item's per-band levels as the loader derives the
        # aggregate (scorefiles.band_added_db); a gain cancels, so no level
        first = dict(scorefiles.BAND_LIMITED_ROWS).get(record.get("condition"))
        raw = (scorefiles.band_added_db(metrics, first)
               if first is not None else None)
    elif metric in ("babble_supp_db", "babble_supp_db_nb"):
        energies = metrics.get(metric.replace("supp_db", "energy"))
        raw = levels.suppression_db(energies, cal) if energies else None
    elif metric in ("babble_resid_db", "babble_resid_db_nb"):
        energies = metrics.get(metric.replace("resid_db", "energy"))
        raw = residual_db(energies) if energies else None
    elif metric == "nonvoiced_supp_db" and isinstance(raw, (int, float)):
        raw = levels.energy_db(raw, cal)
    elif metric in CALIBRATED and isinstance(raw, (int, float)):
        raw = levels.energy_db(raw, cal)
    if not isinstance(raw, (int, float)) or not math.isfinite(raw):
        return float("nan")
    return float(raw)


def band_vector(record: dict, first: int | None) -> list | None:
    """Returns one item's per-band levels for the band row, or None.

    None without the bands, with a band missing, or off the band-limited rows
    (``first`` None). The board pools these vectors over the items before it
    derives the row (``scorefiles.band_added_db``), so a comparison pools them
    the same way rather than averaging the items' own values.
    """
    if first is None or record.get("output_status") != "ok":
        return None
    bands = record.get("metrics", {}).get("band_kept_db")
    if not isinstance(bands, list) or len(bands) <= first or not all(
            isinstance(b, (int, float)) and math.isfinite(b) for b in bands):
        return None
    return [float(b) for b in bands]


def paired(
    left: tuple,
    right: tuple,
    condition: str,
    metric: str,
    direction: int = 1,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Returns the mean oriented left-minus-right effect.

    The effect is computed for one condition/metric. The band row
    (``band_added_db``) is pooled as the board pools it: the row's function of
    the mean per-band levels, not the mean of the items' own values.
    """
    if direction not in (-1, 1):
        raise ValueError("metric direction must be -1 or 1")
    lm, rm, lr, rr = aligned(left, right)
    if metric in CALIBRATED and saturated_systems(lm, rm):
        raise ValueError(
            "level calibration saturated for"
            f" {', '.join(saturated_systems(lm, rm))}: {metric} is a calibrated"
            " metric and cannot be compared")
    selected = [(a, b) for a, b in zip(lr, rr) if a["condition"] == condition]
    band_row = metric == "band_added_db"
    first = dict(scorefiles.BAND_LIMITED_ROWS).get(condition)

    def read(record, meta):
        if band_row:
            return band_vector(record, first)
        v = value(record, metric, meta)
        return v if math.isfinite(v) else None

    complete = [(a, b, read(a, lm), read(b, rm)) for a, b in selected]
    complete = [v for v in complete if v[2] is not None and v[3] is not None]
    if len(complete) < 2:
        raise ValueError("too few paired finite observations")
    groups = [[a.get(key) for a, _, _, _ in complete] for key in GROUPS]
    sides = delta = None
    if band_row:
        sides = [
            np.array([v[2] for v in complete]),
            np.array([v[3] for v in complete])
        ]
    else:
        delta = direction * np.array([a - b for _, _, a, b in complete])

    def reduce(w):
        if not w.sum():
            return float("nan")
        if not band_row:
            return float(w @ delta / w.sum())
        pooled = [
            scorefiles.band_added_db({"band_kept_db": list(w @ side / w.sum())},
                                     first) for side in sides
        ]
        return float(direction * (pooled[0] - pooled[1]))

    result = interval(reduce, groups, n_boot, seed)
    left_count = sum(read(a, lm) is not None for a, _ in selected)
    right_count = sum(read(b, rm) is not None for _, b in selected)
    result.update(
        condition=condition,
        metric=metric,
        direction=direction,
        n_paired=len(complete),
        n_expected=len(selected),
        n_left=left_count,
        n_right=right_count,
        matched_cohort=len(complete) == left_count == right_count,
        groups_verified=all(
            a.get("groups_verified") and b.get("groups_verified")
            for a, b, _, _ in complete),
    )
    return result


def registry_parts() -> tuple[dict | None, list]:
    """Returns how the published composite combines its axes.

    The parts of the axis registry, and the axes scored beside the composite.
    """
    # pylint: disable-next=import-outside-toplevel
    import admission  # imported here: admission imports this module

    registry = admission.load()
    return admission.parts(registry), admission.outside(registry)


def registry_boards() -> dict | None:
    """Returns the registry's boards, {board: [parts]}, or None.

    None when the registry publishes one composite of every part
    (``admission.boards``).
    """
    # pylint: disable-next=import-outside-toplevel
    import admission  # imported here: admission imports this module

    return admission.boards(admission.load())


def registry_assignment() -> dict | None:
    """Returns the registry's rule deciding which board ranks a system, or None
    (``admission.assignment``)."""
    # pylint: disable-next=import-outside-toplevel
    import admission  # imported here: admission imports this module

    return admission.assignment(admission.load())


def pooled_bands(records: list) -> dict:
    """Returns {band-limited row: {"band_kept_db": levels}} of one system.

    Each band's level is the mean over the row's items that hold it, the way
    the scorer pools them into the row's cell (``scorebench.aggregate``), so
    ``scorefiles.band_above_input_db`` reads the records as it reads the board.
    """
    cells = {}
    for condition, first in scorefiles.BAND_LIMITED_ROWS:
        rows = [
            r["metrics"].get("band_kept_db") for r in records if
            r.get("condition") == condition and r.get("output_status") == "ok"
        ]
        rows = [b for b in rows if isinstance(b, list) and len(b) > first]
        if not rows:
            continue
        width = min(len(b) for b in rows)
        levels_ = np.array(
            [[
                float(v) if isinstance(v, (int, float)) else np.nan
                for v in b[:width]
            ]
             for b in rows],
            dtype=float,
        )
        counts = np.isfinite(levels_).sum(0)
        totals = np.where(np.isfinite(levels_), levels_, 0.0).sum(0)
        cells[condition] = {
            "band_kept_db": [
                float(t / c) if c else float("nan")
                for t, c in zip(totals, counts)
            ]
        }
    return cells


def records_board(records: list) -> str | None:
    """Returns the board the registry's rule ranks a system on, from its
    per-item records, or None.

    None when the registry publishes no boards, and when the records hold no
    band-limited rows to assign one (a development panel holds none).
    """
    boards = registry_boards()
    if not boards:
        return None
    return scorefiles.assigned_board(pooled_bands(records),
                                     registry_assignment(), next(iter(boards)))


def compared_parts(axes: dict, parts, board: str | None, records: tuple):
    """Returns (parts, axes read, aside, board) of a composite comparison.

    ``parts`` "registry" reads the registry's parts. ``board`` names one of the
    registry's boards to read alone; the axes requested must cover it. Without
    a name, a request that covers more than one board (the whole registry)
    reads the board the registry's rule ranks both systems on, from their
    records (``records_board``); two systems the rule ranks on different
    boards, or a system it cannot place, are refused, since the comparison
    would otherwise answer for a task it did not name. Any other request reads
    every axis requested, in the parts it holds, and an axis beside every part
    as a group of its own; without boards the whole registry reads the
    composite. A custom grouping must hold every axis requested exactly once.
    None reads the plain mean.

    Raises:
        ValueError: The board is unknown, uncovered, unplaceable or ambiguous,
            or a grouping drops or repeats a requested axis.
    """
    if parts != "registry":
        if board is not None:
            raise ValueError("a board is read with the registry's parts")
        if parts is None:
            return None, axes, (), None
        grouped = {
            name: [a for a in members if a in axes]
            for name, members in parts.items()
        }
        grouped = {
            name: members for name, members in grouped.items() if members
        }
        if sorted(a for members in grouped.values()
                  for a in members) != sorted(axes):
            raise ValueError(
                "a custom grouping holds every axis requested exactly once")
        return grouped, axes, (), None
    members, aside = registry_parts()
    boards = registry_boards()
    if board is not None and not (boards and board in boards):
        raise ValueError(f"no board {board!r} in the registry")
    if not members:
        return None, axes, (), None
    covered = [
        name for name, held in (boards or {}).items()
        if {a for part in held for a in members[part]} <= set(axes)
    ]
    if board is not None and board not in covered:
        raise ValueError(f"the axes requested do not cover the {board} board")
    if len(covered) > 1 and board is None:
        sides = [records_board(r) for r in records]
        if None in sides:
            raise ValueError(
                "the records hold no band-limited rows to assign a board; name"
                " the board to compare on (" + ", ".join(boards) + ")")
        if sides[0] != sides[1]:
            raise ValueError(
                f"the two systems are ranked on different boards ({sides[0]},"
                f" {sides[1]}); name the board to compare on")
        board = sides[0]
    if board is not None:
        held = {part: members[part] for part in boards[board]}
        read = {a for axes_ in held.values() for a in axes_}
        return held, {a: r for a, r in axes.items() if a in read}, (), board
    inside = {a for part in members.values() for a in part}
    if not boards and inside <= set(axes):
        # one composite of every part, the axes beside it shown and not read
        return members, axes, tuple(aside), None
    grouped = {
        name: [a for a in held if a in axes] for name, held in members.items()
    }
    grouped = {name: held for name, held in grouped.items() if held}
    for axis in axes:
        if axis not in inside:
            grouped[axis] = [axis]
    return grouped, axes, (), None


def composite_delta(
    left: tuple,
    right: tuple,
    calibration: dict,
    axes: dict,
    n_boot: int = 2000,
    seed: int = 0,
    parts: dict | str | None = "registry",
    board: str | None = None,
) -> dict:
    """Bootstrap the actual composite.

    Its nonlinear reducers and anchors are preserved.

    Clean-level calibration and anchor endpoints stay frozen. Deletion pools
    segment counts and the voice outside speech its block counts; HF error takes
    the absolute value after band averaging, exactly as the leaderboard does.
    Both systems need the same metric cohort. ``parts`` and ``board`` say what
    is compared (``compared_parts``): one board's score, named or assigned by
    the registry's rule, every axis of a subset, a custom grouping or the plain
    mean. The result names the board it read (None for anything else) and the
    grouping.
    """
    lm, rm, lr, rr = aligned(left, right)
    parts, axes, aside, board = compared_parts(axes, parts, board, (lr, rr))
    if saturated_systems(lm, rm):
        raise ValueError(
            "level calibration saturated for"
            f" {', '.join(saturated_systems(lm, rm))}: no calibrated metric is"
            " valid, so there is no composite to compare")
    for meta in (lm, rm):
        anchors.validate_scores(
            calibration,
            {
                "paired": {
                    "_set_id": meta["set_id"],
                    "_subset": meta["subset"],
                    "_anchors_version": meta["anchors_version"],
                    "_anchors_sha256": meta["anchors_sha256"],
                }
            },
        )
    conditions = {
        condition for members in axes.values() for condition, _, _ in members
    }
    for pooled in conditions & set(POOLED_ROWS):
        conditions |= set(POOLED_ROWS[pooled][1])
    selected = [
        i for i, r in enumerate(lr) if r["condition"] in conditions or
        r["id"] in calibration["deletion_cohort"]
    ]
    lr, rr = [lr[i] for i in selected], [rr[i] for i in selected]
    if not all(r.get("groups_verified") for r in lr + rr):
        raise ValueError("composite intervals require verified source groups")
    ids = {r["id"]: i for i, r in enumerate(lr)}
    cells = {}
    for axis, members in axes.items():
        cells[axis] = []
        for condition, metric, _ in members:
            point = calibration["points"][anchors.row_key(condition, metric)]
            extra = None
            if condition == "_deletion":
                cohort = calibration["deletion_cohort"]
                if not set(cohort) <= set(ids):
                    raise ValueError(
                        "fixed deletion cohort is missing from per-item records"
                    )
                indices = np.array([ids[uid] for uid in sorted(cohort)])
                arrays = []
                for records, meta in ((lr, lm), (rr, rm)):
                    counts = []
                    cal = meta.get("level_cal_db") or 0.0
                    for uid in sorted(cohort):
                        record = records[ids[uid]]
                        drops = record["metrics"].get("primary_drops") or []
                        if (record["output_status"] != "ok" or
                                len(drops) != cohort[uid]):
                            raise ValueError(
                                "primary deletion counts do not cover the fixed"
                                " cohort")
                        counts.append((sum(levels.deleted(drops,
                                                          cal)), len(drops)))
                    arrays.append(np.array(counts, dtype=float))
                kind = "deletion"
            elif condition in POOLED_ROWS:
                how, families = POOLED_ROWS[condition]
                members = []
                for family in families:
                    chosen = [
                        i for i, r in enumerate(lr) if r["condition"] == family
                    ]
                    if any(lr[i]["output_status"] != "ok" or
                           rr[i]["output_status"] != "ok" for i in chosen):
                        raise ValueError(
                            "incomplete system output in a composite condition")
                    left = [
                        i for i in chosen
                        if math.isfinite(value(lr[i], metric, lm))
                    ]
                    right = [
                        i for i in chosen
                        if math.isfinite(value(rr[i], metric, rm))
                    ]
                    if left != right:
                        raise ValueError(
                            f"different per-metric cohorts: {family}/{metric}")
                    members.append(np.array(left, dtype=int))
                if how == "items":
                    members = [np.concatenate(members)]
                if any(len(m) < scorefiles.MIN_ITEMS for m in members):
                    raise ValueError(
                        f"incomplete per-metric cohorts: {condition}/{metric}")
                indices = members
                arrays = [[
                    np.array([value(rows[i], metric, meta)
                              for i in m])
                    for m in members
                ]
                          for rows, meta in ((lr, lm), (rr, rm))]
                kind = "families"
            else:
                selected = [
                    i for i, r in enumerate(lr) if r["condition"] == condition
                ]
                if any(lr[i]["output_status"] != "ok" or
                       rr[i]["output_status"] != "ok" for i in selected):
                    raise ValueError(
                        "incomplete system output in a composite condition")
                if metric == "band_added_db":
                    # pooled as the board pools it: the row's function of the
                    # mean per-band levels (scorefiles.band_added_db)
                    first = dict(scorefiles.BAND_LIMITED_ROWS).get(condition)
                    mask_left = [
                        i for i in selected
                        if band_vector(lr[i], first) is not None
                    ]
                    mask_right = [
                        i for i in selected
                        if band_vector(rr[i], first) is not None
                    ]
                    if (mask_left != mask_right or
                            len(mask_left) < scorefiles.MIN_ITEMS):
                        raise ValueError(
                            "incomplete or different per-metric cohorts:"
                            f" {condition}/{metric}")
                    indices = np.array(mask_left)
                    arrays = [
                        np.array([band_vector(rows[i], first)
                                  for i in indices])
                        for rows in (lr, rr)
                    ]
                    kind, extra = "bands", first
                elif metric in POOLED:
                    indices = np.array(selected)
                    arrays = []
                    for records in (lr, rr):
                        counts = [[
                            records[i]["metrics"].get(k) for k in POOLED[metric]
                        ] for i in selected]
                        if any(not isinstance(v, int) or v < 0
                               for pair in counts
                               for v in pair):
                            raise ValueError(
                                f"missing pooled counts: {condition}/{metric}")
                        arrays.append(np.array(counts, dtype=float))
                    # items carrying the observation
                    observing = int((arrays[0][:, 1] > 0).sum())
                    if observing < scorefiles.MIN_ITEMS or not np.array_equal(
                            arrays[0][:, 1], arrays[1][:, 1]):
                        raise ValueError(
                            "incomplete or different pooled cohorts:"
                            f" {condition}/{metric} ({observing} items"
                            " observe it)")
                    kind = "pooled"
                else:
                    field = metric
                    mask_left = [
                        i for i in selected
                        if math.isfinite(value(lr[i], field, lm))
                    ]
                    mask_right = [
                        i for i in selected
                        if math.isfinite(value(rr[i], field, rm))
                    ]
                    if (mask_left != mask_right or
                            len(mask_left) < scorefiles.MIN_ITEMS):
                        raise ValueError(
                            "incomplete or different per-metric cohorts:"
                            f" {condition}/{metric}")
                    indices = np.array(mask_left)
                    arrays = [
                        np.array([value(rows[i], field, meta)
                                  for i in indices])
                        for rows, meta in ((lr, lm), (rr, rm))
                    ]
                    kind = "mean"
            cells[axis].append((indices, arrays, kind, point, extra))

    def score(w, system):
        axis_scores = {}
        for axis, members in cells.items():
            values = []
            for indices, arrays, kind, point, extra in members:
                array = arrays[system]
                if kind == "families":
                    means = []
                    for member, values_f in zip(indices, array):
                        weight = w[member]
                        if not weight.sum():
                            return float("nan")
                        means.append(weight @ values_f / weight.sum())
                    values.append(
                        anchors.normalize(float(np.mean(means)), point))
                    continue
                weight = w[indices]
                if not weight.sum():
                    return float("nan")
                if kind in ("deletion", "pooled"):
                    total = weight @ array[:, 1]
                    if not total:
                        return float("nan")
                    raw = 100 * (weight @ array[:, 0]) / total
                elif kind == "bands":
                    raw = scorefiles.band_added_db(
                        {"band_kept_db": list(weight @ array / weight.sum())},
                        extra)
                else:
                    raw = weight @ array / weight.sum()
                values.append(anchors.normalize(float(raw), point))
            axis_scores[axis] = sum(values) / len(values)
        return anchors.composite(axis_scores, parts, aside)

    groups = [[r.get(key) for r in lr] for key in GROUPS]
    result = interval(lambda w: score(w, 0) - score(w, 1), groups, n_boot, seed)
    ones = np.ones(len(lr))
    result.update(left=score(ones, 0),
                  right=score(ones, 1),
                  n_items=len(lr),
                  board=board,
                  parts=parts)
    return result


def main() -> None:
    """Write an aggregate paired comparison.

    No item IDs or transcripts are exported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--results", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    left, right = (
        load_system(args.results, name) for name in (args.left, args.right))
    if left[0].get("sealed") or right[0].get("sealed"):
        raise SystemExit(
            "sealed comparisons must use the confirmation gate and its query"
            " ledger")
    result = paired(
        left,
        right,
        args.condition,
        args.metric,
        args.direction,
        args.bootstrap,
        args.seed,
    )
    result.update({k: left[0].get(k) for k in IDENTITY})
    result.update(left=args.left, right=args.right)
    anchors.atomic_json(args.out, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
