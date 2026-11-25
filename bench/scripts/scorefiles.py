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
"""Loads the score files a board reads and refuses stale or foreign ones.

A system's base metrics come with its sidecars merged in (NISQA, the kept
voice, the real recordings). A sidecar of another set build, subset, anchor
version or render, or one read at another level calibration, is ignored or
refused, never mixed in; the checks at the end keep a filtered pass, a mix of
scorer implementations or scores of replaced audio off a board.
"""

import glob
import json
import math
import os
import sys

import common

KEPT_KEYS = (
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
)  # every dB ratio a gain shifts

REAL_FAMILIES = ("real_dirty", "real_multi", "real_noisy", "real_wild")
# a row (a condition's, or a pooled one's) feeds an axis or a pairing only
# with this many items behind it: one admission rule for the board and for
# the paired comparison
MIN_ITEMS = 20

# sidecars that read the output gain: refused at another calibration than
# the base scores'
CALIBRATED_SIDECARS = (".kept.json", ".residual.json", ".events.json")
# the level calibration is a median over the clean items, read in float32 on
# a GPU: scoring the same records again can move it by a millionth of a
# decibel; a sidecar read beyond this of its base scores' calibration was
# read at another one
CALIBRATION_TOLERANCE_DB = 1e-4


def same_calibration(a, b) -> bool:
    """Whether two level calibrations are one read (both absent, or within
    ``CALIBRATION_TOLERANCE_DB``)."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= CALIBRATION_TOLERANCE_DB


def real_quality(d):
    """Returns the pooled quality row of the real recordings, or None.

    The equal mean of the four families' NISQA, so no family weighs by its
    size. Its count is the smallest family's. None unless every family carries
    the metric on every item it was asked for: a missing item is an incomplete
    row, never a smaller mean.
    """
    values, counts = [], []
    for family in REAL_FAMILIES:
        cell = d.get(family) or {}
        value = cell.get("nisqa_mos")
        missing = (cell.get("metric_missing") or
                   {}).get("nisqa_mos", cell.get("n_missing", 0))
        if (not isinstance(value, (int, float)) or not math.isfinite(value) or
                missing):
            return None
        values.append(float(value))
        counts.append((cell.get("metric_counts") or
                       {}).get("nisqa_mos", cell.get("n", 0)))
    return {
        "nisqa_mos": sum(values) / len(values),
        "n": min(counts),
        "metric_counts": {
            "nisqa_mos": min(counts)
        },
    }


def finite(o):
    """Returns ``o`` with NaN as None, so the file is strict JSON.

    Browsers' ``JSON.parse`` rejects NaN.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def record_outputs(path):
    """Returns {item id: hash of the output samples scored}.

    Read from a per-item record file; None when the file is absent (a public
    clone holds the aggregate scores alone).
    """
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as fh:
        next(fh, None)  # the header
        for line in fh:
            record = json.loads(line)
            output = (record.get("identity") or {}).get("output")
            if output:
                out[record["id"]] = output
    return out


def _sidecar(results,
             name,
             ext,
             set_id,
             subset="all",
             anchor_identity=None,
             outputs=None):
    """Returns a sidecar's content, or None when it may not be merged.

    None when it is absent, was computed on another build of the set, on another
    subset of it (a sidecar without a subset stamp counts as whole-set data), or
    on other output samples than the base scores were (``outputs``: the base
    records' hashes, when the per-item records are at hand), or read another
    render. A stale or foreign sidecar must never overwrite fresh scores.
    """
    path = os.path.join(results, f"{name}{ext}")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        side = json.load(fh)
    scored = (record_outputs(path[:-len(".json")] +
                             ".items.jsonl") if outputs else None)
    if scored:
        differ = [
            uid for uid, output in scored.items()
            if outputs.get(uid, output) != output
        ]
        if differ:
            print(
                f"[warn] {os.path.basename(path)} was measured on other outputs"
                f" than the base scores for {len(differ)} items (first:"
                f" {differ[0]}): ignored, re-run the sidecar",
                file=sys.stderr,
            )
            return None
    if side.get("_set_id") != set_id:
        print(
            f"[warn] {os.path.basename(path)} was computed on set"
            f" {str(side.get('_set_id'))[:12]}, metrics on {str(set_id)[:12]}:"
            " ignored",
            file=sys.stderr,
        )
        return None
    if side.get("_subset", "all") != subset:
        print(
            f"[warn] {os.path.basename(path)} was computed on subset "
            f"{side.get('_subset', 'all')!r}, metrics on {subset!r}: ignored",
            file=sys.stderr,
        )
        return None
    if anchor_identity is not None and any(
            side.get(k) != anchor_identity.get(k)
            for k in ("_anchors_version", "_anchors_sha256")):
        raise ValueError(
            f"{name}{ext}: anchor version differs from its base metrics")
    if anchor_identity is not None and bool(side.get("_sealed")) != bool(
            anchor_identity.get("_sealed")):
        raise ValueError(
            f"{name}{ext}: private/public scope differs from its base metrics")
    if anchor_identity is not None and side.get(
            "_outputs_sha256") != anchor_identity.get("_outputs_sha256"):
        print(
            f"[warn] {os.path.basename(path)} read another render than the base"
            " scores: ignored, re-run the sidecar",
            file=sys.stderr,
        )
        return None
    return side


def load(results):
    """Returns each system's metrics.json with its sidecars merged in.

    The NISQA, kept, residual, events and realx sidecars, and the pooled
    quality of the real recordings (``real_quality``).

    Every sidecar that reads the output gain must have read it at the base
    scores' calibration (``_level_cal_db``); one that did not is refused, never
    mixed in.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(results, "*.metrics.json"))):
        name = os.path.basename(f)[:-len(".metrics.json")]
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        outputs = record_outputs(f[:-len(".metrics.json")] + ".items.jsonl")
        d["_scorers"] = {"metrics": d.get("_scorer_sha256")}
        for ext, pref in (
            (".nisqa.json", "nisqa"),
            (".kept.json", ("nv_", "mag_")),
            (".residual.json", ("pause_", "floor_", "babble_resid")),
            (".events.json", ("event_", "floor_resid", "voice_level")),
        ):
            side = _sidecar(
                results,
                name,
                ext,
                d.get("_set_id"),
                d.get("_subset", "all"),
                d,
                outputs,
            )
            if side is None:
                continue
            if side.get("_selection"):
                print(
                    f"[warn] {name}{ext} reads part of each condition"
                    f" ({side['_selection']}): ignored",
                    file=sys.stderr,
                )
                continue
            d["_scorers"][ext.split(".")[1]] = side.get("_scorer_sha256")
            if ext in CALIBRATED_SIDECARS and not same_calibration(
                    side.get("_level_cal_db"), d.get("_level_cal_db")):
                raise ValueError(
                    f"{name}{ext}: read at another level calibration than its"
                    " base metrics: re-run the sidecar")
            for cond, vals in side.items():
                if isinstance(vals, dict):
                    d.setdefault(cond, {}).update({
                        k: v for k, v in vals.items() if k.startswith(pref)
                    })
                    # coverage per metric, the pooled rows included: the items
                    # carrying it (the sidecar's own count when it states one),
                    # and what the sidecar did not get back
                    counts = vals.get("metric_counts") or {}
                    keys = [
                        k for k in vals
                        if k.startswith(pref) and k not in ("n",)
                    ]
                    if "n" in vals:
                        d[cond].setdefault("metric_counts", {}).update(
                            {k: counts.get(k, vals["n"]) for k in keys})
                    if vals.get("n_missing"):
                        d[cond].setdefault("metric_missing", {}).update(
                            {k: vals["n_missing"] for k in keys})
        # per-item absolute errors, averaged (levels.py)
        for vals in d.values():
            if isinstance(vals, dict):
                for row, key in (
                    ("_hf_err", "hf_err_db"),
                    ("_level_err", "level_err_db"),
                ):
                    if isinstance(vals.get(key), (int, float)):
                        vals[row] = float(vals[key])
        derive_band_rows(d)
        pooled = real_quality(d)
        if pooled is not None:
            d["_real_quality"] = pooled
        rx = _sidecar(
            results,
            name,
            ".realx.json",
            d.get("_set_id"),
            d.get("_subset", "all"),
            d,
            outputs,
        )
        if rx is not None and not same_calibration(rx.get("_level_cal_db"),
                                                   d.get("_level_cal_db")):
            raise ValueError(
                f"{name}.realx.json: read at another level calibration than its"
                " base metrics: re-run the sidecar")
        if rx is not None:
            d["_scorers"]["realx"] = rx.get("_scorer_sha256")
        items = rx.get("_items_primary") if rx is not None else None
        d["_deletion"] = {
            "pct": math.nan,
            "primary_anchored": items is not None,
            "items": items or {},
        }
        lvl = d.get("_level_cal_db")
        if isinstance(lvl, (int, float)) and lvl == lvl:
            for vals in d.values():
                if isinstance(vals, dict):
                    for k in KEPT_KEYS:
                        if isinstance(vals.get(k), (int, float)):
                            vals[k] = float(vals[k]) - float(lvl)
            d["_level_cal_applied_db"] = float(lvl)
        out[name] = d
    return out


def get(d, cond, key):
    """Returns one metric of one condition as a float, NaN when absent."""
    v = d.get(cond, {}).get(key, math.nan)
    return float(v) if isinstance(v, (int, float)) else math.nan


# the band-limited rows and the index of the first band wholly above each
# input's band (band edges at 0.5, 2, 4, 8, 12 and 16 kHz)
BAND_LIMITED_ROWS = (("srsweep_8k", 3), ("srsweep_16k", 4), ("srsweep_24k", 5))


def band_added_db(cell, first):
    """Returns the energy a system adds above a band-limited input's band on one
    row, in dB: the highest of the bands above the input's band, each read
    against the reference band-limited the same way, less the mean of the bands
    inside it, so a gain applied everywhere cancels. NaN without the bands.

    The band axis beside the scores reads it (``band_added_db`` in the row's
    cell, derived here at load from the scorer's per-band levels); the
    restoration mark reads its maximum over the rows (``band_above_input_db``).
    """
    bands = (cell or {}).get("band_kept_db")
    if not isinstance(bands, list) or len(bands) <= first:
        return math.nan
    levels = [
        float(b) if isinstance(b, (int, float)) and math.isfinite(b) else None
        for b in bands
    ]
    inside = [b for b in levels[:first] if b is not None]
    above = [b for b in levels[first:] if b is not None]
    if not inside or not above:
        return math.nan
    return max(above) - sum(inside) / len(inside)


def band_above_input_db(d):
    """Returns the most energy a system adds above a band-limited input's band,
    over the band-limited rows (``band_added_db``). NaN when none was scored."""
    values = [
        v for v in (band_added_db(d.get(cond), first)
                    for cond, first in BAND_LIMITED_ROWS) if not math.isnan(v)
    ]
    return max(values) if values else math.nan


def assigned_board(d, rule, default):
    """Returns the board an assignment rule sends a system to, or None.

    ``rule`` is the axis registry's (``admission.assignment``): a system whose
    ``band_above_input_db`` exceeds ``above_db`` goes to ``board``, every other
    to ``otherwise``; one without the band-limited rows to neither, since
    missing evidence is not a kept band. Without a rule every system goes to
    ``default``.
    """
    if not rule:
        return default
    value = band_above_input_db(d)
    if math.isnan(value):
        return None
    return rule["board"] if value > rule["above_db"] else rule["otherwise"]


def derive_band_rows(d):
    """Writes ``band_added_db`` into each band-limited row's cell of one system,
    with its coverage: the items that carry the per-band levels."""
    for cond, first in BAND_LIMITED_ROWS:
        cell = d.get(cond)
        value = band_added_db(cell, first)
        if cell is None or math.isnan(value):
            continue
        cell["band_added_db"] = value
        if "n" in cell:
            counts = cell.setdefault("metric_counts", {})
            counts["band_added_db"] = counts.get("band_kept_db", cell["n"])


def partial_passes(scores):
    """Returns {system: its selection} for a filtered scoring pass.

    Score files of a pass restricted with ``--conds`` or ``--limit``: a debug
    look at part of the set, never a row of a board.
    """
    selected = {n: d.get("_selection") for n, d in scores.items()}
    return {n: selection for n, selection in selected.items() if selection}


def render_problem(scores, results):
    """Returns why the scores do not fit the stamped renders, or None.

    Each system's base scores must have read the outputs its stamp vouches for
    (``outputs_sha256``), so a render replaced after scoring is never shown with
    the scores of the one before.
    """
    stale = sorted(
        n for n, d in scores.items()
        if d.get("_outputs_sha256") != common.outputs_digest(n, results))
    if stale:
        return (f"the scores of {', '.join(stale)} read other audio than the"
                " render stamped now: re-score them")
    return None


def scorer_problem(scores):
    """Returns why the scores cannot share one board, or None.

    Each kind of score file (the base metrics and every sidecar) must come from
    one implementation of its scorer.
    """
    for kind in ("metrics", "nisqa", "kept", "residual", "events", "realx"):
        digests = {
            n: (d.get("_scorers") or {}).get(kind) for n, d in scores.items()
        }
        known = {v for v in digests.values() if v}
        if len(known) > 1:
            groups = {}
            for n, v in digests.items():
                groups.setdefault(str(v)[:12], []).append(n)
            return (
                f"the {kind} files come from {len(known)} implementations of"
                f" their scorer ({groups}): re-score so one implementation"
                " reads every system")
    return None
