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
"""Deleted-rate over the real-recording conditions, pooled, as a sidecar.

Deleted-rate is a RARE BINARY event over VAD speech segments; the 24-file
conditions alone give confidence intervals too wide to separate systems, so
this pools every real condition (~900 segments) and reports a 95%
Clopper-Pearson interval.

Primary-speaker anchoring (--primary-anchor): a deleted BACKGROUND voice is
an isolator working as intended; only a deleted PRIMARY-speaker segment is a
defect. Each segment's INPUT audio is embedded (WavLM-SV) and compared to the
file's anchor segment (max energy x duration); deletions are then charged only
for primary segments (sim >= threshold). A denoiser that keeps every voice
scores identically under both definitions. The board pools the
primary-anchored counts (``_items_primary``) over the frozen INPUT cohort
(``ranking.pool_deletion``).

A segment is deleted when the output's energy over it, with the system's output
gain undone (``levels.deleted`` at the base scores' ``_level_cal_db``), is more
than 10 dB under the input's: a louder or quieter output convention neither
hides nor invents a deletion. The records keep every segment's raw drop, so any
view re-reads them at its own calibration.

Every segment the detector reports counts (``MIN_SEG``, its own minimum), so a
deleted short word costs what a long one does; this is segment coverage, not
word-level detection, and a word removed inside a longer segment that keeps
its energy is not seen here. A segment that touches the item's first or last
50 ms (``EDGE``) holds a word the recording cut off: it loses ``TRIM`` at that
edge, and what remains, when it is still a segment, is read like any other.
A remainder shorter than that is a fragment: it is in the input and no system's
to repair, so it leaves the pools and is counted apart (``_edge_segments``).

The primary-speaker flags are decided once, on INPUT, and read by every other
system from INPUT's records (``frozen_sims``): the embeddings are not
bit-reproducible across machines, and a flag that flips between two scoring
runs would change the cohort under the board. Each system's records are bound
to INPUT's decisions by a digest of its segment boundaries and similarities in
their identity, so a record made under another INPUT pass is never reused, and
a fresh pass must reproduce INPUT's boundaries exactly. A segment shorter than
the embedder reads is embedded with the input around it and counted as widened
(``_widened_segments``): a consistent decision, not a confident one.

Usage: scorerealx.py --systems a,b [--conds ...] [--primary-anchor]
Output: <BENCH_RESULTS>/<system>.realx.json
"""

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torchaudio

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors
import common
import levels
import observations
import scorebench

SCORER_SHA256 = observations.implementation(
    __file__,
    common.__file__,
    observations.__file__,
    levels.__file__,
    scorebench.__file__,
)

# Silero's own shortest segment, 250 ms at 16 kHz: nothing shorter is reported
MIN_SEG = 4000
# a segment starting within 50 ms of the item's start, or ending within 50 ms
# of its end, holds a word the recording cut off
EDGE = 800
# what such a segment loses at that edge, 300 ms at 16 kHz: the cut-off word
TRIM = 4800
# the shortest input the speaker embedder reads: half a second at 16 kHz
EMB_MIN = 8000


def embedding_span(start, end, length):
    """Returns ``[start, end)`` widened to ``EMB_MIN`` samples inside the item.

    Centred on the segment where the item allows, so a short word is embedded
    with the voice around it.
    """
    short = EMB_MIN - (end - start)
    if short <= 0:
        return start, end
    start = max(0, start - short // 2)
    end = min(length, start + EMB_MIN)
    return max(0, end - EMB_MIN), end


def select_items(rows, conds):
    """Returns the manifest rows named by ``conds``.

    A comma-separated list of exact condition names, or a single prefix (the
    default ``real``) when no row carries that exact name.
    """
    want = [c for c in conds.split(",") if c]
    if len(want) == 1 and not any(r["condition"] == want[0] for r in rows):
        return [r for r in rows if r["condition"].startswith(want[0])]
    return [r for r in rows if r["condition"] in want]


def ci95(k, n):
    # pylint: disable-next=import-outside-toplevel
    from scipy import stats

    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * lo, 100 * hi


def anchor_sims(mix, vad, spk):
    """Returns each usable input segment's similarity to the anchor segment.

    The anchor is the segment with the largest energy x duration; None when the
    input holds no usable segment.
    """
    i16, segs = vad.segments(mix)
    usable = [(s["start"], min(s["end"], len(i16)))
              for s in segs
              if min(s["end"], len(i16)) - s["start"] >= MIN_SEG]
    if not usable:
        return None
    embs, scores = [], []
    for b, e in usable:
        w48 = mix[b * 3:e * 3]
        scores.append(float(w48.pow(2).mean()) * (e - b))
        # the embedder needs half a second: a shorter segment is read with the
        # input around it
        if e - b < EMB_MIN:
            b, e = embedding_span(b, e, len(i16))
        embs.append(spk.emb(mix[b * 3:e * 3]))
    anchor = int(np.argmax(scores))
    return [float(torch.dot(em, embs[anchor])) for em in embs]


def score_item(it, out, mix, vad, sims, threshold):
    """Returns one item's usable input segments and the output's drop on each.

    The output's raw energy drop on each segment, and whether the segment
    belongs to the primary speaker (``sims`` against ``threshold``; unknown
    without anchoring).
    """
    n = min(len(out), len(mix))
    i16, segs = (scorebench.speech_segments(vad, mix, it["mix_path"])
                 if n == len(mix) else vad.segments(mix[:n]))
    o16 = torchaudio.functional.resample(out[:n], common.SR, 16000)[:len(i16)]
    i16 = i16[:len(o16)]
    drops, prim, edge, trimmed, widened = [], [], [], [], []
    starts, ends = [], []
    k = -1
    for s in segs:
        b, e = s["start"], min(s["end"], len(o16))
        if e - b < MIN_SEG:
            continue
        k += 1
        widened.append(1.0 if e - b < EMB_MIN else 0.0)
        # the edge a segment touches takes the cut-off word with it; what
        # remains is read whole, or reported as a fragment when too short
        tb = b + TRIM if b <= EDGE else b
        te = e - TRIM if e >= len(o16) - EDGE else e
        touches = (tb, te) != (b, e)
        fragment = touches and te - tb < MIN_SEG
        trimmed.append(1.0 if touches and not fragment else 0.0)
        edge.append(1.0 if fragment else 0.0)
        if touches and not fragment:
            b, e = tb, te
        starts.append(int(b))
        ends.append(int(e))
        drop = float(10 * torch.log10(
            (o16[b:e].pow(2).mean() + 1e-9) / (i16[b:e].pow(2).mean() + 1e-9)))
        drops.append(drop)
        prim.append(1.0 if sims is not None and k < len(sims) and
                    sims[k] >= threshold else 0.0)
    return {
        "seg_drop": drops,
        "seg_primary": prim,
        "seg_edge": edge,
        "seg_trimmed": trimmed,
        "seg_widened": widened,
        # 16 kHz sample indices of the input's segments as read: frozen
        # boundaries every output is read on, never an output's own VAD
        "seg_start": starts,
        "seg_end": ends,
        "anchor_sims": [list(sims)] if sims is not None and drops else [],
    }


def item_metrics(m):
    """Returns per-item segment counts and the primary segments' raw drops.

    They do not depend on the view: the deletions are read at a calibration
    (``levels.deleted``).
    """
    drops, prim = m.get("seg_drop", []), m.get("seg_primary", [])
    edge = m.get("seg_edge") or [0.0] * len(drops)
    metrics = {
        "n_segments": len(drops),
        "n_edge_segments": int(sum(edge)),
        "n_trimmed_segments": int(sum(m.get("seg_trimmed") or [])),
        "n_widened_segments": int(sum(m.get("seg_widened") or [])),
    }
    primary = [d for d, p, x in zip(drops, prim, edge) if p and not x]
    if primary:
        metrics["primary_segments"] = len(primary)
        metrics["primary_drops"] = primary
    return metrics


def frozen_key(decision):
    """Returns the digest of INPUT's decision on one item (``frozen_sims``)."""
    return hashlib.sha256(json.dumps(
        decision, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def frozen_sims(path, items):
    """Returns {item id: INPUT's decision} for ``items``.

    A decision is ``{"sims", "start", "end"}``: the anchor similarities every
    other system reads its flags from (None where the input holds no usable
    segment) and the segment boundaries, in 16 kHz samples, every other system
    must reproduce. INPUT's records must be of this set, the whole set and this
    scorer, anchored, and cover every item.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: score INPUT with --primary-anchor"
                         " first, into the same results directory")
    metadata, records = observations.read(path)
    if metadata.get("system") != "INPUT":
        raise SystemExit(f"{path}: not INPUT's records")
    if (metadata.get("set_id") != common.set_id() or
            metadata.get("subset", "all") != "all"):
        raise SystemExit(f"{path}: records of another set or subset")
    if metadata.get("scorer_sha256") != SCORER_SHA256:
        raise SystemExit(f"{path}: made by another deletion scorer; re-score"
                         " INPUT first")
    by_id = {r["id"]: r for r in records}
    missing = [it["id"] for it in items if it["id"] not in by_id]
    if missing:
        raise SystemExit(f"{path} lacks {len(missing)} items (first:"
                         f" {missing[0]}): re-score INPUT over the whole set")
    out = {}
    for it in items:
        m = by_id[it["id"]].get("measurements") or {}
        if m.get("seg_drop") and not m.get("anchor_sims"):
            raise SystemExit(f"{path}: {it['id']} carries no speaker anchoring;"
                             " re-score INPUT with --primary-anchor")
        sims = m.get("anchor_sims") or []
        first = list(sims[0]) if sims else []
        out[it["id"]] = {
            "sims": first or None,
            "start": [int(v) for v in m.get("seg_start") or []],
            "end": [int(v) for v in m.get("seg_end") or []],
        }
    return out


def assemble(name,
             items,
             vad,
             spk,
             threshold,
             reuse=None,
             hashes=None,
             frozen=None):
    """Returns per-item records over the real ``items``.

    Reused by content identity (output, mixture, the anchoring rule and INPUT's
    decision on the item) since an earlier pass of this sidecar, scored
    otherwise. With ``frozen`` (INPUT's decisions, ``frozen_sims``) the flags
    are INPUT's, no embedding is computed, and the segments read must be
    INPUT's to the sample.

    Raises:
        SystemExit: An item's segments differ from INPUT's.
    """
    hashes = hashes or common.file_hashes()
    reuse = reuse or {}
    records, counts = [], collections.Counter()
    sims_cache = {}
    for it in items:
        record = observations.item(it)
        records.append(record)
        op = (it["mix_path"] if name == "INPUT" else os.path.join(
            common.OUT_ROOT, name, it["id"]))
        if not os.path.exists(op):
            record["output_status"] = "missing"
            continue
        out, mix = common.load48(op), common.load48(it["mix_path"])
        if len(out) != len(mix):
            record["output_status"] = "wrong_length"
            continue
        if not bool(torch.isfinite(out).all()):
            record["output_status"] = "nonfinite"
            continue
        record["identity"] = scorebench.item_identity(
            it, scorebench.sample_hash(out), hashes)
        record["identity"]["semantics"]["sim_threshold"] = (
            threshold if spk is not None else None)
        # INPUT's decision is part of what this record depends on: another
        # INPUT pass means another record
        decision = frozen[it["id"]] if frozen is not None else None
        record["identity"]["semantics"]["frozen"] = (
            frozen_key(decision) if decision is not None else None)
        old = reuse.get(scorebench.reuse_key(record["identity"]))
        if old is not None:
            record["measurements"] = dict(old["measurements"])
            counts["reused"] += 1
        else:
            sims = None
            if decision is not None:
                sims = decision["sims"]
            elif spk is not None:
                if it["id"] not in sims_cache:
                    sims_cache[it["id"]] = anchor_sims(mix, vad, spk)
                sims = sims_cache[it["id"]]
            measured = score_item(it, out, mix, vad, sims, threshold)
            if decision is not None and (
                    measured["seg_start"] != decision["start"] or
                    measured["seg_end"] != decision["end"]):
                raise SystemExit(
                    f"{it['id']}: the segments read differ from INPUT's"
                    f" ({len(measured['seg_start'])} against"
                    f" {len(decision['start'])}, or other boundaries): the"
                    " voice detector or its cache changed; re-score INPUT")
            record["measurements"] = observations.finite(measured)
            counts["scored"] += 1
        record["metrics"] = item_metrics(
            observations.revive(record["measurements"]))
    if reuse:
        print(
            f"  reused {counts['reused']} records, scored {counts['scored']}"
            " items",
            flush=True,
        )
    return records


def aggregate(records, items, threshold, level_cal_db):
    """Returns the sidecar file for ``items`` from their records.

    Deletions at the system's calibration pooled per condition and over the
    view, the primary-anchored pool, and the per-item cohorts. Pure, so any view
    of the set derives from the whole-set records.
    """
    by_id = {r["id"]: r for r in records}
    per_cond = collections.defaultdict(lambda: [0, 0])
    prim = collections.defaultdict(lambda: [0, 0])
    edges = collections.defaultdict(lambda: [0, 0])
    items_real, items_prim, sims_out = {}, {}, {}
    missing = trimmed = widened = 0
    for it in items:
        rec = by_id.get(it["id"])
        if rec is None or rec.get("output_status") != "ok":
            missing += 1
            continue
        m = observations.revive(rec.get("measurements", {}))
        trimmed += int(sum(m.get("seg_trimmed") or []))
        widened += int(sum(m.get("seg_widened") or []))
        drops = m.get("seg_drop", [])
        edge = m.get("seg_edge") or [0.0] * len(drops)
        inner = [i for i, x in enumerate(edge) if not x]
        cut = levels.deleted([d for d, x in zip(drops, edge) if x],
                             level_cal_db)
        if cut:
            edges[it["condition"]][0] += int(sum(cut))
            edges[it["condition"]][1] += len(cut)
        dels = levels.deleted([drops[i] for i in inner], level_cal_db)
        flags = m.get("seg_primary", [])
        prims = [flags[i] for i in inner] if flags else []
        if not dels:
            continue
        cell = per_cond[it["condition"]]
        cell[0] += int(sum(dels))
        cell[1] += len(dels)
        items_real[it["id"]] = [int(sum(dels)), len(dels)]
        sims = m.get("anchor_sims") or []
        if sims:
            sims_out[it["id"]] = sims[0]
        n_prim = int(sum(prims))
        if n_prim:
            k_prim = int(sum(d for d, p in zip(dels, prims) if p))
            prim[it["condition"]][0] += k_prim
            prim[it["condition"]][1] += n_prim
            items_prim[it["id"]] = [k_prim, n_prim]
    out_d = {}
    k_all = n_all = 0
    for cond, (k, n) in sorted(per_cond.items()):
        lo, hi = ci95(k, n)
        out_d[cond] = {
            "deleted_rate_pct": 100 * k / max(1, n),
            "n_segments": n,
            "n_deleted": k,
            "ci95": [lo, hi],
        }
        k_all += k
        n_all += n
    if n_all:
        lo, hi = ci95(k_all, n_all)
        out_d["_pooled_real"] = {
            "deleted_rate_pct": 100 * k_all / n_all,
            "n_segments": n_all,
            "n_deleted": k_all,
            "ci95": [lo, hi],
        }
    if prim:
        kp = sum(v[0] for v in prim.values())
        np_ = sum(v[1] for v in prim.values())
        if np_:
            lo, hi = ci95(kp, np_)
            out_d["_pooled_primary"] = {
                "deleted_rate_pct": 100 * kp / np_,
                "n_segments": np_,
                "n_deleted": kp,
                "ci95": [lo, hi],
                "sim_threshold": threshold,
            }
        out_d["_anchor_sims"] = sims_out
        out_d["_items_primary"] = items_prim
    out_d["_items_real"] = items_real
    # the words the recordings cut off at an edge: shown, never charged
    out_d["_edge_segments"] = {
        cond: {
            "n_segments": n,
            "n_deleted": k
        } for cond, (k, n) in sorted(edges.items())
    }
    # segments read without the edge they touched, and short segments whose
    # speaker was decided with the input around them
    out_d["_trimmed_segments"] = trimmed
    out_d["_widened_segments"] = widened
    out_d["_missing_outputs"] = missing
    out_d["_level_cal_db"] = level_cal_db
    return out_d


def level_calibration(results, name):
    """Returns the system's output gain from its base scores in ``results``.

    The view's own scores.
    """
    path = os.path.join(results, f"{name}.metrics.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path} is missing: score {name} before its deletion sidecar")
    with open(path, encoding="utf-8") as fh:
        return float(json.load(fh).get("_level_cal_db") or 0.0)


def threshold_of(records, given):
    """Returns the anchoring threshold the records were scored with.

    One threshold for all records, or ``given`` when they state none.
    """
    stated = set()
    for r in records:
        if r.get("output_status") == "ok":
            semantics = (r.get("identity") or {}).get("semantics", {})
            stated.add(semantics.get("sim_threshold"))
    stated.discard(None)
    if len(stated) > 1:
        raise SystemExit(
            f"records scored with {len(stated)} anchoring thresholds")
    return stated.pop() if stated else given


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument(
        "--conds",
        default="real",
        help=("comma-separated conditions, or a prefix every pooled condition"
              " starts with"),
    )
    ap.add_argument("--primary-anchor", action="store_true")
    ap.add_argument("--sim-threshold", type=float, default=0.85)
    ap.add_argument(
        "--reuse-from",
        default="",
        metavar="DIR",
        help=(
            "results of an earlier pass of this sidecar: unchanged items keep "
            "their records"),
    )
    ap.add_argument(
        "--aggregate-only",
        default="",
        metavar="DIR",
        help="derive this subset's files from the whole-set records under DIR",
    )
    a = ap.parse_args()

    items = select_items(common.manifest(), a.conds)
    common.guard_private_results(items)
    vad = spk = None
    if not a.aggregate_only:
        vad = common.Vad()
        spk = common.SpeakerEmbedder() if a.primary_anchor else None

    os.makedirs(common.RESULTS, exist_ok=True)
    # INPUT first: its records carry the flags every other system reads
    names = [n for n in a.systems.split(",") if n]
    names = [n for n in names if n == "INPUT"
            ] + [n for n in names if n != "INPUT"]
    for name in names:
        if a.aggregate_only:
            src = os.path.join(a.aggregate_only, f"{name}.realx.items.jsonl")
            ids = {it["id"] for it in items}
            problem, source, records = scorebench.view_records(
                src, name, ids, SCORER_SHA256)
            if problem:
                print(f"skip {name}: {problem}", flush=True)
                continue
            outputs = source.get("outputs_sha256")
        else:
            problem = name != "INPUT" and scorebench.render_problem(name)
            if problem:
                print(f"skip {name}: {problem}", flush=True)
                continue
            own = os.path.join(common.RESULTS, f"{name}.realx.items.jsonl")
            reuse, why = {}, "no earlier records"
            if a.reuse_from:
                reuse, why = scorebench.reuse_index(
                    os.path.join(a.reuse_from, f"{name}.realx.items.jsonl"),
                    name,
                    SCORER_SHA256,
                )
                why = f"reuse from {a.reuse_from}: {why}"
            # never skipped as scored: re-validated item by item
            elif os.path.isfile(own):
                reuse, why = scorebench.reuse_index(own, name, SCORER_SHA256)
                why = f"existing records: {why}"
            print(f"  {why}", flush=True)
            frozen = None
            if spk is not None and name != "INPUT":
                frozen = frozen_sims(
                    os.path.join(common.RESULTS, "INPUT.realx.items.jsonl"),
                    items)
            records = assemble(name,
                               items,
                               vad,
                               spk,
                               a.sim_threshold,
                               reuse=reuse,
                               frozen=frozen)
            outputs = common.outputs_digest(name)
        level_cal_db = level_calibration(common.RESULTS, name)
        out_d = aggregate(records, items,
                          threshold_of(records, a.sim_threshold), level_cal_db)
        observations.sidecar(
            os.path.join(common.RESULTS, f"{name}.realx.items.jsonl"),
            records,
            name,
            common.set_id(),
            common.SUBSET,
            common.is_sealed(items),
            SCORER_SHA256,
            level_cal_db=level_cal_db,
            outputs_sha256=outputs,
        )
        # the leaderboard ignores a sidecar from another build, and one from
        # another subset of the set
        out_d["_set_id"] = common.set_id()
        out_d["_subset"] = common.SUBSET
        out_d.update(anchors.stamp())
        out_d["_sealed"] = common.is_sealed(common.manifest())
        out_d["_scorer_sha256"] = SCORER_SHA256
        out_d["_outputs_sha256"] = outputs  # the render these records read
        path = os.path.join(common.RESULTS, f"{name}.realx.json")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out_d, f, indent=1)
        os.replace(tmp, path)
        p = out_d.get("_pooled_real", {})
        pp = out_d.get("_pooled_primary", {})
        extra = (f"  PRIMARY={pp.get('deleted_rate_pct', 0):.1f}% "
                 f"({pp.get('n_deleted', 0)}/{pp.get('n_segments', 0)})"
                 if pp else "")
        print(
            f"{name:24s} pooled deleted={p.get('deleted_rate_pct', 0):5.1f}% "
            f"({p.get('n_deleted', 0)}/{p.get('n_segments', 0)}){extra}  "
            f"missing_outputs={out_d['_missing_outputs']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
