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
"""NISQA v2 no-reference quality sidecar over wavs already on disk.

Every other metric is reference-based, so a system that removes the noise but
leaves the speech buzzy, warbly or spectrally scooped still scores well;
NISQA's discontinuity and coloration dimensions are the "cleaned it but it
hurts to listen to" axes. Never read NISQA alone: it rewards a clean-sounding
rewrite even when speaker identity is destroyed.

Usage: scorenisqa.py [--systems a,b] [--conds substr,substr] [--limit N]
Output: <BENCH_RESULTS>/<system>.nisqa.json
"""

import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors
import common
import observations
import scorebench

SCORER_SHA256 = observations.implementation(__file__, common.__file__,
                                            observations.__file__,
                                            scorebench.__file__)

KEYS = ("mos", "noisiness", "discontinuity", "coloration", "loudness")

# every real family: the board reads their equal mean (scorefiles.real_quality)
DEFAULT_CONDS = ("babble@+4,babble@+0,ambience@-5,clean,real_noisy,"
                 "real_dirty,real_multi,real_wild,expressive_tail@-5,"
                 "reverb_real")

# a primary-speaker segment shorter than this is not scored on its own
MIN_PRIMARY_S = 2.0


def primary_segments(root=None, rows=None):
    """Returns the input's usable segments and primary-speaker flags per item.

    Per real item, from the deletion sidecar's records of INPUT: boundaries
    frozen on the input side, identical for every output. The sidecar must be
    INPUT's, of this set and the whole set, made by the current deletion scorer
    with speaker anchoring, and cover every real item of the manifest; stale or
    unanchored boundaries are refused.
    """
    # pylint: disable-next=import-outside-toplevel
    import scorerealx

    path = os.path.join(root or common.RESULTS, "INPUT.realx.items.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: score INPUT with scorerealx.py"
                         " --primary-anchor first")
    metadata, records = observations.read(path)
    if metadata.get("system") != "INPUT":
        raise SystemExit(f"{path}: not INPUT's records")
    if (metadata.get("set_id") != common.set_id() or
            metadata.get("subset", "all") != "all"):
        raise SystemExit(
            f"{path}: records of set {str(metadata.get('set_id'))[:8]} subset "
            f"{metadata.get('subset', 'all')!r}, not the whole current set")
    if metadata.get("scorer_sha256") != scorerealx.SCORER_SHA256:
        raise SystemExit(
            f"{path}: made by another deletion scorer; re-run scorerealx.py"
            " first")
    real_ids = {
        r["id"]
        for r in (rows if rows is not None else common.manifest(which="all"))
        if r["condition"].startswith("real")
    }
    have = {r["id"] for r in records}
    if not real_ids <= have:
        raise SystemExit(
            f"{path} covers {len(real_ids & have)} of {len(real_ids)} real"
            " items: re-run scorerealx.py --primary-anchor for INPUT")
    # a pass without --primary-anchor leaves the segments but flags none as the
    # primary speaker and stores no anchor similarities: that is not a recording
    # without speech, it is an unanchored pass, and its empty primary set would
    # silently score nothing
    unanchored = [
        r["id"]
        for r in records
        if (r.get("measurements") or {}).get("seg_start") and
        not (r.get("measurements") or {}).get("anchor_sims")
    ]
    if unanchored:
        raise SystemExit(
            f"{path}: {len(unanchored)} records with speech segments carry no"
            f" speaker anchoring (first: {unanchored[0]}): re-run scorerealx.py"
            " --primary-anchor for INPUT")
    out = {}
    for r in records:
        m = r.get("measurements") or {}
        if m.get("seg_start") and m.get("seg_primary"):
            out[r["id"]] = [(int(b), int(e)) for b, e, p in zip(
                m["seg_start"], m["seg_end"], m["seg_primary"]) if p]
    if not out:
        raise SystemExit(
            f"{path} carries no segment boundaries: re-run scorerealx.py")
    return out


def primary_quality(out, segments, min_seconds=MIN_PRIMARY_S):
    """Returns NISQA over the primary speaker's segments that stand alone.

    Segments of ``out`` (16 kHz indices, cut at 48 kHz) long enough to stand
    alone; the item's value is their duration-weighted mean, with the eligible
    duration and its share of the primary speaker's total. Segments are never
    joined: a stitched signal has discontinuities of its own, which is what a
    quality model reacts to.
    """
    total = sum(e - b for b, e in segments) / 16000
    eligible = [(b, e) for b, e in segments if (e - b) / 16000 >= min_seconds]
    if not eligible:
        return {
            "nisqa_primary_eligible_s": 0.0,
            "nisqa_primary_coverage": 0.0 if total else float("nan"),
        }
    weights, values = [], []
    for b, e in eligible:
        mos = common.nisqa(out[b * 3:e * 3])[0]
        if mos == mos:
            weights.append((e - b) / 16000)
            values.append(mos)
    covered = sum((e - b) for b, e in eligible) / 16000
    return {
        "nisqa_mos_primary": (float(np.average(values, weights=weights))
                              if values else float("nan")),
        "nisqa_primary_eligible_s": covered,
        "nisqa_primary_coverage": covered / total if total else float("nan"),
    }


def assemble(name, rows, reuse=None, hashes=None, primary=None):
    """Returns per-item NISQA records over ``rows``.

    An item whose output samples are unchanged since an earlier pass of this
    sidecar keeps its record (``reuse``, by content identity). With ``primary``
    (real items' frozen segments), a real item also gets the primary-speaker
    values.
    """
    hashes = hashes or common.file_hashes()
    reuse = reuse or {}
    records, counts = [], collections.Counter()
    for it in rows:
        record = observations.item(it)
        records.append(record)
        path = (it["mix_path"] if name == "INPUT" else os.path.join(
            common.OUT_ROOT, name, it["id"]))
        if not os.path.exists(path):
            record["output_status"] = "missing"
            continue
        out = common.load48(path)
        record["identity"] = scorebench.item_identity(
            it, scorebench.sample_hash(out), hashes)
        segs = ((primary or {}).get(it["id"])
                if primary is not None and not it.get("clean_path") else None)
        if primary is not None and not it.get("clean_path"):
            record["identity"]["semantics"]["primary_segments"] = (
                anchors.digest(segs or []))
        old = reuse.get(scorebench.reuse_key(record["identity"]))
        if old is not None:
            record["metrics"] = dict(measured(old))
            record["output_status"] = old["output_status"]
            counts["reused"] += 1
            continue
        values = common.nisqa(out)
        record["metrics"] = observations.finite({
            f"nisqa_{k}": v for k, v in zip(KEYS, values)
        })
        if segs is not None:
            record["metrics"].update(
                observations.finite(primary_quality(out, segs)))
        if not np.isfinite(values[0]):
            record["output_status"] = "metric_failed"
        counts["scored"] += 1
    if reuse:
        print(
            f"  reused {counts['reused']} records, scored {counts['scored']}"
            " items",
            flush=True,
        )
    return records


def measured(record):
    """Returns a record's NISQA values, inside the published cohort or not.

    Values in the cohort sit under ``metrics``; those outside it are kept under
    ``measurements``, so a paired comparison reads the cohort the board reads.
    """
    return (record.get("metrics") or
            record.get("measurements", {}).get("nisqa") or {})


def cohort(rows, want, limit):
    """Returns the items whose NISQA the published mean reads.

    Every item of each condition, or the first ``limit`` of each when a limit is
    given (a quick look, never published).
    """
    by = collections.defaultdict(list)
    for r in rows:
        if r["condition"] in want:
            by[r["condition"]].append(r)
    return {r["id"] for c in want for r in (by[c][:limit] if limit else by[c])}


def mark(records, members):
    """Puts each record's values where its cohort membership says.

    ``metrics`` inside the cohort, ``measurements`` outside it, so every reader
    of per-item metrics sees the board's cohort.
    """
    for record in records:
        values = measured(record)
        record["cohort"] = record["id"] in members
        record["metrics"] = dict(values) if record["cohort"] else {}
        record["measurements"] = ({} if record["cohort"] else ({
            "nisqa": dict(values)
        } if values else {}))


def aggregate(records, rows, want, limit):
    """Returns the sidecar file for ``rows``.

    Per condition, the mean over its items that carry a measurement (the first
    ``limit`` when one is given), with the condition's size beside it. Pure: any
    view of the set is this function over the whole-set records of its items.
    """
    by_id = {r["id"]: r for r in records}
    by = collections.defaultdict(list)
    for r in rows:
        if r["condition"] in want:
            by[r["condition"]].append(r)
    out = {}
    for cond in want:
        selected = by[cond][:limit] if limit else by[cond]
        values = [
            measured(by_id[r["id"]])
            for r in selected
            if r["id"] in by_id and measured(by_id[r["id"]])
        ]
        if not values:
            continue
        a = np.array(
            [[observations.revive(m.get(f"nisqa_{k}"))
              for k in KEYS]
             for m in values],
            dtype=float,
        )
        out[cond] = {
            f"nisqa_{k}": float(np.nanmean(a[:, i])) for i, k in enumerate(KEYS)
        }
        out[cond]["n"] = int(np.isfinite(a[:, 0]).sum())
        out[cond]["n_expected"] = len(selected)
        out[cond]["n_missing"] = out[cond]["n_expected"] - out[cond]["n"]
        prim = [
            observations.revive(m["nisqa_mos_primary"])
            for m in values
            if "nisqa_mos_primary" in m
        ]
        prim = [v for v in prim if v == v]
        # the diagnostic: quality where the primary speaker talks, on frozen
        # segments
        if prim:
            out[cond]["nisqa_mos_primary"] = float(np.mean(prim))
            out[cond]["nisqa_primary_items"] = len(prim)
            eligible = [
                observations.revive(m.get("nisqa_primary_eligible_s"))
                for m in values
            ]
            # an item without primary segments revives to NaN, which is
            # truthy: only the finite durations are summed
            out[cond]["nisqa_primary_eligible_s"] = float(
                sum(v for v in eligible
                    if isinstance(v, (int, float)) and math.isfinite(v)))
            cov = [
                observations.revive(m.get("nisqa_primary_coverage"))
                for m in values
                if m.get("nisqa_primary_coverage") is not None
            ]
            cov = [c for c in cov if c == c]
            out[cond]["nisqa_primary_coverage"] = (float(np.mean(cov))
                                                   if cov else float("nan"))
    return out


def write(name, records, out, rows, members, limit=0, outputs=None):
    mark(records, members)
    observations.sidecar(
        os.path.join(common.RESULTS, f"{name}.nisqa.items.jsonl"),
        records,
        name,
        common.set_id(),
        common.SUBSET,
        common.is_sealed(rows),
        SCORER_SHA256,
        outputs_sha256=outputs,
    )
    if out:
        # the leaderboard ignores a sidecar from another build
        out["_set_id"] = common.set_id()
        out["_subset"] = common.SUBSET  # and one from another subset of the set
        out.update(anchors.stamp())
        out["_sealed"] = common.is_sealed(rows)
        out["_scorer_sha256"] = SCORER_SHA256
        out["_outputs_sha256"] = outputs  # the render these records read
        # the first items of each condition: a quick look, never a board row
        if limit:
            out["_selection"] = {"limit": limit}
        # whole or absent: parallel passes each re-validate INPUT
        path = os.path.join(common.RESULTS, f"{name}.nisqa.json")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="")
    ap.add_argument("--conds", default=DEFAULT_CONDS)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help=("items per condition the mean reads (0, the published rule: every"
              " item); every item of the conditions is scored, so a view can be"
              " derived"),
    )
    ap.add_argument(
        "--reuse-from",
        default="",
        metavar="DIR",
        help=(
            "results of an earlier pass of this sidecar: unchanged outputs keep"
            " their records"),
    )
    ap.add_argument(
        "--aggregate-only",
        default="",
        metavar="DIR",
        help="derive this subset's files from the whole-set records under DIR",
    )
    ap.add_argument(
        "--segments",
        choices=("clip", "primary"),
        default="clip",
        help=(
            "primary: also score each real item's primary-speaker segments on"
            " their own (boundaries frozen from the input's deletion sidecar),"
            " a diagnostic"),
    )
    args = ap.parse_args()

    want = [c for c in args.conds.split(",") if c]
    rows = common.manifest()
    common.guard_private_results(rows)
    selected = [r for r in rows if r["condition"] in want]

    systems = [s for s in args.systems.split(",") if s] or sorted(
        os.path.basename(d)
        for d in glob.glob(common.OUT_ROOT + "/*")
        if os.path.isdir(d))
    if systems != ["REF"]:
        systems = ["INPUT"] + [s for s in systems if s != "INPUT"]
    os.makedirs(common.RESULTS, exist_ok=True)
    for sysname in systems:
        if args.aggregate_only:
            src = os.path.join(args.aggregate_only,
                               f"{sysname}.nisqa.items.jsonl")
            ids = {r["id"] for r in selected}
            problem, source, records = scorebench.view_records(
                src, sysname, ids, SCORER_SHA256)
            if problem:
                print(f"skip {sysname}: {problem}", flush=True)
                continue
            outputs = source.get("outputs_sha256")
        else:
            problem = sysname != "INPUT" and scorebench.render_problem(sysname)
            if problem:
                print(f"skip {sysname}: {problem}", flush=True)
                continue
            own = os.path.join(common.RESULTS, f"{sysname}.nisqa.items.jsonl")
            reuse, why = {}, "no earlier records"
            if args.reuse_from:
                reuse, why = scorebench.reuse_index(
                    os.path.join(args.reuse_from,
                                 f"{sysname}.nisqa.items.jsonl"),
                    sysname,
                    SCORER_SHA256,
                )
                why = f"reuse from {args.reuse_from}: {why}"
            # never skipped as scored: re-validated item by item
            elif os.path.isfile(own):
                reuse, why = scorebench.reuse_index(own, sysname, SCORER_SHA256)
                why = f"existing records: {why}"
            print(f"  {why}", flush=True)
            primary = primary_segments() if args.segments == "primary" else None
            records = assemble(sysname, selected, reuse=reuse, primary=primary)
            outputs = common.outputs_digest(sysname)
        out = aggregate(records, selected, want, args.limit)
        write(
            sysname,
            records,
            out,
            rows,
            cohort(selected, want, args.limit),
            args.limit,
            outputs,
        )
        if out:
            print(
                f"{sysname:24s} " +
                "  ".join(f"{c}: mos {out[c]['nisqa_mos']:.2f}" for c in want
                          if c in out),
                flush=True,
            )


if __name__ == "__main__":
    main()
