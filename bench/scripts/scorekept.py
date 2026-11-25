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
"""What the voice kept where the main scorer does not look.

The vocal sounds that are not speech, and the spectrum of speech that needed no
cleaning.

    python bench/scripts/scorekept.py [--systems a,b]

**Voice outside speech** (the ``nonverbal`` conditions). The main scorer reads
retention over the blocks a voice-activity detector calls speech, so a laugh, a
breath or a sigh it does not call speech is never read. Here the blocks are the
rest of the voice: the 100 ms blocks of the clean recording that hold sound
(within ``VOCAL_DB`` of its loudest block), that the detector leaves out, and
that the bed does not bury (the voice of the block no more than ``BURIED_DB``
under the bed of the block, the lowest level the set mixes at; nobody keeps what
nobody could hear). The recording is one voice in a quiet room, so whatever
sounds there is that voice. A block is lost when the part of the output that is
coherent with it, less the system's level calibration, falls under
``levels.LOST_DB``: the main scorer's rule for a speech block (``levels.lost``).
``nv_lost_pct`` is the share of those blocks that are lost, pooled over the
condition, so a long laugh weighs what it lasts. Consecutive blocks are one
event, and ``nv_events_lost_pct`` reads the same for whole events (the output
projected on the recording over the event).

**Clean-speech magnitude fidelity** (the ``clean`` condition). ``mag_snr`` is
the energy of the input's magnitude spectrogram over the energy of its
difference from the output's, in dB, after the one gain that best matches them:
a signal-to-error ratio that does not read phase. A waveform ratio calls a
change of phase an error; this one counts only a spectrum that changed.
``mag_snr_nb`` reads the same inside 0-8 kHz.

Delay and polarity are undone as the main scorer does. Output:
``<system>.kept.json`` and the per-item records ``<system>.kept.items.jsonl``
beside the other sidecars; views derive from the whole-set records
(``--aggregate-only``) and unchanged items are reused by content.
"""

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import torch

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors
import common
import levels
import observations
import scorebench

SCORER_SHA256 = observations.implementation(__file__, common.__file__,
                                            observations.__file__,
                                            scorebench.__file__,
                                            levels.__file__)

FRAME = 4800  # 100 ms at 48 kHz: the scorer's block
# a block holds a vocal sound when it sits within this of the loudest block
VOCAL_DB = 30.0
# and is there to keep when the voice is no further than this under the bed
BURIED_DB = -10.0
N_FFT, HOP = 2048, 480
MAG_KEYS = ("mag_snr", "mag_snr_nb")
# derived per view, under the view's level calibration
ITEM_KEYS = ("nv_lost", "nv_lost_pct")


def family(condition: str) -> str:
    return condition.split("@", 1)[0]


def wanted(row: dict) -> bool:
    if not row.get("clean_path"):
        return False
    condition = row["condition"]
    return family(condition).startswith("nonverbal") or condition == "clean"


def vocal_blocks(ref: torch.Tensor, speech: torch.Tensor,
                 mix: torch.Tensor) -> torch.Tensor:
    """Returns the blocks of a clean recording that hold vocal sound to keep.

    They hold sound, are not speech and are not buried.

    Args:
        ref: The clean recording at 48 kHz.
        speech: The detector's speech blocks of it (bool, one per 100 ms).
        mix: The mixture the recording sits in, sample-aligned with it (the
            recording itself when nothing was added).

    Returns:
        A bool mask over the recording's whole blocks.
    """
    _, _, loud = common.coherent_blocks(ref, ref, FRAME, VOCAL_DB)
    n = len(loud)
    voice = ref[:n * FRAME].reshape(-1, FRAME).pow(2).mean(1)
    residual = mix[:n * FRAME] - ref[:n * FRAME]
    bed = residual.reshape(-1, FRAME).pow(2).mean(1)
    heard = 10 * torch.log10((voice + 1e-12) / (bed + 1e-12)) >= BURIED_DB
    mask = torch.zeros(n, dtype=torch.bool)
    k = min(n, len(speech))
    mask[:k] = loud[:k] & ~speech[:k] & heard[:k]
    return mask


def events(mask: torch.Tensor) -> list:
    """Returns the runs of consecutive blocks in a mask.

    As [(first block, last block + 1)].
    """
    out, start = [], None
    for b, on in enumerate(mask.tolist()):
        if on and start is None:
            start = b
        elif not on and start is not None:
            out.append((start, b))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def event_kept_db(out: torch.Tensor, ref: torch.Tensor, span: tuple) -> float:
    """Returns how much of one event the output keeps, in dB.

    20 log10 of the projection of the output on the recording over the event.
    """
    start, end = span[0] * FRAME, span[1] * FRAME
    o, r = out[start:end], ref[start:end]
    o, r = o - o.mean(), r - r.mean()
    coefficient = float((o * r).sum() / (r.pow(2).sum() + 1e-9))
    return float(np.clip(20 * np.log10(max(coefficient, 1e-4)), -80.0, 20.0))


def mag_snr(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Returns the magnitude spectrograms' signal-to-error ratio, in dB.

    After the best single gain.
    """
    n = min(len(out), len(ref))
    if n < N_FFT:
        return float("nan")
    window = torch.hann_window(N_FFT)
    x = torch.stft(ref[:n], N_FFT, HOP, window=window, return_complex=True)
    y = torch.stft(out[:n], N_FFT, HOP, window=window, return_complex=True)
    x, y = x.abs(), y.abs()
    scale = float((x * y).sum() / (y * y).sum().clamp(min=1e-12))
    error = (scale * y - x).pow(2).sum().clamp(min=1e-12)
    ratio = float(10 * torch.log10(x.pow(2).sum().clamp(min=1e-12) / error))
    return min(120.0, max(-20.0, ratio))


def score_item(it: dict, out: torch.Tensor, vad) -> tuple:
    """(metrics, measurements) of one item after the main scorer's alignment."""
    ref, mix = common.load48(it["clean_path"]), common.load48(it["mix_path"])
    n = min(len(ref), len(mix))
    lag, sign = common.best_alignment(out, ref)
    o, r = sign * common.apply_lag(out, lag, n), ref[:n]
    if it["condition"] == "clean":
        return {
            "mag_snr": mag_snr(o, r),
            "mag_snr_nb": mag_snr(common.nb8k(o), common.nb8k(r)),
        }, {}
    r16, segments = scorebench.speech_segments(vad, ref, it["clean_path"])
    active = torch.zeros(len(r16), dtype=torch.bool)
    for s in segments:
        active[s["start"]:s["end"]] = True
    mask = vocal_blocks(r, common.block_mask(active, len(r) // FRAME), mix[:n])
    kept, _, _ = common.coherent_blocks(o, r, FRAME)
    spans = events(mask)
    return {
        "nv_blocks": int(mask.sum()),
        "nv_events": len(spans)
    }, {
        "nv_kept_db": [float(v) for v in kept[mask[:len(kept)]]],
        "nv_event_kept_db": [event_kept_db(o, r, span) for span in spans],
        "nv_event_blocks": [b - a for a, b in spans],
    }


def assemble(name: str,
             rows: list,
             reuse: dict | None = None,
             hashes=None,
             vad=None) -> list:
    """Returns per-item records over ``rows``.

    Reused by content identity where output, mixture and reference are unchanged
    since an earlier pass of this sidecar.
    """
    hashes = hashes or common.file_hashes()
    reuse = reuse or {}
    records, counts = [], collections.Counter()
    for it in rows:
        record = observations.item(it)
        records.append(record)
        op = (it["mix_path"] if name == "INPUT" else
              (it["clean_path"] if name == "REF" else os.path.join(
                  common.OUT_ROOT, name, it["id"])))
        if not os.path.exists(op):
            record["output_status"] = "missing"
            continue
        out = common.load48(op)
        record["identity"] = scorebench.item_identity(
            it, scorebench.sample_hash(out), hashes)
        old = reuse.get(scorebench.reuse_key(record["identity"]))
        if old is not None:
            record["metrics"] = {
                k: v for k, v in old["metrics"].items() if k not in ITEM_KEYS
            }
            record["measurements"] = dict(old.get("measurements") or {})
            counts["reused"] += 1
            continue
        if vad is None and it["condition"] != "clean":
            vad = common.Vad()
        record["metrics"], record["measurements"] = score_item(it, out, vad)
        counts["scored"] += 1
    if reuse:
        print(
            f"  reused {counts['reused']} records, scored {counts['scored']}"
            " items",
            flush=True,
        )
    return records


def level_calibration(results: str, name: str) -> float:
    """Returns the system's level calibration from its base scores.

    Kept levels are read less it.
    """
    path = os.path.join(results, f"{name}.metrics.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing: score {name} with scorebench.py first (kept"
            " levels are read less the system's level calibration)")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if (doc.get("_set_id") != common.set_id() or
            doc.get("_subset", "all") != common.SUBSET):
        raise SystemExit(
            f"{path} was scored on another build or subset of the set")
    return float(doc.get("_level_cal_db") or 0.0)


def aggregate(records: list, rows: list, level_cal_db: float) -> dict:
    """Returns the sidecar file for ``rows``.

    Every item's lost share is written into its record. Pure, so any view of the
    set derives from the whole-set records under its own level calibration, as
    the main scorer's lost-block share does.
    """
    by_id = {r["id"]: r for r in records}
    by = collections.defaultdict(list)
    for r in rows:
        if wanted(r):
            by[r["condition"]].append(r)
    out = {}
    for cond, items in sorted(by.items()):
        have = [
            by_id[r["id"]]
            for r in items
            if r["id"] in by_id and by_id[r["id"]].get("output_status") == "ok"
            and by_id[r["id"]].get("metrics")
        ]
        if not have:
            continue
        # metric_counts: the items that carry the observation, which the board's
        # coverage rule reads instead of the items scored (a recording without a
        # laugh observes nothing)
        cell = {
            "n": len(have),
            "n_missing": len(items) - len(have),
            "metric_counts": {},
        }
        if cond == "clean":
            for key in MAG_KEYS:
                vals = [r["metrics"].get(key) for r in have]
                vals = [
                    v for v in vals
                    if isinstance(v, (int, float)) and np.isfinite(v)
                ]
                if vals:
                    cell[key] = float(np.mean(vals))
                    cell["metric_counts"][key] = len(vals)
            out[cond] = cell
            continue
        lost, total, event_levels, n_items = 0, 0, [], 0
        for rec in have:
            measured = rec.get("measurements") or {}
            blocks = np.array(measured.get("nv_kept_db") or [], dtype=float)
            for key in ITEM_KEYS:
                rec["metrics"].pop(key, None)
            gone = int(np.sum(levels.lost(blocks, level_cal_db)))
            # with nv_blocks, what a paired composite pools
            rec["metrics"]["nv_lost"] = gone
            if len(blocks):
                rec["metrics"]["nv_lost_pct"] = 100.0 * gone / len(blocks)
                n_items += 1
            lost, total = lost + gone, total + len(blocks)
            event_levels += list(measured.get("nv_event_kept_db") or [])
        if not total:
            continue
        events_lost = levels.lost(event_levels, level_cal_db)
        cell.update(
            nv_lost_pct=100.0 * lost / total,
            nv_events_lost_pct=100.0 * float(np.mean(events_lost)),
            nv_items=n_items,
            nv_blocks=total,
            nv_events=len(events_lost),
            metric_counts={
                "nv_lost_pct": n_items,
                "nv_events_lost_pct": n_items,
            },
        )
        out[cond] = cell
    return out


def write(name: str,
          records: list,
          out: dict,
          rows: list,
          level_cal_db: float,
          outputs: str | None = None) -> None:
    observations.sidecar(
        os.path.join(common.RESULTS, f"{name}.kept.items.jsonl"),
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
        out["_level_cal_db"] = level_cal_db
        out["_scorer_sha256"] = SCORER_SHA256
        out["_outputs_sha256"] = outputs  # the render these records read
        out.update(anchors.stamp())
        out["_sealed"] = common.is_sealed(rows)
        path = os.path.join(common.RESULTS, f"{name}.kept.json")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, path)


def describe(out: dict) -> str:
    shown = []
    for cond in sorted(c for c in out if not c.startswith("_")):
        cell = out[cond]
        if "nv_lost_pct" in cell:
            shown.append(
                f"{cond}: {cell['nv_lost_pct']:.1f} % of {cell['nv_blocks']}"
                f" blocks, {cell['nv_events_lost_pct']:.1f} % of"
                f" {cell['nv_events']} events lost")
        elif "mag_snr" in cell:
            shown.append(f"{cond}: magnitude SNR {cell['mag_snr']:.1f} dB")
    return "  ".join(shown)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="")
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
    rows = common.manifest()
    common.guard_private_results(rows)
    selected = [r for r in rows if wanted(r)]
    systems = [s for s in a.systems.split(",") if s] or sorted(
        os.path.basename(d)
        for d in glob.glob(common.OUT_ROOT + "/*")
        if os.path.isdir(d))
    os.makedirs(common.RESULTS, exist_ok=True)
    for name in systems:
        if a.aggregate_only:
            src = os.path.join(a.aggregate_only, f"{name}.kept.items.jsonl")
            ids = {r["id"] for r in selected}
            problem, source, records = scorebench.view_records(
                src, name, ids, SCORER_SHA256)
            if problem:
                print(f"skip {name}: {problem}", flush=True)
                continue
            outputs = source.get("outputs_sha256")
        else:
            exempt = name in ("INPUT", "REF")
            problem = not exempt and scorebench.render_problem(name)
            if problem:
                print(f"skip {name}: {problem}", flush=True)
                continue
            own = os.path.join(common.RESULTS, f"{name}.kept.items.jsonl")
            reuse, why = {}, "no earlier records"
            if a.reuse_from:
                reuse, why = scorebench.reuse_index(
                    os.path.join(a.reuse_from, f"{name}.kept.items.jsonl"),
                    name,
                    SCORER_SHA256,
                )
                why = f"reuse from {a.reuse_from}: {why}"
            # never skipped as scored: re-validated item by item
            elif os.path.isfile(own):
                reuse, why = scorebench.reuse_index(own, name, SCORER_SHA256)
                why = f"existing records: {why}"
            print(f"  {why}", flush=True)
            records = assemble(name, selected, reuse=reuse)
            outputs = common.outputs_digest(name)
        level = level_calibration(common.RESULTS, name)
        out = aggregate(records, selected, level)
        write(name, records, out, rows, level, outputs)
        print(f"{name:24s} {describe(out)}", flush=True)


if __name__ == "__main__":
    main()
