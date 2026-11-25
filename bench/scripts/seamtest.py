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
"""Chunk seams: whether the output depends on where the windows fell.

It also reads whether the voice is hurt at the seams::

    seamtest.py --model voice-isolator-v1 [--conds long@+4] [--limit 12]
        [--windows 8/2,5/1] [--shift-seconds 1.0] [--precision fp32]
        [--out staging/ops/<tag>.seams.json]

The isolator processes long audio in overlapped windows (``chunk_seconds`` /
``overlap_seconds``, crossfaded), and its hint frontend has windows of its own.
Two passes per item tell the two things that matter. The clip processed as it is
and processed again with ``--shift-seconds`` of silence in front (removed
afterwards) differ only where the window placement mattered, so the per-block
difference between the two outputs is the position dependence of the whole
processing path, inside the crossfade zones and away from them. On paired items
the voice kept (the coherent projection on the reference) and the residual (bed
and artifacts) inside the zones against outside them are the damage the seams
do; the worst block of a zone is judged against the worst block of stretches of
the same length away from the seams, the same order statistic on both sides, so
a system whose blocks merely fluctuate reads zero. Clips shorter than one window
have no seams; their shift difference is still reported (the frontend's own
windows and any non-determinism show there). Reports per window setting; the
block is 100 ms.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

FRAME = 4800  # 100 ms at 48 kHz
DIFF_FLOOR_DB, DIFF_CEIL_DB = -100.0, 40.0


def window_samples(chunk_seconds, overlap_seconds, sr=common.SR):
    """Returns (chunk, overlap, hop) in samples.

    Rounded and capped as the separator does.
    """
    chunk = int(chunk_seconds * sr)
    overlap = min(int(overlap_seconds * sr), chunk // 2)
    return chunk, overlap, chunk - overlap


def seam_zones(
    n,
    chunk_seconds,
    overlap_seconds,
    shift_seconds=0.0,
    margin_seconds=0.2,
    sr=common.SR,
    frame=FRAME,
):
    """Returns the crossfade zones of an ``n``-sample clip as block ranges.

    Each zone is [first_block, last_block).

    Window k (k >= 1) starts at k * hop and fades in over ``overlap`` samples
    while window k - 1 fades out; that zone exists only when window k - 1 did
    not already reach the end. A clip processed with ``shift_seconds`` of
    silence in front sees the same zones ``shift`` earlier in its own
    coordinates. ``margin_seconds`` widens every zone on both sides: seam damage
    does not stop at the crossfade's edge.
    """
    chunk, overlap, hop = window_samples(chunk_seconds, overlap_seconds, sr)
    shift, margin = int(shift_seconds * sr), int(margin_seconds * sr)
    n_blocks = n // frame
    zones = []
    k = 1
    while (k - 1) * hop + chunk < n + shift:
        start = k * hop - shift
        if start >= n:
            break
        lo, hi = max(0, start - margin), min(n, start + overlap + margin)
        b0, b1 = lo // frame, min(n_blocks, -(-hi // frame))
        if b1 > b0:
            zones.append((b0, b1))
        k += 1
    return zones


def seam_mask(n_blocks, zones):
    m = torch.zeros(n_blocks, dtype=torch.bool)
    for b0, b1 in zones:
        m[b0:b1] = True
    return m


def control_windows(n_blocks, zones, width):
    """Returns ``width``-block windows tiled over the blocks outside every zone.

    Non-overlapping: the statistic read over a zone (its worst block) is read
    over these too, so a seam is judged against what a stretch of the same
    length shows anywhere else.
    """
    seam = seam_mask(n_blocks, zones)
    out, b = [], 0
    while b < n_blocks:
        if bool(seam[b]):
            b += 1
            continue
        e = b
        while e < n_blocks and not bool(seam[e]) and e - b < width:
            e += 1
        if e - b == width:
            out.append((b, e))
        b = e
    return out


def block_diff_db(a, b, base, frame=FRAME, floor_db=40.0):
    """Returns per-block 10 log10 |a - b|^2 / |base|^2 and the usable blocks.

    The blocks where ``base`` is measurable.
    """
    n = min(len(a), len(b), len(base))
    n = n // frame * frame
    d = (a[:n] - b[:n]).reshape(-1, frame).pow(2).mean(1)
    e = base[:n].reshape(-1, frame).pow(2).mean(1)
    loud = (10 * torch.log10(e + 1e-12)
            > 10 * torch.log10(e.max() + 1e-12) - floor_db)
    # identical signals read the floor exactly
    ratio = d / (e + 1e-12) + 10**(DIFF_FLOOR_DB / 10)
    return (10 * torch.log10(ratio)).clamp(DIFF_FLOOR_DB, DIFF_CEIL_DB), loud


def _mean(t):
    return float(t.mean()) if len(t) else float("nan")


def _zone_extremes(values, loud, zones, fn):
    """Returns ``fn`` (min or max) of ``values`` over each zone's loud blocks.

    Zones without a loud block are skipped.
    """
    out = []
    for b0, b1 in zones:
        v = values[b0:b1][loud[b0:b1]]
        if len(v):
            out.append(float(fn(v)))
    return out


def measure_item(
    run,
    x,
    ref,
    chunk_seconds,
    overlap_seconds,
    shift_seconds=1.0,
    margin_seconds=0.2,
    frame=FRAME,
):
    """Returns every per-item number of the report for one clip.

    Both passes go through ``run(x, chunk_seconds, overlap_seconds)``; ``ref``
    is None for a reference-free clip.
    """
    n = len(x)
    y0 = run(x, chunk_seconds, overlap_seconds)[:n]
    shift = int(shift_seconds * common.SR)
    padded = torch.nn.functional.pad(x, (shift, 0))
    y1 = run(padded, chunk_seconds, overlap_seconds)[shift:shift + n]
    base = ref if ref is not None else x
    diff, loud = block_diff_db(y0, y1, base, frame)
    n_blocks = len(diff)
    own = seam_zones(n,
                     chunk_seconds,
                     overlap_seconds,
                     0.0,
                     margin_seconds,
                     frame=frame)
    both = own + seam_zones(
        n,
        chunk_seconds,
        overlap_seconds,
        shift_seconds,
        margin_seconds,
        frame=frame,
    )
    seam_both = seam_mask(n_blocks, both)
    seam_own = seam_mask(n_blocks, own)
    width = max((b1 - b0 for b0, b1 in own), default=0)
    ctrl_both = control_windows(n_blocks, both, width) if width else []
    ctrl_own = control_windows(n_blocks, own, width) if width else []
    seam_max = _zone_extremes(diff, loud, both, torch.max)
    ctrl_max = _zone_extremes(diff, loud, ctrl_both, torch.max)
    d = {
        "seconds":
            n / common.SR,
        "n_blocks":
            n_blocks,
        "n_seam_blocks":
            int((seam_own & loud).sum()),
        "n_else_blocks":
            int((~seam_both & loud).sum()),
        "n_control_windows":
            len(ctrl_own),
        "shift_diff_db":
            _mean(diff[loud]),
        "shift_diff_seam_mean_db":
            _mean(diff[seam_both & loud]),
        "shift_diff_else_mean_db":
            _mean(diff[~seam_both & loud]),
        "shift_diff_seam_max_db":
            (float(np.mean(seam_max)) if seam_max else float("nan")),
        "shift_diff_else_max_db":
            (float(np.mean(ctrl_max)) if ctrl_max else float("nan")),
    }
    if bool(loud.any()):
        masked = torch.where(loud, diff,
                             torch.full_like(diff, DIFF_FLOOR_DB - 1))
        worst = int(masked.argmax())
        d.update({
            "worst_block_s": worst * frame / common.SR,
            "worst_diff_db": float(diff[worst]),
            "worst_in_seam": bool(seam_both[worst]),
        })
    if ref is not None:
        kept, resid, loudr = common.coherent_blocks(y0, ref, frame)
        m = min(len(kept), n_blocks)
        kept, resid, loudr, so = kept[:m], resid[:m], loudr[:m], seam_own[:m]
        else_kept = kept[~so & loudr]
        else_resid = resid[~so & loudr]
        dips = _zone_extremes(kept, loudr, own, torch.min)
        ctrl_dips = _zone_extremes(kept, loudr, ctrl_own, torch.min)
        d.update({
            "kept_seam_db":
                _mean(kept[so & loudr]),
            "kept_else_db":
                _mean(else_kept),
            "seam_penalty_db":
                _mean(kept[so & loudr]) - _mean(else_kept),
            # the worst block of a zone against the worst block of a stretch of
            # the same length away from the seams: the same order statistic on
            # both sides
            "seam_dip_db": (float(np.mean(dips)) - float(np.mean(ctrl_dips))
                            if dips and ctrl_dips else float("nan")),
            "resid_seam_db":
                _mean(resid[so & loudr]),
            "resid_else_db":
                _mean(else_resid),
            "seam_resid_excess_db":
                (_mean(resid[so & loudr]) - _mean(else_resid)),
        })
    return d


SUMMARY_KEYS = (
    "shift_diff_db",
    "shift_diff_seam_mean_db",
    "shift_diff_seam_max_db",
    "shift_diff_else_mean_db",
    "shift_diff_else_max_db",
    "kept_seam_db",
    "kept_else_db",
    "seam_penalty_db",
    "seam_dip_db",
    "resid_seam_db",
    "resid_else_db",
    "seam_resid_excess_db",
)


def aggregate(items):
    """Means over items of every per-item number (NaN-aware), plus counts."""
    out = {
        "n_items": len(items),
        "n_with_seams": sum(1 for d in items if d["n_seam_blocks"]),
    }
    for k in SUMMARY_KEYS:
        vals = [d[k] for d in items if k in d and d[k] == d[k]]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    worst = [
        d["worst_in_seam"]
        for d in items
        if "worst_in_seam" in d and d["n_seam_blocks"]
    ]
    out["worst_in_seam_frac"] = float(np.mean(worst)) if worst else float("nan")
    return out


def parse_windows(spec, default):
    """"8/2,5/1" -> [(8.0, 2.0), (5.0, 1.0)]; empty -> [default]."""
    if not spec:
        return [tuple(float(v) for v in default)]
    out = []
    for part in spec.split(","):
        c, o = part.split("/")
        out.append((float(c), float(o)))
    return out


def label(chunk_seconds, overlap_seconds):
    return f"{chunk_seconds:g}/{overlap_seconds:g}"


def main():
    ap = argparse.ArgumentParser()
    common.add_model_args(ap)
    ap.set_defaults(model="voice-isolator-v1")
    ap.add_argument(
        "--conds",
        default="long@+4",
        help="conditions (comma-separated) or a prefix",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--windows",
        default="",
        help='"chunk/overlap,..." in seconds; default: the model\'s own window',
    )
    ap.add_argument("--shift-seconds", type=float, default=1.0)
    ap.add_argument("--margin-seconds", type=float, default=0.2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = common.pick_rows(common.manifest(), a.conds, a.limit)
    if not rows:
        raise SystemExit(f"no items match {a.conds!r} in {common.EVAL}")
    vi, run = common.isolator_runner(a)
    windows = parse_windows(a.windows, vi.model.default_window())
    report = {
        "check": "seams",
        "model": common.model_tag(a.model),
        "precision": a.precision,
        "device": str(vi.device),
        "mode": vi.mode,
        "lookahead_frames": vi.lookahead_frames,
        "set_id": common.set_id(),
        "conds": a.conds,
        "shift_seconds": a.shift_seconds,
        "margin_seconds": a.margin_seconds,
        "frame": FRAME,
        "windows": {},
    }
    for chunk_seconds, overlap_seconds in windows:
        items = []
        for r in rows:
            x = common.load48(r["mix_path"])
            ref = common.load48(
                r["clean_path"]) if r.get("clean_path") else None
            d = measure_item(
                run,
                x,
                ref,
                chunk_seconds,
                overlap_seconds,
                a.shift_seconds,
                a.margin_seconds,
            )
            d["id"] = r["id"]
            items.append(d)
            print(
                f"  {label(chunk_seconds, overlap_seconds)} {r['id']}: shift"
                f" diff seam {d['shift_diff_seam_mean_db']:6.1f} / else"
                f" {d['shift_diff_else_mean_db']:6.1f} dB" +
                (f", seam dip {d['seam_dip_db']:5.1f} dB"
                 if "seam_dip_db" in d else ""),
                flush=True,
            )
        report["windows"][label(chunk_seconds, overlap_seconds)] = {
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "summary": aggregate(items),
            "items": items,
        }
    out = common.ops_path(a.model, "seams", a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n{'window':>8s} {'items':>5s} {'shift seam':>10s}"
          f" {'shift else':>10s} {'seam dip':>8s} {'resid xs':>8s}"
          f" {'worst@seam':>10s}")
    for w, d in report["windows"].items():
        s = d["summary"]
        print(
            f"{w:>8s} {s['n_items']:5d} {s['shift_diff_seam_mean_db']:10.1f} "
            f"{s['shift_diff_else_mean_db']:10.1f} {s['seam_dip_db']:8.1f} "
            f"{s['seam_resid_excess_db']:8.1f} {s['worst_in_seam_frac']:10.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
