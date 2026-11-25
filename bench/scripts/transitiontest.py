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
"""Recovery after a background change: a bed that starts or stops mid-speech.

    transitiontest.py --model voice-isolator-v1
        [--conds ambience@+0,babble@+0,stationary@+0,traffic@+0] [--limit 16]
        [--ramp-ms 20] [--tolerance-db 3]
        [--out staging/ops/<tag>.transitions.json]

The set's beds run for the whole clip; real backgrounds start and stop (a door,
a fan, a passing car). From every paired item, bed = mix - ref (exact: mix and
reference share one gain), and two stimuli are built: ``onset``, the bed enters
at the centre of the speech, and ``offset``, it leaves there. Every output is
read per 100 ms block against the unchanged reference (voice kept, the coherent
projection; residual, bed leak and artifacts) and compared block by block with
the model's own output on the steady stimulus of the new state: the whole mix
for an onset, the clean reference for an offset. That pairing removes the speech
content from the comparison (a pause reads the same in both), so what remains is
the model's reaction to the change.

For each stimulus the test reports the dip (the worst block of the first second
after the change, against the steady output at the same block), the recovery
time (blocks until the output is back within ``--tolerance-db`` of the steady
one and stays there), the leak (residual over the first 300 ms after the change,
against the steady residual), the pre-echo (the worst block of the second before
the change against the steady output of the old state: a windowed model that
reacts before the change shows it here), and the steady delta (the median
difference from one second after the change to the end, which a model that
converges reads as zero). A model that adapts instantly reads zero everywhere.
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

FRAME = 4800
AFTER_FROM = 10  # steady after: from t + 1.0 s
DIP_BLOCKS = 10  # the first second after the change
LEAK_BLOCKS = 3  # the first 300 ms after the change
HOLD = 3  # blocks the recovered level must hold


def speech_center_block(ref, frame=FRAME, floor_db=40.0):
    """The block at the centre of the reference's measurable speech span."""
    _, _, loud = common.coherent_blocks(ref, ref, frame, floor_db)
    idx = loud.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return len(ref) // frame // 2
    return int((int(idx[0]) + int(idx[-1])) // 2)


def build_stimulus(ref, bed, t_block, kind, frame=FRAME, ramp=960):
    """Returns ``ref`` plus ``bed`` gated on or off at block ``t_block``.

    On for ``onset``, off for ``offset``, over a ``ramp``-sample linear ramp
    that ends at the block boundary, so every block from ``t_block`` on holds
    the new state entirely.
    """
    n = min(len(ref), len(bed))
    t = t_block * frame
    g = torch.zeros(n)
    lo, hi = max(0, t - ramp), min(n, t)
    g[hi:] = 1.0
    g[lo:hi] = torch.linspace(0.0, 1.0, hi - lo)
    if kind == "offset":
        g = 1.0 - g
    elif kind != "onset":
        raise ValueError(f"kind must be onset or offset, not {kind!r}")
    return ref[:n] + bed[:n] * g


def recovery_blocks(kept, loud, t_block, steady_after, tolerance_db, hold=HOLD):
    """Returns the blocks after ``t_block`` until the kept voice recovers.

    Recovered means within ``tolerance_db`` of ``steady_after`` for ``hold``
    consecutive loud blocks; NaN when it never is.
    """
    # a quiet block neither helps nor hurts
    ok = (kept >= steady_after - tolerance_db) | ~loud
    run = 0
    for b in range(t_block, len(kept)):
        if bool(ok[b]):
            run += int(bool(loud[b]))
            if run >= hold:
                first = b
                while first > t_block and bool(ok[first - 1]):
                    first -= 1
                # the recovery point is a speech block
                while not bool(loud[first]):
                    first += 1
                return first - t_block
        else:
            run = 0
    return float("nan")


def _median(t):
    return float(t.median()) if len(t) else float("nan")


def read_response(y, y_old, y_new, ref, t_block, tolerance_db, frame=FRAME):
    """Returns the per-stimulus numbers, block by block over the loud blocks.

    The transition output ``y`` against the steady outputs of the old state
    (``y_old``, what the input was before the change) and of the new state
    (``y_new``), over the reference's loud blocks.
    """
    kept, resid, loud = common.coherent_blocks(y, ref, frame)
    k_old, _, _ = common.coherent_blocks(y_old, ref, frame)
    k_new, r_new, _ = common.coherent_blocks(y_new, ref, frame)
    nb = min(len(kept), len(k_old), len(k_new))
    kept, resid, loud = kept[:nb], resid[:nb], loud[:nb]
    d_after = kept - k_new[:nb]  # the reaction to the new state
    # before the change: the same input as the old steady state
    d_before = kept - k_old[:nb]
    r_after = resid - r_new[:nb]
    t = min(t_block, nb)
    dip_win = d_after[t:t + DIP_BLOCKS][loud[t:t + DIP_BLOCKS]]
    pre_win = d_before[max(0, t - DIP_BLOCKS):t][loud[max(0, t - DIP_BLOCKS):t]]
    leak_win = r_after[t:t + LEAK_BLOCKS][loud[t:t + LEAK_BLOCKS]]
    steady = d_after[t + AFTER_FROM:][loud[t + AFTER_FROM:]]
    rec = (recovery_blocks(d_after, loud, t, 0.0, tolerance_db)
           if nb > t else float("nan"))
    return {
        "t_s":
            t_block * frame / common.SR,
        "n_blocks":
            nb,
        "dip_db":
            float(dip_win.min()) if len(dip_win) else float("nan"),
        "pre_db":
            float(pre_win.min()) if len(pre_win) else float("nan"),
        "recovery_ms":
            (rec * frame / common.SR * 1000 if rec == rec else float("nan")),
        "leak_db":
            float(leak_win.mean()) if len(leak_win) else float("nan"),
        "steady_delta_db":
            _median(steady),
        "kept_new_state_db":
            _median(k_new[:nb][t + AFTER_FROM:][loud[t + AFTER_FROM:]]),
        "resid_new_state_db":
            _median(r_new[:nb][t + AFTER_FROM:][loud[t + AFTER_FROM:]]),
    }


def measure_item(run,
                 mix,
                 ref,
                 tolerance_db=3.0,
                 ramp=960,
                 t_block=None,
                 frame=FRAME):
    """Returns both stimuli of one paired item through ``run(x)``.

    Next to the two steady passes (``run(mix)``, ``run(ref)``): {"onset": {...},
    "offset": {...}}.
    """
    n = min(len(mix), len(ref))
    mix, ref = mix[:n], ref[:n]
    bed = mix - ref
    t = speech_center_block(ref, frame) if t_block is None else t_block
    y_mix = torch.as_tensor(run(mix)).float()[:n]
    y_ref = torch.as_tensor(run(ref)).float()[:n]
    out = {}
    for kind, y_old, y_new in (
        ("onset", y_ref, y_mix),
        ("offset", y_mix, y_ref),
    ):
        stim = build_stimulus(ref, bed, t, kind, frame, ramp)
        y = torch.as_tensor(run(stim)).float()[:n]
        out[kind] = read_response(y, y_old, y_new, ref, t, tolerance_db, frame)
    return out


RESPONSE_KEYS = (
    "dip_db",
    "pre_db",
    "recovery_ms",
    "leak_db",
    "steady_delta_db",
    "kept_new_state_db",
    "resid_new_state_db",
)


def aggregate(items):
    """Returns per-stimulus means, the worst dip and the longest recovery.

    Means are NaN-aware; the count of items that never recovered is beside them.
    """
    out = {"n_items": len(items)}
    for kind in ("onset", "offset"):
        rs = [d[kind] for d in items if kind in d]
        s = {}
        for k in RESPONSE_KEYS:
            vals = [r[k] for r in rs if r[k] == r[k]]
            s[k] = float(np.mean(vals)) if vals else float("nan")
        dips = [r["dip_db"] for r in rs if r["dip_db"] == r["dip_db"]]
        recs = [
            r["recovery_ms"] for r in rs if r["recovery_ms"] == r["recovery_ms"]
        ]
        s["dip_worst_db"] = min(dips) if dips else float("nan")
        s["recovery_max_ms"] = max(recs) if recs else float("nan")
        s["n_never_recovered"] = sum(
            1 for r in rs if r["recovery_ms"] != r["recovery_ms"])
        out[kind] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    common.add_model_args(ap)
    ap.set_defaults(model="voice-isolator-v1")
    ap.add_argument("--conds",
                    default="ambience@+0,babble@+0,stationary@+0,traffic@+0")
    ap.add_argument("--limit",
                    type=int,
                    default=16,
                    help="items per condition (0: all)")
    ap.add_argument("--ramp-ms", type=float, default=20.0)
    ap.add_argument("--tolerance-db", type=float, default=3.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = common.manifest()
    vi, run = common.isolator_runner(a)
    ramp = int(a.ramp_ms / 1000 * common.SR)
    report = {
        "check": "transitions",
        "model": common.model_tag(a.model),
        "precision": a.precision,
        "device": str(vi.device),
        "mode": vi.mode,
        "lookahead_frames": vi.lookahead_frames,
        "set_id": common.set_id(),
        "frame": FRAME,
        "ramp_ms": a.ramp_ms,
        "tolerance_db": a.tolerance_db,
        "conditions": {},
    }
    for cond in [c for c in a.conds.split(",") if c]:
        items = []
        for r in common.pick_rows(rows, cond, a.limit, paired=True):
            mix = common.load48(r["mix_path"])
            ref = common.load48(r["clean_path"])
            d = measure_item(run, mix, ref, a.tolerance_db, ramp)
            d["id"] = r["id"]
            items.append(d)
        s = aggregate(items)
        report["conditions"][cond] = {"summary": s, "items": items}
        for kind in ("onset", "offset"):
            k = s[kind]
            print(
                f"{cond:>16s} {kind:6s}: n {s['n_items']:3d}  dip"
                f" {k['dip_db']:6.1f} dB (worst {k['dip_worst_db']:6.1f})  pre"
                f" {k['pre_db']:5.1f}  recovery {k['recovery_ms']:5.0f} ms (max"
                f" {k['recovery_max_ms']:4.0f}, never {k['n_never_recovered']})"
                f"  leak {k['leak_db']:5.1f} dB  steady delta"
                f" {k['steady_delta_db']:5.1f}",
                flush=True,
            )
    out = common.ops_path(a.model, "transitions", a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
