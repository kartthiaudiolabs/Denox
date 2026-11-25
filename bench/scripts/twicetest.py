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
"""A second pass over a processed file: its cost, idempotence, determinism.

    twicetest.py --model voice-isolator-v1
        [--conds babble@+4,ambience@+0,stationary@+0,wild@+0] [--limit 16]
        [--no-speaker] [--out staging/ops/<tag>.twice.json]

Files reach a denoiser that were denoised already (a re-export, a second tool in
the chain), and nothing in the product stops a user from running one twice. Per
paired item: y1 = run(mix), y2 = run(y1), then y1' = run(mix) again after that
other work. Reports the voice kept (coherent projection on the reference, mean
and 10th percentile of 100 ms blocks) and the residual after one and after two
passes and what the second pass cost; idempotence, 10 log10 |y2 - y1|^2 / |y1|^2
(0 dB: the second pass rewrote everything, -60: it changed nothing);
determinism, |y1' - y1| the same way (a stateless model reads the floor); and
speaker similarity after each pass. A pass that raises or returns NaN or the
wrong length is a failure, counted and never averaged in.
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
DIFF_FLOOR_DB = -100.0


def diff_db(a, b):
    """10 log10 |a - b|^2 / |b|^2 over the common length, floored at -100 dB."""
    n = min(len(a), len(b))
    num = float((a[:n] - b[:n]).pow(2).mean())
    den = float(b[:n].pow(2).mean())
    return max(
        DIFF_FLOOR_DB,
        float(10 * np.log10(num / (den + 1e-14) + 10**(DIFF_FLOOR_DB / 10))),
    )


def _pass(run, x, n):
    """One pass, or the failure it raised / produced."""
    try:
        y = run(x)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # a failure is a result here, not a crash
        return None, f"raised: {e}"
    y = torch.as_tensor(y).float()
    if not bool(torch.isfinite(y).all()):
        return None, "nan"
    if len(y) != n:
        return None, f"length {len(y)} != {n}"
    return y, None


def _voice(y, ref):
    kept, resid, loud = common.coherent_blocks(y, ref, FRAME)
    k = kept[loud]
    return {
        "kept_coh_mean_db": float(k.mean()) if len(k) else float("nan"),
        "kept_coh_p10_db":
            (float(torch.quantile(k, 0.1)) if len(k) else float("nan")),
        "resid_db": float(resid[loud].mean()) if len(k) else float("nan"),
    }


def measure_item(run, mix, ref, spk=None, key=None):
    """Returns one paired item's numbers over three passes through ``run(x)``.

    A failed pass returns ``{"failure": reason}``. ``spk`` is a speaker
    embedder with ``sim(a, b, key_b=...)``.
    """
    n = len(mix)
    y1, err = _pass(run, mix, n)
    if err:
        return {"failure": f"pass 1 {err}"}
    y2, err = _pass(run, y1, n)
    if err:
        return {"failure": f"pass 2 {err}"}
    y1b, err = _pass(run, mix, n)
    if err:
        return {"failure": f"repeat {err}"}
    one, two = _voice(y1, ref), _voice(y2, ref)
    d = {f"pass1_{k}": v for k, v in one.items()}
    d.update({f"pass2_{k}": v for k, v in two.items()})
    d.update({
        "second_pass_kept_delta_db":
            (two["kept_coh_mean_db"] - one["kept_coh_mean_db"]),
        "second_pass_p10_delta_db":
            (two["kept_coh_p10_db"] - one["kept_coh_p10_db"]),
        "second_pass_resid_delta_db": two["resid_db"] - one["resid_db"],
        "idempotence_db": diff_db(y2, y1),
        "determinism_db": diff_db(y1b, y1),
    })
    if spk is not None:
        d["pass1_spk_sim"] = spk.sim(y1, ref, key_b=key)
        d["pass2_spk_sim"] = spk.sim(y2, ref, key_b=key)
        d["second_pass_spk_delta"] = d["pass2_spk_sim"] - d["pass1_spk_sim"]
    return d


def aggregate(items):
    """Means over the items that did not fail, the failure count and reasons."""
    ok = [d for d in items if "failure" not in d]
    out = {"n_items": len(items), "n_failed": len(items) - len(ok)}
    out["failures"] = [d["failure"] for d in items if "failure" in d]
    keys = sorted({k for d in ok for k in d if k != "id"})
    for k in keys:
        vals = [d[k] for d in ok if k in d and d[k] == d[k]]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    if ok:
        out["determinism_max_db"] = max(d["determinism_db"] for d in ok)
        out["idempotence_max_db"] = max(d["idempotence_db"] for d in ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    common.add_model_args(ap)
    ap.set_defaults(model="voice-isolator-v1")
    ap.add_argument("--conds",
                    default="babble@+4,ambience@+0,stationary@+0,wild@+0")
    ap.add_argument("--limit",
                    type=int,
                    default=16,
                    help="items per condition (0: all)")
    ap.add_argument(
        "--no-speaker",
        action="store_true",
        help="skip the WavLM speaker similarity",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = common.manifest()
    vi, run = common.isolator_runner(a)
    spk = None if a.no_speaker else common.SpeakerEmbedder()
    report = {
        "check": "twice",
        "model": common.model_tag(a.model),
        "precision": a.precision,
        "device": str(vi.device),
        "mode": vi.mode,
        "lookahead_frames": vi.lookahead_frames,
        "set_id": common.set_id(),
        "frame": FRAME,
        "conditions": {},
    }
    for cond in [c for c in a.conds.split(",") if c]:
        items = []
        for r in common.pick_rows(rows, cond, a.limit, paired=True):
            mix = common.load48(r["mix_path"])
            ref = common.load48(r["clean_path"])
            d = measure_item(run, mix, ref, spk, key=f"{r['id']}.ref")
            d["id"] = r["id"]
            items.append(d)
        s = aggregate(items)
        report["conditions"][cond] = {"summary": s, "items": items}
        print(
            f"{cond:>18s}: n {s['n_items']:3d} failed {s['n_failed']:d}  "
            f"kept p10 {s.get('pass1_kept_coh_p10_db', float('nan')):6.1f} -> "
            f"{s.get('pass2_kept_coh_p10_db', float('nan')):6.1f} dB  "
            f"idempotence {s.get('idempotence_db', float('nan')):6.1f} dB  "
            f"determinism {s.get('determinism_db', float('nan')):6.1f} dB",
            flush=True,
        )
    out = common.ops_path(a.model, "twice", a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
