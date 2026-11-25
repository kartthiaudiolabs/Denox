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
"""Runtime of the public processing path on real audio, cold and warm.

The cold start, the warm p50 and p95, the memory and the failures::

    speedtest.py --model voice-isolator-v1 [--durations 3,10,45]
        [--per-duration 3] [--repeat 5] [--device cuda] [--precision fp32]
        [--out staging/ops/<tag>.speed.json]
    speedtest.py --model voice-isolator-v1 --synthetic 30   # one noise clip

Clips come from the eval set (VOICE_ISOLATION_EVALS): for every requested
duration, the ``--per-duration`` items whose length is nearest, so the numbers
describe real speech through the whole public path (hint frontend, resampling,
level conditioning, chunking). The first call after loading is the cold start
(CUDA context, kernel selection, the frontend's first pass) and is reported on
its own; the warm figures are p50 / p95 / max over ``--repeat`` passes of every
clip, with the realtime factor from the p50. Peak GPU memory is read after the
warm passes. A pass that raises, returns NaN or the wrong length is a failure,
counted and never averaged in. Prints the README's "Tested on" line and writes
the full report as JSON.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import soundfile as sf
import torch

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def summarize(times):
    """p50 / p95 / max / mean of a list of seconds (NaN when empty)."""
    if not times:
        return {
            "n": 0,
            "p50": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    t = np.array(times, dtype=float)
    return {
        "n": len(t),
        "p50": float(np.percentile(t, 50)),
        "p95": float(np.percentile(t, 95)),
        "max": float(t.max()),
        "mean": float(t.mean()),
    }


def check_output(y, n):
    """Returns the failure reason of an output, or None when it is valid.

    Valid is a finite signal of ``n`` samples.
    """
    try:
        y = torch.as_tensor(y)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # whatever cannot become a tensor is not a signal
        return f"not a signal: {e}"
    if y.ndim != 1 or len(y) != n:
        return f"length {tuple(y.shape)} != ({n},)"
    if not bool(torch.isfinite(y.float()).all()):
        return "nan"
    return None


def timed(run, x, clock, sync):
    """(seconds, failure reason or None) of one pass."""
    t0 = clock()
    try:
        y = run(x)
        if sync:
            sync()
    except Exception as e:  # pylint: disable=broad-exception-caught
        # the failure is the measurement
        return clock() - t0, f"raised: {e}"
    return clock() - t0, check_output(y, len(x))


def measure(run, clips, repeat=5, clock=time.perf_counter, sync=None):
    """Cold start on the first clip, then ``repeat`` warm passes of every clip.

    ``clips`` is a list of (group, label, x48). Returns the cold pass and, per
    group, the warm timings, their summary, the realtime factor at the p50 and
    the failures.
    """
    if not clips:
        raise ValueError("no clips to time")
    group0, label0, x0 = clips[0]
    cold_s, cold_err = timed(run, x0, clock, sync)
    report = {
        "cold": {
            "group": group0,
            "clip": label0,
            "seconds": cold_s,
            "failure": cold_err,
        },
        "groups": {},
    }
    for group, label, x in clips:
        g = report["groups"].setdefault(
            group,
            {
                "clip_seconds": [],
                "clips": [],
                "times": [],
                "failures": []
            },
        )
        g["clips"].append(label)
        g["clip_seconds"].append(len(x) / common.SR)
        for _ in range(repeat):
            s, err = timed(run, x, clock, sync)
            if err:
                g["failures"].append({
                    "clip": label,
                    "reason": err,
                    "seconds": s
                })
            else:
                g["times"].append(s)
    for g in report["groups"].values():
        g["summary"] = summarize(g["times"])
        secs = float(np.mean(g["clip_seconds"]))
        g["mean_clip_seconds"] = secs
        g["realtime_p50"] = (secs / g["summary"]["p50"]
                             if g["times"] else float("nan"))
        g["n_failed"] = len(g["failures"])
    return report


def clip_seconds(rows):
    """Returns [(row, seconds)] from the wav headers, with no decoding.

    The manifest rows carry resolved paths.
    """
    return [(r, sf.info(r["mix_path"]).duration) for r in rows]


def pick_clips(rows_with_seconds, durations, per_duration):
    """Returns, for every requested duration, the rows nearest in length.

    ``per_duration`` rows per duration, no row serving two durations: [(group
    label, row, seconds)], one group per duration.
    """
    out, used = [], set()
    for d in durations:
        near = sorted(rows_with_seconds,
                      key=lambda rs, d=d: (abs(rs[1] - d), rs[0]["id"]))
        for r, s in [rs for rs in near if rs[0]["id"] not in used
                    ][:per_duration]:
            out.append((f"{d:g}s", r, s))
            used.add(r["id"])
    return out


def readme_line(report, memory):
    """Returns the README's "Tested on" cell.

    Every group at its p50, then the memory.
    """
    parts = []
    for name, g in report["groups"].items():
        p50 = g["summary"]["p50"]
        if p50 != p50:
            parts.append(f"{name} clip: every pass failed")
            continue
        parts.append(f"{g['mean_clip_seconds']:.0f} s clip in {p50:.1f} s "
                     f"({g['realtime_p50']:.1f}× realtime)")
    return "; ".join(parts) + f"; {memory}"


def main():
    ap = argparse.ArgumentParser()
    common.add_model_args(ap)
    ap.set_defaults(model="voice-isolator-v1")
    ap.add_argument("--durations",
                    default="3,10,45",
                    help="seconds, comma-separated")
    ap.add_argument("--per-duration", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--synthetic",
        type=float,
        default=0.0,
        help="time one S-second noise clip",
    )
    ap.add_argument("--chunk-seconds", type=float, default=None)
    ap.add_argument("--overlap-seconds", type=float, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    vi, run_ = common.isolator_runner(a)

    def run(x):
        return run_(x, a.chunk_seconds, a.overlap_seconds)

    dev = vi.device
    if a.synthetic:
        g = torch.Generator().manual_seed(0)
        clips = [(
            f"{a.synthetic:g}s",
            "synthetic noise",
            torch.randn(int(a.synthetic * common.SR), generator=g) * 0.3,
        )]
    else:
        rows = common.pick_rows(common.manifest(), "", 0)
        durations = [float(v) for v in a.durations.split(",") if v]
        picked = pick_clips(clip_seconds(rows), durations, a.per_duration)
        clips = [
            (grp, r["id"], common.load48(r["mix_path"])) for grp, r, _ in picked
        ]
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    report = measure(run, clips, a.repeat, sync=sync)
    memory = (
        f"{torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB peak GPU memory"
        if dev.type == "cuda" else "CPU")
    name = (torch.cuda.get_device_name(dev)
            if dev.type == "cuda" else platform.processor() or "cpu")
    report.update({
        "check": "speed",
        "model": common.model_tag(a.model),
        "device": name,
        "device_type": dev.type,
        "precision": a.precision,
        "mode": vi.mode,
        "lookahead_frames": vi.lookahead_frames,
        "chunk_seconds": (a.chunk_seconds if a.chunk_seconds is not None else
                          vi.model.default_window()[0]),
        "overlap_seconds": (a.overlap_seconds if a.overlap_seconds is not None
                            else vi.model.default_window()[1]),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "synthetic": bool(a.synthetic),
        "set_id": None if a.synthetic else common.set_id(),
        "peak_memory": memory,
        "readme_line": readme_line(report, memory),
    })
    print(f"device: {name}  precision {a.precision}  torch {torch.__version__} "
          f" cuda {torch.version.cuda}  python {platform.python_version()}")
    c = report["cold"]
    print(f"cold start: {c['seconds']:.2f} s on {c['clip']}" +
          (f"  FAILED {c['failure']}" if c["failure"] else ""))
    for grp, g in report["groups"].items():
        s = g["summary"]
        print(
            f"{grp:>5s} ({g['mean_clip_seconds']:5.1f} s, {len(g['clips'])}"
            f" clips x {a.repeat}): p50 {s['p50']:6.2f} s  p95 {s['p95']:6.2f} "
            f" max {s['max']:6.2f}  {g['realtime_p50']:5.1f}x realtime  failed"
            f" {g['n_failed']}")
    print(report["readme_line"])
    out = common.ops_path(a.model, "speed", a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
