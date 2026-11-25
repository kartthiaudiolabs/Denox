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
"""Residual noise in the pauses of real recordings and under the voice.

Two rows the board reads from this sidecar, both bounded per item: no single
item contributes more than ``ITEM_CLIP_DB`` of credit or of penalty to a
pooled mean, so a few silenced pauses cannot buy the whole axis.

``pause_resid_db`` is what a system leaves in the pauses of a real recording
over what the recording held there, in dB (lower is cleaner), beyond what it
did to the voice: the item's own gain, read on the loudest fifth of its speech
frames, is undone, so an item merely turned down earns nothing and a voice
turned down with its pauses earns nothing for the part the voice lost. The
system's delay and polarity are undone first. The pauses are the input's:
outside every detected speech segment widened by 100 ms before an onset and
200 ms after an offset (breaths, onsets and tails belong to the voice), never
inside the item's first and last 300 ms (a word cut off at an edge is the
recording's, not the system's), and never where the input holds digital
silence. It is read at 48 kHz in one-third-octave bands weighted by the ITU-R
468 curve: an audibility proxy only (the standard's quasi-peak detection is not
applied), under which hiss at 2-10 kHz counts more than hum of the same energy.
An item needs 0.4 s of pause. ``pause_level_db`` reads the same residual
against the output's own voice, capped where it is taken as inaudible, so it
says whether what is left is audible at all; the per-band energies stay
unweighted in the records for the data-cleaning reading of the floor, whose
worst band is taken over the bands the voice occupies.

``babble_resid_db`` is, at each crowd-chatter level, the energy of the output
that no scaling of the clean voice explains, at the voice's own scale, over the
chatter the mixture held, in dB: the mixture's SDR minus the output's SI-SDR.
It is interference and distortion together, over the whole item, not noise
under speech alone. ``babble_supp_db`` reads |g o - r|^2 with one gain per
system, so a voice 2 dB quieter on one item reads as chatter left; this row
reads |o - a r|^2 / a^2 at the least-squares scale a = <o, r> / |r|^2 (the
mixture's SDR minus the output's SI-SDR), so a quieter output earns nothing and
loses nothing. A deleted or distorted voice counts, as it does for SI-SDR; an
output holding none of the voice reads the bound, the worst value, never the
floor. It comes from the base records' energies, so no audio is read; the base
records must be this set's, this render's and the base scorer's.

Usage: scoreresidual.py --systems a,b [--reuse-from DIR] [--aggregate-only DIR]
Output: <BENCH_RESULTS>/<system>.residual.json and .residual.items.jsonl
"""

import argparse
import collections
import glob
import json
import math
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

SCORER_SHA256 = observations.implementation(
    __file__,
    common.__file__,
    observations.__file__,
    levels.__file__,
    scorebench.__file__,
)

N_FFT, HOP = 2048, 480
PRE_S, POST_S = 0.100, 0.200
EDGE_S = 0.300
MIN_PAUSE_S = 0.4
SILENCE_DBFS = -80.0
# the loudest fifth of the speech frames give the voice's level
TOP_SHARE = 0.2
# a residual this far under the voice is taken as inaudible
LEVEL_CLIP_DB = -45.0
# no item reads beyond this much credit or penalty in any pooled row
ITEM_CLIP_DB = 20.0
# a band the voice occupies holds at least this share of its loudest band
OCCUPIED = 1e-3
# one-third-octave bands, 50 Hz to 20 kHz
CENTRES_HZ = tuple(1000.0 * 2.0**(k / 3.0) for k in range(-13, 14))
# the bands a 16 kHz-native system can fill
NB_BANDS = sum(1 for f in CENTRES_HZ if f * 2.0**(1 / 6) <= 8000.0)
REAL = ("real_dirty", "real_everyday", "real_multi", "real_noisy", "real_wild")
BABBLE = ("babble@+4", "babble@+0", "babble@-5")


def weighting(hz):
    """Returns the ITU-R 468 amplitude response at ``hz``, one at 1 kHz."""

    def response(f):
        h1 = (-4.737338981378384e-24 * f**6 + 2.043828333606125e-15 * f**4 -
              1.363894795463638e-7 * f**2 + 1.0)
        h2 = (1.306612257412824e-19 * f**5 - 2.118150887518656e-11 * f**3 +
              5.559488023498642e-4 * f)
        return f / math.sqrt(h1 * h1 + h2 * h2)

    return response(float(hz)) / response(1000.0)


WEIGHTS = np.array([weighting(f)**2 for f in CENTRES_HZ])


def band_index():
    """Returns each STFT bin's band, -1 outside the bands."""
    freqs = np.arange(N_FFT // 2 + 1) * common.SR / N_FFT
    index = np.full(len(freqs), -1, dtype=np.int64)
    for b, centre in enumerate(CENTRES_HZ):
        low, high = centre * 2.0**(-1 / 6), centre * 2.0**(1 / 6)
        index[(freqs >= low) & (freqs < high)] = b
    return index


BAND_OF_BIN = band_index()
# the bands at least one STFT bin falls in: the lowest holds none
SUPPORTED = np.array(
    [bool((BAND_OF_BIN == b).any()) for b in range(len(CENTRES_HZ))])


def spectrum(x):
    """Returns the power spectrogram of ``x`` as [frames, bins]."""
    window = torch.hann_window(N_FFT, dtype=x.dtype)
    z = torch.stft(x,
                   N_FFT,
                   HOP,
                   window=window,
                   center=False,
                   return_complex=True)
    return z.abs().pow(2).T.numpy()


def band_energies(power, frames):
    """Returns the energy per band over the given ``frames`` of ``power``."""
    total = power[frames].sum(0) if len(frames) else np.zeros(power.shape[1])
    out = np.zeros(len(CENTRES_HZ))
    for b in range(len(CENTRES_HZ)):
        out[b] = float(total[BAND_OF_BIN == b].sum())
    return out


def frame_masks(n, segs):
    """Returns the speech, widened-speech and edge masks over the frames.

    ``segs`` are the input's segments in 16 kHz samples; frames are those of
    ``spectrum`` over ``n`` samples at 48 kHz.
    """
    frames = (n - N_FFT) // HOP + 1
    centre = (np.arange(frames) * HOP + N_FFT / 2) / common.SR
    # a pause frame lies in the pause whole: one that straddles the voice's
    # boundary would read the voice as residual
    half = N_FFT / 2 / common.SR
    speech = np.zeros(frames, dtype=bool)
    wide = np.zeros(frames, dtype=bool)
    for s in segs:
        start, end = s["start"] / 16000.0, s["end"] / 16000.0
        speech |= (centre >= start) & (centre < end)
        wide |= (centre >= start - PRE_S - half) & (centre
                                                    < end + POST_S + half)
    edge = (centre < EDGE_S + half) | (centre > n / common.SR - EDGE_S - half)
    return speech, wide, edge


def score_real(it, out, mix, vad):
    """Returns one real item's pause and voice energies per band.

    The output's delay and polarity against the input are undone first. The
    energies are raw: the item's gain is undone when they are read.
    """
    n = min(len(out), len(mix))
    lag, sign = common.best_alignment(out[:n], mix[:n])
    o = sign * common.apply_lag(out, lag, n)
    x = mix[:n]
    _, segs = (scorebench.speech_segments(vad, x, it["mix_path"])
               if n == len(mix) else vad.segments(x))
    if n < N_FFT:
        return {"pause_seconds": 0.0, "speech_seconds": 0.0}
    power_o, power_x = spectrum(o), spectrum(x)
    speech, wide, edge = frame_masks(n, segs)
    frames = x.unfold(0, N_FFT, HOP).pow(2).mean(1).numpy()
    silent = 10 * np.log10(frames + 1e-20) < SILENCE_DBFS
    pause = ~wide & ~edge & ~silent
    speech &= ~edge
    loud = np.zeros(len(frames), dtype=bool)
    if speech.any():
        cut = np.quantile(frames[speech], 1.0 - TOP_SHARE)
        loud = speech & (frames >= cut)
    return {
        "lag": lag,
        "polarity": sign,
        "pause_seconds": float(pause.sum() * HOP / common.SR),
        "speech_seconds": float(speech.sum() * HOP / common.SR),
        "pause_frames": int(pause.sum()),
        "loud_frames": int(loud.sum()),
        "pause_out": band_energies(power_o, np.flatnonzero(pause)).tolist(),
        "pause_in": band_energies(power_x, np.flatnonzero(pause)).tolist(),
        "speech_in": band_energies(power_x, np.flatnonzero(loud)).tolist(),
        "speech_out": band_energies(power_o, np.flatnonzero(loud)).tolist(),
    }


def bounded(db, low=-ITEM_CLIP_DB, high=ITEM_CLIP_DB):
    """Returns ``db`` held inside one item's bounds."""
    return float(min(max(db, low), high))


def pause_metrics(m):
    """Returns one item's pause rows from its measurements.

    Nothing for an item with too little pause, or whose records lack the
    output's voice energy (an earlier scorer's).
    """
    if (m.get("pause_seconds", 0.0) < MIN_PAUSE_S or not m.get("pause_out") or
            not m.get("loud_frames") or not m.get("speech_out")):
        return {}
    # energies per frame: the pauses and the voice's loudest frames are
    # different numbers of frames
    out = np.array(m["pause_out"]) / m["pause_frames"]
    inp = np.array(m["pause_in"]) / m["pause_frames"]
    voice_in = np.array(m["speech_in"]) / m["loud_frames"]
    voice_out = np.array(m["speech_out"]) / m["loud_frames"]

    def ratio(a, b, w):
        return 10 * math.log10((float((w * a).sum()) + 1e-20) / (float(
            (w * b).sum()) + 1e-20))

    nb = np.array([1.0 if b < NB_BANDS else 0.0 for b in range(len(WEIGHTS))])
    # the item's own gain, from its voice: what the pause lost beyond what
    # the voice lost is cleanup; what both lost is the item turned down
    gain = ratio(voice_out, voice_in, WEIGHTS)
    gain_nb = ratio(voice_out, voice_in, WEIGHTS * nb)
    occupied = SUPPORTED & (voice_in >= voice_in.max() * OCCUPIED) & (voice_out
                                                                      > 0)
    worst = ((out[occupied] + 1e-20) /
             (voice_out[occupied] + 1e-20)).max() if occupied.any() else None
    return {
        "pause_resid_db":
            bounded(ratio(out, inp, WEIGHTS) - gain),
        "pause_resid_db_nb":
            bounded(ratio(out, inp, WEIGHTS * nb) - gain_nb),
        # the level row is bounded per item like the reduction row: an item
        # earns or loses at most ITEM_CLIP_DB against its own input, then the
        # cap at which a residual under the voice is taken as inaudible
        "pause_level_db":
            bounded(
                bounded(ratio(out, voice_out, WEIGHTS),
                        ratio(inp, voice_in, WEIGHTS) - ITEM_CLIP_DB,
                        ratio(inp, voice_in, WEIGHTS) + ITEM_CLIP_DB),
                LEVEL_CLIP_DB),
        "pause_gain_db":
            gain,
        "floor_db":
            ratio(out, voice_out, np.ones(len(WEIGHTS))),
        "floor_worst_db":
            10 * math.log10(float(worst)) if worst is not None else math.nan,
    }


def babble_metrics(energies):
    """Returns one item's bounded voice-free residual in dB, from its energies.

    Args:
        energies: ``[|o|^2, <o, r>, |r|^2, |m - r|^2]``.
    """
    oo, orr, rr, mr = (float(v) for v in energies)
    if rr <= 0.0:
        # no voice in the reference: nothing to read against (never in the set)
        return 0.0
    # the voice's scale in the output; the residual is read at that scale, so
    # a quieter output earns nothing and loses nothing. An output holding none
    # of the voice, or its opposite, removed the voice: the worst reading
    scale = orr / rr
    if scale <= 0.0:
        return ITEM_CLIP_DB
    resid = max(oo - orr * orr / rr, 0.0) / (scale * scale)
    return bounded(10 * math.log10((resid + 1e-9) / (mr + 1e-9)))


def base_energies(path, ids, name, outputs):
    """Returns {item id: (identity, energies, nb energies)} of base records.

    The records must be this system's, this set's, the whole set's, the base
    scorer's and of the render under evaluation: the chatter rows derive from
    them and are stamped with this run's identity.

    Raises:
        SystemExit: The base records are missing or of another provenance.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: score the system with scorebench"
                         " first")
    problem = scorebench.view_source(path, name, scorebench.SCORER_SHA256)
    if problem:
        raise SystemExit(f"{path}: {problem}")
    metadata, records = observations.read(path)
    # the base records carry the render's stamp in their header
    stamped = (metadata.get("render_identity") or {}).get("outputs_sha256")
    if stamped != outputs:
        raise SystemExit(f"{path}: records of another render of {name}:"
                         " re-score it with scorebench first")
    out = {}
    for r in records:
        if r["id"] not in ids or r.get("output_status") != "ok":
            continue
        m = r.get("measurements") or {}
        if m.get("babble_energy") and m.get("babble_energy_nb"):
            out[r["id"]] = (r["identity"], m["babble_energy"][0],
                            m["babble_energy_nb"][0])
    return out


def assemble(name, rows, vad, reuse=None, hashes=None, base=None):
    """Returns per-item records over ``rows``.

    Real items are read from audio; crowd-chatter items take their energies
    from the base records (``base``). An item whose output samples are
    unchanged since an earlier pass keeps its record (``reuse``).
    """
    hashes = hashes or common.file_hashes()
    reuse, base = reuse or {}, base or {}
    records, counts = [], collections.Counter()
    for it in rows:
        record = observations.item(it)
        records.append(record)
        if it.get("clean_path"):
            held = base.get(it["id"])
            if held is None:
                record["output_status"] = "missing"
                continue
            identity, energies, energies_nb = held
            record["identity"] = identity
            record["measurements"] = {
                "babble_energy": [list(energies)],
                "babble_energy_nb": [list(energies_nb)],
            }
            record["metrics"] = {
                "babble_resid_db": babble_metrics(energies),
                "babble_resid_db_nb": babble_metrics(energies_nb),
            }
            continue
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
        old = reuse.get(scorebench.reuse_key(record["identity"]))
        if old is not None:
            record["measurements"] = dict(old["measurements"])
            counts["reused"] += 1
        else:
            record["measurements"] = observations.finite(
                score_real(it, out, mix, vad))
            counts["scored"] += 1
    if reuse:
        print(
            f"  reused {counts['reused']} records, scored {counts['scored']}"
            " items",
            flush=True,
        )
    return records


def aggregate(records, rows, level_cal_db):
    """Returns the sidecar file for ``rows``.

    Per real family the mean of its items' rows, and pooled over every real
    item (``_real_cleanup``); per crowd-chatter level the mean residual. Every
    condition keeps its expected and missing counts whether or not it has a
    value; an item without enough pause is counted as short and one whose
    pauses hold no speech to read the voice from as no_speech, never as
    missing. Pure, so any view of the set derives from the whole-set records.
    The rows undo each item's own gain, so the system's calibration is recorded
    and not applied.
    """
    by_id = {r["id"]: r for r in records}
    out = {}
    pooled = collections.defaultdict(list)
    short, no_speech = collections.Counter(), collections.Counter()
    for it in rows:
        cond = it["condition"]
        rec = by_id.get(it["id"])
        cell = out.setdefault(cond, {"n_expected": 0, "n_missing": 0, "n": 0})
        cell["n_expected"] += 1
        if rec is None or rec.get("output_status") != "ok":
            cell["n_missing"] += 1
            continue
        m = observations.revive(rec.get("measurements", {}))
        if it.get("clean_path"):
            values = {
                "babble_resid_db": babble_metrics(m["babble_energy"][0]),
                "babble_resid_db_nb": babble_metrics(m["babble_energy_nb"][0]),
            }
        else:
            values = pause_metrics(m)
            rec["metrics"] = observations.finite(values)
            if not values:
                # too little pause, or no speech left to read the voice from
                if (m.get("pause_seconds", 0.0) >= MIN_PAUSE_S and
                        not m.get("loud_frames")):
                    no_speech[cond] += 1
                else:
                    short[cond] += 1
                continue
            for k, v in values.items():
                if math.isfinite(v):
                    pooled[k].append(v)
        cell["n"] += 1
        for k, v in values.items():
            if math.isfinite(v):
                cell.setdefault("_values", {}).setdefault(k, []).append(v)
    for cond, cell in out.items():
        for k, vs in cell.pop("_values", {}).items():
            cell[k] = float(np.mean(vs))
        if short[cond]:
            cell["n_short"] = short[cond]
        if no_speech[cond]:
            cell["n_no_speech"] = no_speech[cond]
    real = [c for c in REAL if c in out]
    if real:
        cleanup = {k: float(np.mean(vs)) for k, vs in pooled.items()}
        cleanup.update({
            "n": sum(out[c]["n"] for c in real),
            "n_expected": sum(out[c]["n_expected"] for c in real),
            "n_missing": sum(out[c]["n_missing"] for c in real),
            "n_short": sum(short[c] for c in real),
            "n_no_speech": sum(no_speech[c] for c in real),
            "by_family": {
                c: out[c]["pause_resid_db"]
                for c in real
                if "pause_resid_db" in out[c]
            },
        })
        out["_real_cleanup"] = cleanup
    out["_level_cal_db"] = level_cal_db
    return out


def level_calibration(results, name, outputs):
    """Returns the system's output gain from its base scores.

    The base scores must be the base scorer's and of the render under
    evaluation.

    Raises:
        SystemExit: The base scores are missing or of another provenance.
    """
    path = os.path.join(results, f"{name}.metrics.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: score {name} with scorebench"
                         " before this sidecar")
    with open(path, encoding="utf-8") as fh:
        base = json.load(fh)
    if base.get("_scorer_sha256") != scorebench.SCORER_SHA256:
        raise SystemExit(f"{path}: scored by another base scorer; re-score"
                         f" {name} with scorebench first")
    if base.get("_outputs_sha256") != outputs:
        raise SystemExit(f"{path}: scores of another render of {name};"
                         " re-score it with scorebench first")
    return float(base.get("_level_cal_db") or 0.0)


def write(name, records, out, rows, level_cal_db, outputs):
    observations.sidecar(
        os.path.join(common.RESULTS, f"{name}.residual.items.jsonl"),
        records,
        name,
        common.set_id(),
        common.SUBSET,
        common.is_sealed(rows),
        SCORER_SHA256,
        level_cal_db=level_cal_db,
        outputs_sha256=outputs,
    )
    out["_set_id"] = common.set_id()
    out["_subset"] = common.SUBSET
    out.update(anchors.stamp())
    out["_sealed"] = common.is_sealed(rows)
    out["_scorer_sha256"] = SCORER_SHA256
    out["_outputs_sha256"] = outputs
    path = os.path.join(common.RESULTS, f"{name}.residual.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(observations.finite(out), fh, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="")
    ap.add_argument("--reuse-from", default="", metavar="DIR")
    ap.add_argument("--aggregate-only", default="", metavar="DIR")
    a = ap.parse_args()

    rows = common.manifest()
    common.guard_private_results(rows)
    selected = [r for r in rows if r["condition"] in REAL + BABBLE]
    systems = [s for s in a.systems.split(",") if s] or sorted(
        os.path.basename(d)
        for d in glob.glob(common.OUT_ROOT + "/*")
        if os.path.isdir(d))
    vad = None if a.aggregate_only else common.Vad()
    os.makedirs(common.RESULTS, exist_ok=True)
    for name in systems:
        if a.aggregate_only:
            src = os.path.join(a.aggregate_only, f"{name}.residual.items.jsonl")
            ids = {r["id"] for r in selected}
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
            outputs = common.outputs_digest(name)
            own = os.path.join(common.RESULTS, f"{name}.residual.items.jsonl")
            reuse, why = {}, "no earlier records"
            if a.reuse_from:
                reuse, why = scorebench.reuse_index(
                    os.path.join(a.reuse_from, f"{name}.residual.items.jsonl"),
                    name,
                    SCORER_SHA256,
                )
                why = f"reuse from {a.reuse_from}: {why}"
            elif os.path.isfile(own):
                reuse, why = scorebench.reuse_index(own, name, SCORER_SHA256)
                why = f"existing records: {why}"
            print(f"  {why}", flush=True)
            base = base_energies(
                os.path.join(common.RESULTS, f"{name}.items.jsonl"),
                {r["id"] for r in selected if r.get("clean_path")},
                name,
                outputs,
            )
            records = assemble(name, selected, vad, reuse=reuse, base=base)
        level_cal_db = level_calibration(common.RESULTS, name, outputs)
        out = aggregate(records, selected, level_cal_db)
        write(name, records, out, rows, level_cal_db, outputs)
        pooled = out.get("_real_cleanup", {})
        babble = out.get("babble@+4", {}).get("babble_resid_db", math.nan)
        print(
            f"{name:24s} pauses {pooled.get('pause_resid_db', math.nan):6.1f}"
            f" dB over {pooled.get('n', 0)} real items; babble@+4 residual"
            f" {babble:6.1f} dB",
            flush=True,
        )


if __name__ == "__main__":
    main()
