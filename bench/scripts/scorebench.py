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
"""Score every system's outputs with the benchmark's own metric implementations.

Layout:  <BENCH_OUT>/<system>/<id>.wav
Output:  <BENCH_RESULTS>/<system>.metrics.json

FAIRNESS MACHINERY (all three are reported, never silently applied):

1. LEVEL CALIBRATION. ``level_cal_db`` = median over the ``clean``-condition
   items of 20*log10(rms(out)/rms(in)), bounded at ``common.MAX_LEVEL_CAL_DB``:
   one gain per system, scored once, as the level change on clean speech
   (``level_err_db``), and undone everywhere else (``levels.py``). The kept
   energy rows are stored raw and calibrated by the board (``raw -
   level_cal_db``); the rows a single subtraction cannot calibrate are computed
   here at the calibration, from per-item measurements: competing-voice
   suppression (``babble_supp_db``, from each item's energies), non-voiced
   suppression, the high-band error (``hf_err_db``, per-item absolute errors,
   averaged) and deleted speech. Their raw values stay beside them as
   ``*_raw_db``. si_sdr is scale-invariant by construction.

2. ALGORITHMIC DELAY. Each item is aligned to the reference by the best lag
   within +-100 ms (FFT cross-correlation) before reference-based scoring; the
   median lag is reported per condition.

3. BAND-LIMITED REFERENCE. Every reference metric is also computed with BOTH
   sides band-limited to 8 kHz (``*_nb``), so a 16 kHz-native model is not
   judged for HF it structurally cannot emit. ``band_kept_db`` per (0-0.5,
   0.5-2, 2-4, 4-8, 8-12, 12-16, 16-24) kHz is the direct measurement of
   full-band integrity.

Adding a metric to an already-scored field: write a SIDECAR (scorenisqa.py,
scorerealx.py, scorekept.py), never ``--force`` a full re-score.
"""

import argparse
import collections
import glob
import hashlib
import io
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
import stamprender

NISQA_ON = os.environ.get("SCORE_NISQA", "1") != "0"
# the non-voiced floor of a real recording is read on the 20 ms frames whose
# input is above this: digital silence has no floor to suppress
NONVOICED_FLOOR_DBFS = -80.0
# and clear of the detected speech by this many 20 ms frames on either side
NONVOICED_GUARD_FRAMES = 2
# the log's guard, far below any floor the row admits (-200 dB): a 1e-9 guard
# capped the output at about -90 dBFS and read 40 dB of suppression of a
# -78 dBFS floor as 12
NONVOICED_EPS = 1e-20
NISQA_N = int(os.environ.get("SCORE_NISQA_N", "12"))
SCORER_SHA256 = observations.implementation(__file__, common.__file__,
                                            observations.__file__,
                                            levels.__file__)


def speech_segments(vad, wav48, path):
    """Returns ``vad.segments`` of a file, cached under the set's ``.cache``.

    The cache is keyed by the audio's samples and the detector's identity: a
    mixture and its reference share a name but never an entry, and a new
    detector never reads an old one's segments. The references and mixtures are
    the same for every system scored, and the voice-activity pass is the
    scorer's largest cost. ``path`` names the file in error messages only.
    """
    del path  # the samples, not the name, identify the audio
    salt = f"{getattr(vad, 'identity', 'vad')}|".encode()
    samples = wav48.contiguous().numpy().tobytes()
    key = hashlib.sha256(salt + samples).hexdigest()
    cache_dir = os.path.join(common.EVAL, ".cache", "vad2")
    cached = os.path.join(cache_dir, key + ".json")
    w16 = torchaudio.functional.resample(wav48, common.SR, 16000)
    if os.path.exists(cached):
        with open(cached, encoding="utf-8") as fh:
            return w16, json.load(fh)
    w16, segs = vad.segments(wav48)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{cached}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump([{
            "start": int(s["start"]),
            "end": int(s["end"])
        } for s in segs], fh)
    os.replace(tmp, cached)
    return w16, segs


def rms_db(x):
    return float(20 * torch.log10(x.pow(2).mean().sqrt() + 1e-12))


def sample_hash(x):
    """Returns the identity of decoded audio.

    The 48 kHz float32 samples the scorer consumes.
    """
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def item_identity(r, out_hash, hashes):
    """Returns what one item's observation depends on, besides the scorer.

    The output samples, the mixture and reference files, and the item's scoring
    semantics.
    """
    return {
        "output": out_hash,
        "mix": hashes(r["mix_path"]),
        "ref": hashes(r["clean_path"]) if r.get("clean_path") else None,
        "semantics": {
            "condition": r["condition"],
            "score_from": int(r.get("score_from") or 0),
            "score_policy": r.get("score_policy"),
            "paired": bool(r.get("clean_path")),
        },
    }


def reuse_key(identity):
    return json.dumps(identity, sort_keys=True)


NISQA_KEYS = (
    "nisqa_mos",
    "nisqa_noisiness",
    "nisqa_discontinuity",
    "nisqa_coloration",
    "nisqa_loudness",
)


def suppression_energies(out, ref, mix):
    """Returns an item's mean squares |o|^2, <o, r>, |r|^2 and |m - r|^2.

    In float64, so suppression can be read at any output gain
    (``levels.suppression_db``).
    """
    o, r, m = out.double(), ref.double(), mix.double()
    return [
        float(o.pow(2).mean()),
        float((o * r).mean()),
        float(r.pow(2).mean()),
        float((m - r).pow(2).mean()),
    ]


def nisqa_measurements(out):
    values = {}
    for k, v in zip(NISQA_KEYS, common.nisqa(out)):
        if v == v:
            values[k] = [v]
    return values


def score_item(r, out, mix, spk, vad, dnsmos_ovr, want_nisqa):
    """Returns one item's raw measurements from decoded audio of equal length.

    The lists the condition reducers pool. The lost-block share is not among
    them: it depends on the system's level calibration, so the coherence blocks
    are kept and the share derived per view.
    """
    acc = collections.defaultdict(list)
    cond = r["condition"]
    real = cond.startswith("real") or not r.get("clean_path")
    acc["out_in_rms_db"].append(rms_db(out) - rms_db(mix))
    if want_nisqa:
        for k, v in nisqa_measurements(out).items():
            acc[k].extend(v)
    if not real:
        ref = common.load48(r["clean_path"])
        n = min(len(ref), len(mix))
        lag, sign = common.best_alignment(out, ref)
        acc["lag"].append(lag)
        acc["polarity"].append(sign)
        o = sign * common.apply_lag(out, lag, n)
        ref, mix_a = ref[:n], mix[:n]
        sfrom = int(r.get("score_from") or 0)
        if sfrom:
            o, ref, mix_a = o[sfrom:], ref[sfrom:], mix_a[sfrom:]
        # the reference is a wild recording: kept speech, speaker and bed
        # residual, never SI-SDR
        retention = r.get("score_policy") == "retention"
        if not retention:
            acc["si_sdr"].append(common.si_sdr(o, ref))
        acc["kept"].append(common.voice_kept_db(o, ref))
        # the voice that survived: the output's projection on the reference per
        # 100 ms block over the reference's speech blocks (a VAD mask, so quiet
        # speech counts)
        r16, segs = speech_segments(vad, ref, r["clean_path"])
        m16 = torch.zeros(len(r16), dtype=torch.bool)
        for sg in segs:
            m16[sg["start"]:sg["end"]] = True
        mask = common.block_mask(m16, len(ref) // 4800)
        coh = common.coherent_kept_db(o, ref, act=mask)
        if len(coh):
            acc["kept_coh_mean"].append(float(coh.mean()))
            acc["kept_coh_p10"].append(float(torch.quantile(coh, 0.1)))
            acc[observations.BLOCKS].append([float(v) for v in coh])
        acc["terc"].append(common.tercile_kept_db(o, ref))
        acc["band"].append(common.band_kept_db(o, ref))
        acc["spk_sim"].append(spk.sim(o, ref, key_b=f"{r['id']}.ref"))
        onb, rnb = common.nb8k(o), common.nb8k(ref)
        coh_nb = common.coherent_kept_db(onb, rnb, act=mask)
        if len(coh_nb):
            acc["kept_coh_nb_mean"].append(float(coh_nb.mean()))
            acc["kept_coh_nb_p10"].append(float(torch.quantile(coh_nb, 0.1)))
        if not retention:
            acc["si_sdr_nb"].append(common.si_sdr(onb, rnb))
        else:
            # what was removed where the wild recording holds no speech: bed and
            # the recording's own floor alike, against the input (lower is
            # better)
            o16 = torchaudio.functional.resample(o, common.SR, 16000)[:len(r16)]
            x16 = torchaudio.functional.resample(mix_a, common.SR, 16000)
            x16 = x16[:len(r16)]
            nv = ~m16[:min(len(o16), len(x16))]
            if nv.any():
                acc["nonvoiced_supp_db"].append(
                    float(10 * torch.log10(
                        (o16[:len(nv)][nv].pow(2).mean() + 1e-9) /
                        (x16[:len(nv)][nv].pow(2).mean() + 1e-9))))
        acc["kept_nb"].append(common.voice_kept_db(onb, rnb))
        acc["spk_sim_nb"].append(spk.sim(onb, rnb, key_b=f"{r['id']}.refnb"))
        if cond.startswith("babble"):
            acc["babble_energy"].append(suppression_energies(o, ref, mix_a))
            acc["babble_energy_nb"].append(
                suppression_energies(onb, rnb, common.nb8k(mix_a)))
    else:
        n = min(len(out), len(mix))
        # the output's delay and polarity undone against the recording, as
        # every reference metric undoes them against the reference: a late
        # passthrough is not a changed floor
        lag, sign = common.best_alignment(out[:n], mix[:n])
        out = sign * common.apply_lag(out, lag, n)
        i16, segs = (speech_segments(vad, mix, r["mix_path"])
                     if n == len(mix) else vad.segments(mix[:n]))
        o16 = torchaudio.functional.resample(out[:n], common.SR, 16000)
        o16 = o16[:len(i16)]
        i16 = i16[:len(o16)]
        if dnsmos_ovr is not None:
            acc["dnsmos"].append(dnsmos_ovr(o16))
        # every segment the detector marks is speech; the deletion rule reads
        # the ones of at least half a second (a shorter segment cannot show a
        # deletion apart from the detector's own jitter)
        m = torch.zeros(len(i16), dtype=torch.bool)
        for sg in segs:
            a, b = sg["start"], min(sg["end"], len(o16))
            m[a:b] = True
            if b - a < 8000:
                continue
            d = float(10 * torch.log10((o16[a:b].pow(2).mean() + 1e-9) /
                                       (i16[a:b].pow(2).mean() + 1e-9)))
            acc["seg_drop"].append(d)
        fr16, nf = 320, min(len(i16), len(o16))
        mfrm = m[:nf // fr16 * fr16].reshape(-1, fr16).float().mean(1) > 0.5
        # the non-voiced floor by 20 ms frame: frames clear of speech by a
        # guard band (a word's tail spills past the detector's boundary) whose
        # input is above digital silence, the output read on the same frames
        # (reading sample by sample where the input is large read any output
        # that is not sample-identical, a delayed passthrough included, as
        # suppression)
        in_frames = i16[:nf // fr16 * fr16].reshape(-1, fr16).pow(2).mean(1)
        out_frames = o16[:nf // fr16 * fr16].reshape(-1, fr16).pow(2).mean(1)
        any_speech = m[:nf // fr16 * fr16].reshape(-1, fr16).any(1)
        near_speech = any_speech.clone()
        for shift in range(1, NONVOICED_GUARD_FRAMES + 1):
            near_speech[shift:] |= any_speech[:-shift]
            near_speech[:-shift] |= any_speech[shift:]
        nonvoiced = (~near_speech &
                     (in_frames > 10**(NONVOICED_FLOOR_DBFS / 10)))
        if nonvoiced.any():
            acc["nonvoiced_supp_db"].append(
                float(10 * torch.log10(
                    (out_frames[nonvoiced].mean() + NONVOICED_EPS) /
                    (in_frames[nonvoiced].mean() + NONVOICED_EPS))))
        acc["terc"].append(
            common.tercile_kept_db(o16[:nf], i16[:nf], frame=fr16, act=mfrm))
        if m[:nf].any():
            acc["speech_rms_delta_db"].append(
                float(10 * torch.log10(
                    (o16[:nf][m[:nf]].pow(2).mean() + 1e-9) /
                    (i16[:nf][m[:nf]].pow(2).mean() + 1e-9))))
        acc["band_vs_in"].append(common.band_kept_db(out[:n], mix[:n]))
    return acc


def reuse_index(path, system, scorer_sha256=None):
    """Returns an earlier pass's records indexed by content identity.

    When that pass ran this scorer (or the sidecar whose digest is given);
    otherwise an empty index with the reason. Reuse never crosses a scorer
    change: the digest in the records' header must equal the caller's.
    """
    if not os.path.isfile(path):
        return {}, "no earlier records"
    try:
        metadata, records = observations.read(path)
    except (OSError, ValueError) as e:
        return {}, f"earlier records unreadable ({e})"
    if metadata.get("system") != system:
        return {}, "earlier records belong to another system"
    if metadata.get("scorer_sha256") != (scorer_sha256 or SCORER_SHA256):
        return {}, "earlier records were made by another scorer"
    index = {
        reuse_key(r["identity"]): r
        for r in records
        if r.get("identity") and r.get("output_status") == "ok"
    }
    return index, f"{len(index)} reusable records"


def assemble(name, rows, spk, vad, dnsmos_ovr, reuse=None, hashes=None):
    """Returns a system's per-item records over ``rows``.

    Reused from ``reuse`` (an earlier pass, by content identity) where the
    output samples, the mixture, the reference and the scoring semantics are
    unchanged, scored otherwise. The NISQA sample stays the first NISQA_N items
    of each condition with an output, exactly as a pass without reuse would take
    it.
    """
    outd = os.path.join(common.OUT_ROOT, name)
    hashes = hashes or common.file_hashes()
    reuse = reuse or {}
    by_cond = collections.defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    records, counts = [], collections.Counter()
    for _, items in sorted(by_cond.items()):
        n_have = 0
        for r in items:
            record = observations.item(r)
            records.append(record)
            op = os.path.join(outd, r["id"])
            if not os.path.exists(op):
                record["output_status"] = "missing"
                continue
            try:
                with open(op, "rb") as fh:
                    data = fh.read()
                out = common.load48(io.BytesIO(data))
            except (OSError, RuntimeError, ValueError):
                record["output_status"] = "unreadable"
                continue
            # the bytes measured, which the stamp's outputs digest vouches for
            record["output_sha256"] = hashlib.sha256(data).hexdigest()
            mix = common.load48(r["mix_path"])
            if len(out) != len(mix):
                record["output_status"] = "wrong_length"
                continue
            if not bool(torch.isfinite(out).all()):
                record["output_status"] = "nonfinite"
                continue
            n_have += 1
            want_nisqa = NISQA_ON and n_have <= NISQA_N
            record["identity"] = item_identity(r, sample_hash(out), hashes)
            old = reuse.get(reuse_key(record["identity"]))
            if old is not None:
                measurements = {
                    k: v
                    for k, v in old["measurements"].items()
                    if want_nisqa or k not in NISQA_KEYS
                }
                if want_nisqa and "nisqa_mos" not in measurements:
                    measurements.update(nisqa_measurements(out))
                acc = collections.defaultdict(list,
                                              observations.revive(measurements))
                counts["reused"] += 1
            else:
                acc = score_item(r, out, mix, spk, vad, dnsmos_ovr, want_nisqa)
                counts["scored"] += 1
            observations.capture(record, acc, {})
    if reuse:
        print(
            f"  reused {counts['reused']} records, scored {counts['scored']}"
            " items",
            flush=True,
        )
    return records


def aggregate(records, rows):
    """Returns the metrics file for ``rows`` from their per-item records.

    The level calibration from the clean items, each item's lost-block share
    from its coherence blocks under it, then the condition reducers over the
    pooled measurements. Pure: no audio is read, so any view of the set (the
    core, a panel) is this function over the whole-set records of its items.
    """
    by_id = {rec["id"]: rec for rec in records}
    by_cond = collections.defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    def ok(r):
        rec = by_id.get(r["id"])
        return (rec if rec is not None and rec.get("output_status") == "ok" else
                None)

    lvl = [
        ok(r)["metrics"]["out_in_rms_db"]
        for r in by_cond.get("clean", [])
        if ok(r)
    ]
    # a system's gain convention, removed before retention is read; bounded, so
    # an output that is silent on clean speech cannot calibrate its deleted
    # voice back into existence
    level_cal_db = (float(
        np.clip(np.median(lvl), -common.MAX_LEVEL_CAL_DB,
                common.MAX_LEVEL_CAL_DB)) if lvl else 0.0)
    scores = {}
    n_total_have = 0
    for cond, items in sorted(by_cond.items()):
        # reference-free items
        real = cond.startswith("real") or not items[0].get("clean_path")
        acc = collections.defaultdict(list)
        n_have = 0
        for r in items:
            rec = ok(r)
            if rec is None:
                continue
            n_have += 1
            for key, values in observations.revive(rec["measurements"]).items():
                if key == observations.BLOCKS:
                    # speech blocks that are gone: a percentile cannot see one
                    # dropped syllable
                    lost = levels.lost(values[0], level_cal_db)
                    pct = 100 * float(np.mean(lost))
                    acc["kept_coh_lost_pct"].append(pct)
                    rec["metrics"]["kept_coh_lost_pct"] = pct
                else:
                    acc[key].extend(values)
        n_total_have += n_have
        if not n_have:
            continue
        # items the system did not return: a mean over the rest is not a mean
        # over the condition
        d = {
            "n": n_have,
            "n_missing": len(items) - n_have,
            "metric_counts": {
                k: len(v) for k, v in acc.items()
            },
        }
        for name in NISQA_KEYS:
            if acc[name]:
                d[name] = float(np.mean(acc[name]))
        if not real:
            k = np.array(acc["kept"])
            d.update({
                "kept_mean": float(k.mean()),
                "kept_p10": float(np.percentile(k, 10)),
                "spk_sim": float(np.mean(acc["spk_sim"])),
                "kept_nb_mean": float(np.mean(acc["kept_nb"])),
                "spk_sim_nb": float(np.mean(acc["spk_sim_nb"])),
                "lag_median": float(np.median(acc["lag"])),
                "lag_max_abs": float(np.max(np.abs(acc["lag"]))),
            })
            if acc["si_sdr"]:
                d["si_sdr"] = float(np.mean(acc["si_sdr"]))
                d["si_sdr_nb"] = float(np.mean(acc["si_sdr_nb"]))
            for key in (
                    "kept_coh_mean",
                    "kept_coh_p10",
                    "kept_coh_nb_mean",
                    "kept_coh_nb_p10",
                    "kept_coh_lost_pct",
            ):
                if acc[key]:
                    d[key] = float(np.mean(acc[key]))
            d["n_polarity_flipped"] = int(sum(s < 0 for s in acc["polarity"]))
            # retention-only items (score_policy "retention")
            if acc["nonvoiced_supp_db"]:
                raw = float(np.mean(acc["nonvoiced_supp_db"]))
                d["nonvoiced_supp_raw_db"] = raw
                d["nonvoiced_supp_db"] = levels.energy_db(raw, level_cal_db)
                d["score_policy"] = "retention"
            t = np.array(acc["terc"], dtype=float)
            d.update({
                "kept_q1": float(np.nanmean(t[:, 0])),
                "kept_q2": float(np.nanmean(t[:, 1])),
                "kept_q3": float(np.nanmean(t[:, 2])),
            })
            b = np.array(acc["band"], dtype=float)
            d["band_kept_db"] = [
                float(np.nanmean(b[:, j])) for j in range(b.shape[1])
            ]
            hf = [
                levels.hf_error_db(item, level_cal_db) for item in acc["band"]
            ]
            if any(np.isfinite(hf)):
                d["hf_err_db"] = float(np.nanmean(hf))
            for key, energies in (
                ("babble_supp_db", acc["babble_energy"]),
                ("babble_supp_db_nb", acc["babble_energy_nb"]),
            ):
                if energies:
                    d[key] = float(
                        np.mean([
                            levels.suppression_db(e, level_cal_db)
                            for e in energies
                        ]))
                    d[key.replace("_db", "_raw_db", 1)] = float(
                        np.mean(
                            [levels.suppression_db(e, 0.0) for e in energies]))
        else:
            drops = acc["seg_drop"]
            raw_nv = (float(np.mean(acc["nonvoiced_supp_db"]))
                      if acc["nonvoiced_supp_db"] else float("nan"))
            d.update({
                "deleted_rate_pct":
                    (float(100 * np.mean(levels.deleted(drops, level_cal_db)))
                     if drops else float("nan")),
                "n_segments":
                    len(drops),
                "median_drop_db": (levels.energy_db(
                    np.median(drops), level_cal_db) if drops else float("nan")),
                "nonvoiced_supp_raw_db":
                    raw_nv,
                "nonvoiced_supp_db":
                    levels.energy_db(raw_nv, level_cal_db),
                "speech_rms_delta_db":
                    (float(np.mean(acc["speech_rms_delta_db"]))
                     if acc["speech_rms_delta_db"] else float("nan")),
            })
            t = np.array(acc["terc"], dtype=float)
            d.update({
                "kept_q1": float(np.nanmean(t[:, 0])),
                "kept_q2": float(np.nanmean(t[:, 1])),
                "kept_q3": float(np.nanmean(t[:, 2])),
            })
            b = np.array(acc["band_vs_in"], dtype=float)
            d["band_vs_in_db"] = [
                float(np.nanmean(b[:, j])) for j in range(b.shape[1])
            ]
            if acc["dnsmos"]:
                d["dnsmos_ovr"] = float(np.mean(acc["dnsmos"]))
        d["out_in_rms_db"] = float(np.mean(acc["out_in_rms_db"]))
        d["level_err_db"] = float(
            np.mean([levels.level_error_db(v) for v in acc["out_in_rms_db"]]))
        scores[cond] = d
    scores["_level_cal_db"] = level_cal_db
    scores["_n_missing"] = len(rows) - n_total_have
    scores["_set_id"] = common.set_id()
    scores["_subset"] = common.SUBSET
    scores["_sealed"] = common.is_sealed(rows)
    scores["_bands_hz"] = [list(b) for b in common.BANDS_HZ]
    scores.update(anchors.stamp())
    return scores


def dnsmos_scorer():
    """Returns DNSMOS's overall score as a function of 16 kHz audio.

    None when DNSMOS cannot load.
    """
    try:
        # pylint: disable-next=import-outside-toplevel
        from torchmetrics.functional.audio import dnsmos
    # a diagnostic only: any failure to load it leaves the column empty, never
    # the scores
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"  [warn] DNSMOS unavailable: {e}", flush=True)
        return None

    def overall(w16):
        s = dnsmos.deep_noise_suppression_mean_opinion_score(w16,
                                                             fs=16000,
                                                             personalized=False)
        return float(s[..., 3].mean())

    return overall


def score_system(name, rows, spk, vad, dnsmos_ovr, records=None, reuse=None):
    """Scores a system over ``rows``.

    Assembles its per-item records, then aggregates them.
    """
    if name == "REF":
        rows = [r for r in rows if r.get("clean_path")]
    assembled = assemble(name, rows, spk, vad, dnsmos_ovr, reuse=reuse)
    scores = aggregate(assembled, rows)
    if records is not None:
        records.extend(assembled)
    return scores


def accept_render(name, records, stamp):
    """Accepts scores only of the outputs their render's stamp vouches for.

    The bytes each record was measured on (``output_sha256``) must add up to the
    stamp's outputs digest (``stamprender.digest_of``): an output changed or
    gone since the stamp means these are scores of another render. A render the
    set links in rather than stamps (its mixtures, its references) carries no
    digest and is not checked.

    Raises:
        SystemExit: The outputs read are not the stamped render.
    """
    stamped = (stamp or {}).get("outputs_sha256")
    if not stamped:
        return
    if stamp.get("n_files") not in (None, len(records)):
        raise SystemExit(
            f"{name}: {len(records)} of the {stamp['n_files']} outputs its"
            " stamp vouches for were read: a view of a render is derived from"
            " its whole-set records (--aggregate-only), never scored again")
    hashes = {rec["id"]: rec.get("output_sha256") for rec in records}
    if not all(hashes.values()) or stamprender.digest_of(hashes) != stamped:
        raise SystemExit(
            f"{name}: the outputs read are not the render its stamp vouches for"
            " (changed or gone since it was stamped): render it again or stamp"
            " it again")


def view_source(path, system, scorer_sha256=None):
    """Returns why whole-set records cannot stand for a view, or ''.

    The records under ``path`` must be this set's, this scorer's (or the
    sidecar's whose digest is given), and the whole set's.
    """
    if not os.path.isfile(path):
        return "no whole-set records"
    try:
        metadata, _ = observations.read(path)
    except (OSError, ValueError) as e:
        return f"unreadable ({e})"
    if metadata.get("system") != system:
        return "records belong to another system"
    if metadata.get("set_id") != common.set_id():
        return (f"records of set {str(metadata.get('set_id'))[:12]}, current"
                f" {str(common.set_id())[:12]}")
    if metadata.get("subset", "all") != "all":
        return (
            f"records of subset {metadata.get('subset')!r}, not the whole set")
    if metadata.get("scorer_sha256") != (scorer_sha256 or SCORER_SHA256):
        return "records were made by another scorer: re-score the whole set"
    return ""


def view_records(path, system, ids, scorer_sha256=None):
    """Returns the whole-set records under ``path`` that a view keeps.

    A tuple (problem, metadata, records): ``problem`` says why the records
    cannot stand for the view (``view_source``) and is '' when they can. A view
    item the records lack ends the run: a view is never scored on its own.
    """
    problem = view_source(path, system, scorer_sha256)
    if problem:
        return problem, None, None
    metadata, whole = observations.read(path)
    records = [rec for rec in whole if rec["id"] in ids]
    absent = sorted(ids - {rec["id"] for rec in records})
    if absent:
        raise SystemExit(
            f"{path} holds no record for {len(absent)} items of subset"
            f" {common.SUBSET!r} (first: {absent[0]}): score the whole set"
            " first")
    return "", metadata, records


def render_problem(name):
    """Why a system's outputs may not be scored, or None.

    Renders belong to one build of the set and carry its id in
    ``<name>.meta.json`` (written by runsystem.py / runfield.sh). Outputs
    without that record, or from another build, would be scored against
    references they were never made for.
    """
    meta_path = os.path.join(common.RESULTS, f"{name}.meta.json")
    if not os.path.exists(meta_path):
        return (
            "no identity (meta.json missing): render it with runsystem.py or"
            " runfield.sh")
    with open(meta_path, encoding="utf-8") as fh:
        recorded = json.load(fh).get("set_id")
    current = common.set_id()
    if not recorded:
        return "meta carries no set_id: re-render it with the current harness"
    if current and recorded != current:
        return (
            f"rendered on set {recorded[:12]}, current set is {current[:12]}:"
            " re-render it")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="")
    ap.add_argument("--shard", default="", help="i/k over the system list")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--conds", default="", help="comma substrings; debug only")
    ap.add_argument("--limit",
                    type=int,
                    default=0,
                    help="items/cond; debug only")
    ap.add_argument("--suffix", default="", help="results filename suffix")
    ap.add_argument(
        "--anchor-reference",
        action="store_true",
        help=(
            "REF only: score complete conditions required by anchor definitions"
        ),
    )
    ap.add_argument(
        "--reuse-from",
        default="",
        metavar="DIR",
        help=(
            "results of an earlier pass of this scorer: an item whose output"
            " samples, mixture, reference and semantics are unchanged keeps its"
            " record; only the rest is scored"),
    )
    ap.add_argument(
        "--aggregate-only",
        default="",
        metavar="DIR",
        help=("derive this subset's files (BENCH_SUBSET) from the whole-set"
              " per-item records under DIR; no audio is read"),
    )
    a = ap.parse_args()
    if a.aggregate_only and (a.conds or a.limit or a.reuse_from):
        ap.error(
            "--aggregate-only takes no debug filtering and no --reuse-from")

    rows = common.manifest()
    common.guard_private_results(rows)
    if a.anchor_reference:
        if a.systems != "REF" or a.conds or a.limit or a.suffix:
            ap.error("--anchor-reference requires --systems REF and no debug"
                     " filtering")
        rows = [
            r for r in rows if r["condition"] in anchors.reference_conditions()
        ]
    if a.conds:
        pats = a.conds.split(",")
        rows = [r for r in rows if any(p in r["condition"] for p in pats)]
    if a.limit:
        seen = collections.Counter()
        keep = []
        for r in rows:
            if seen[r["condition"]] < a.limit:
                keep.append(r)
                seen[r["condition"]] += 1
        rows = keep
    if a.systems:
        systems = a.systems.split(",")
    else:
        systems = sorted(
            os.path.basename(d)
            for d in glob.glob(common.OUT_ROOT + "/*")
            if os.path.isdir(d))
    # a filtered pass (--conds, --limit) scores what it names and nothing
    # else: INPUT's whole-set files are never replaced by a selection
    if ("INPUT" not in systems and systems != ["REF"] and
            not (a.conds or a.limit)):
        systems = ["INPUT"] + systems
    if (a.conds or a.limit) and not a.suffix:
        print(
            "filtered pass: the whole-set files of the named systems are "
            "replaced by this selection (the board refuses them); pass "
            "--suffix to keep them",
            flush=True)
    # every output directory is a candidate; only identified ones are scored,
    # and a filtered pass scores the discovered INPUT no more than a named one
    if not a.systems:
        systems = [
            s for s in systems if (s == "INPUT" and not (a.conds or a.limit)) or
            (s != "INPUT" and not render_problem(s))
        ]
    if a.shard:
        i, k = (int(x) for x in a.shard.split("/"))
        systems = systems[i::k]
    print(f"scoring {len(systems)} systems: {systems}", flush=True)

    os.makedirs(common.OUT_ROOT + "/INPUT", exist_ok=True)
    if not glob.glob(common.OUT_ROOT + "/INPUT/*.wav"):
        for r in rows:
            lp = os.path.join(common.OUT_ROOT, "INPUT", r["id"])
            if not os.path.exists(lp):
                os.symlink(os.path.abspath(r["mix_path"]), lp)

    spk = vad = dnsmos_ovr = None
    if not a.aggregate_only:
        # reference embeddings are the same for every system: cache them next to
        # the set
        set_tag = (common.set_id() or "noset")[:16]
        spk_cache = os.path.join(common.EVAL, ".cache", "spkemb", set_tag)
        spk = common.SpeakerEmbedder(cache_dir=spk_cache)
        vad = common.Vad()
        dnsmos_ovr = dnsmos_scorer()

    os.makedirs(common.RESULTS, exist_ok=True)
    for s in systems:
        dst = os.path.join(common.RESULTS, f"{s}{a.suffix}.metrics.json")
        own = os.path.join(common.RESULTS, f"{s}{a.suffix}.items.jsonl")
        if a.aggregate_only:
            if s == "REF" and not a.anchor_reference:
                print(
                    "skip REF: derived with --systems REF --anchor-reference",
                    flush=True,
                )
                continue
            src = os.path.join(a.aggregate_only, f"{s}{a.suffix}.items.jsonl")
            view = rows
            if s == "REF":
                view = [r for r in rows if r.get("clean_path")]
            ids = {r["id"] for r in view}
            problem, metadata, records = view_records(src, s, ids)
            if problem:
                print(f"skip {s}: {problem}", flush=True)
                continue
            print(f"\n=== {s}  ({len(records)} whole-set records)", flush=True)
            scores = aggregate(records, view)
            identity = metadata.get("render_identity")
        else:
            if not os.path.isdir(os.path.join(common.OUT_ROOT, s)):
                print(f"skip {s} (no output dir)", flush=True)
                continue
            problem = render_problem(s)
            if problem:
                print(f"skip {s}: {problem}", flush=True)
                continue
            n_out = len(glob.glob(os.path.join(common.OUT_ROOT, s, "*.wav")))
            print(f"\n=== {s}  ({n_out} wavs)", flush=True)
            # a system is never skipped as scored: its existing records (or an
            # earlier pass named by --reuse-from) are reused item by item where
            # the content identity holds, so a changed output or another scorer
            # can never leave a stale score behind
            reuse, why = {}, "--force: every item scored afresh"
            if not a.force:
                if a.reuse_from:
                    reuse, why = reuse_index(
                        os.path.join(a.reuse_from,
                                     f"{s}{a.suffix}.items.jsonl"),
                        s,
                    )
                    why = f"reuse from {a.reuse_from}: {why}"
                elif os.path.isfile(own):
                    reuse, why = reuse_index(own, s)
                    why = (
                        f"existing records: {why}; every item re-validated by"
                        " content")
                else:
                    why = "no earlier records"
            print(f"  {why}", flush=True)
            records = []
            scores = score_system(s,
                                  rows,
                                  spk,
                                  vad,
                                  dnsmos_ovr,
                                  records=records,
                                  reuse=reuse)
            meta_path = os.path.join(common.RESULTS, f"{s}.meta.json")
            with open(meta_path, encoding="utf-8") as fh:
                identity = json.load(fh)
            if not (a.conds or a.limit):
                accept_render(s, records, identity)
        if a.anchor_reference:
            scores["_scope"] = "anchor_reference"
        # one implementation per board (leaderboard.py)
        scores["_scorer_sha256"] = SCORER_SHA256
        # the audio these scores read: the board refuses them once the stamp
        # vouches for other outputs
        scores["_outputs_sha256"] = (identity or {}).get("outputs_sha256")
        # a filtered look at part of the set, never a board row: too few of the
        # render's outputs to vouch for it
        if a.conds or a.limit:
            scores["_selection"] = {"conds": a.conds, "limit": a.limit}
            scores["_outputs_sha256"] = None
        observations.write(
            os.path.join(common.RESULTS, f"{s}{a.suffix}.items.jsonl"),
            records,
            {
                "system": s,
                "set_id": scores["_set_id"],
                "subset": common.SUBSET,
                "anchors_version": scores["_anchors_version"],
                "anchors_sha256": scores["_anchors_sha256"],
                "level_cal_db": scores["_level_cal_db"],
                "render_identity": identity,
                "scorer_sha256": SCORER_SHA256,
                "n_expected": len(records),
                "sealed": common.is_sealed(rows),
            },
        )
        # whole or absent: parallel passes re-validate INPUT
        tmp = f"{dst}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(scores, fh, indent=1)
        os.replace(tmp, dst)
        n_conditions = len([k for k in scores if not k.startswith("_")])
        print(
            f"  level_cal {scores['_level_cal_db']:+.2f} dB |"
            f" {n_conditions} conditions -> {dst}",
            flush=True,
        )


if __name__ == "__main__":
    main()
