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
"""Metric primitives and shared helpers of the benchmark.

The tests use synthetic signals and no models.
"""

# pylint: disable=protected-access,unused-argument

import argparse
import json
import math
import os

import numpy as np
import pytest
import soundfile as sf
import torch

import common
from inworld import voice_isolation

SR = common.SR
HALF_DB = 20 * math.log10(0.5)  # -6.02 dB


def speechlike(n, seed=0, level=0.1):
    """Zero-mean noise under a slow amplitude envelope.

    The envelope makes frame energies vary like speech.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, n)
    env = 0.2 + 0.8 * (0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t)).pow(2)
    return torch.randn(n, generator=g) * env * level


def white(n, seed=0, level=0.1):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed)) * level


def delayed(ref, lag):
    """ref delayed by lag samples (negative: advanced).

    The result has the same length as ref and is zero-filled.
    """
    if lag >= 0:
        return torch.cat([torch.zeros(lag), ref])[:len(ref)]
    return torch.cat([ref[-lag:], torch.zeros(-lag)])


def snr_db(signal, noise):
    return float(10 * torch.log10(signal.pow(2).mean() / noise.pow(2).mean()))


# Reference scorer implementations, retained to check the shared helpers.


def _old_best_lag_fft(out, ref, max_lag=4800):  # scorebench.py before the move
    n = min(len(out), len(ref))
    a, b = out[:n] - out[:n].mean(), ref[:n] - ref[:n].mean()
    if float(a.abs().max()) < 1e-8 or float(b.abs().max()) < 1e-8:
        return 0
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cc = torch.fft.irfft(
        torch.fft.rfft(a, size) * torch.fft.rfft(b, size).conj(), size)
    cc = torch.cat([cc[-max_lag:], cc[:max_lag + 1]])
    return int(cc.argmax()) - max_lag


# a sidecar's own copy before the helpers moved here
def _old_best_lag_conv(out, ref, max_lag=4800):
    n = min(len(out), len(ref))
    a, b = out[:n], ref[:n]
    padded = torch.nn.functional.pad(a.view(1, 1, -1), (max_lag, max_lag))
    c = torch.nn.functional.conv1d(padded, b.view(1, 1, -1)).view(-1)
    return int(torch.argmax(c).item() - max_lag)


def _old_apply_lag_cat(out, lag, n):  # a sidecar's copy before the move
    o = (out[lag:]
         if lag > 0 else torch.cat([torch.zeros(-lag, dtype=out.dtype), out]))
    return torch.nn.functional.pad(o, (0, max(0, n - len(o))))[:n]


def _old_nb8k(x):  # a sidecar's copy before the move
    # pylint: disable-next=import-outside-toplevel
    import torchaudio

    y = torchaudio.functional.resample(x, common.SR, 16000)
    return torchaudio.functional.resample(y, 16000, common.SR)[:len(x)]


def test_alignment_undoes_a_flipped_sign_and_only_a_flipped_sign():
    ref = white(SR, seed=21)
    for lag in (0, 37, -20):
        assert common.best_alignment(delayed(ref, lag), ref) == (lag, 1)
        assert common.best_alignment(-delayed(ref, lag), ref) == (lag, -1)
    t = torch.arange(SR) / SR
    # strong negative lobes
    voiced = torch.sin(2 * torch.pi * 120 * t) * torch.hann_window(SR)
    assert common.best_alignment(voiced, voiced) == (0, 1)
    # uncorrelated: never "flipped"
    assert common.best_alignment(white(SR, seed=22), ref)[1] == 1
    assert common.best_alignment(torch.zeros(SR), ref) == (0, 1)
    assert common.best_lag(-delayed(ref, 37), ref) == 37


def test_shared_alignment_helpers_match_the_scorers_previous_copies():
    ref = white(SR, seed=11)
    n = len(ref)
    for lag in (0, 37, -20, 2400):
        out = delayed(ref, lag)
        assert common.best_lag(out, ref) == lag
        assert _old_best_lag_fft(out, ref) == lag
        assert _old_best_lag_conv(out, ref) == lag
        back = common.apply_lag(out, lag, n)
        assert torch.equal(back, _old_apply_lag_cat(out, lag, n))
        m = abs(lag)
        assert torch.allclose(back[m:n - m], ref[m:n - m])
    x = white(SR, seed=12)
    assert torch.allclose(common.nb8k(x), _old_nb8k(x))
    assert common.MAX_LAG == 4800 and common.NB_RATE == 16000


def test_best_lag_is_zero_for_silence_and_aligned_signals():
    ref = speechlike(SR, seed=1)
    assert common.best_lag(ref, ref) == 0
    assert common.best_lag(torch.zeros(SR), ref) == 0
    # a DC offset does not fool the mean-removed search
    assert common.best_lag(ref + 0.3, ref) == 0


def test_apply_lag_pads_or_trims_to_the_requested_length():
    out = torch.arange(10.0)
    assert torch.equal(common.apply_lag(out, 0, 12),
                       torch.cat([out, torch.zeros(2)]))
    assert torch.equal(common.apply_lag(out, 3, 5), out[3:8])
    assert torch.equal(common.apply_lag(out, -2, 5),
                       torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))


def test_nb8k_removes_everything_above_8_khz():
    x = white(2 * SR, seed=3)
    y = common.nb8k(x)
    assert len(y) == len(x)
    spec_x = torch.fft.rfft(x).abs().pow(2)
    spec_y = torch.fft.rfft(y).abs().pow(2)
    freqs = torch.fft.rfftfreq(len(x), 1 / SR)
    hi, lo = freqs >= 9000, freqs <= 7000
    assert float(spec_y[hi].sum() / spec_x[hi].sum()) < 1e-3  # > 30 dB down
    # passband kept (< 0.5 dB droop)
    passband = float(spec_y[lo].sum() / spec_x[lo].sum())
    assert 0.9 < passband <= 1.01


def test_mix_at_snr_realises_the_requested_snr_and_trims_to_common_length():
    clean = speechlike(2 * SR, seed=1)
    noise = white(3 * SR, seed=2, level=0.05)
    for snr in (-5.0, 0.0, 10.0):
        mix, ref = common.mix_at_snr(clean, noise, snr)
        assert len(mix) == len(ref) == 2 * SR
        assert torch.equal(ref, clean)
        assert snr_db(ref, mix - ref) == pytest.approx(snr, abs=0.05)


def test_si_sdr_scale_and_offset_invariant_ordered_by_noise_large_for_identity(
):
    ref = speechlike(SR, seed=2)
    noise = white(SR, seed=3, level=0.1)
    assert common.si_sdr(ref, ref) > 60
    a, b, c = (common.si_sdr(ref + g * noise, ref) for g in (0.01, 0.1, 1.0))
    assert a > b > c
    scaled = 5 * (ref + 0.1 * noise)
    assert common.si_sdr(scaled, ref) == pytest.approx(b, abs=1e-3)
    shifted = ref + 0.1 * noise + 0.3
    assert common.si_sdr(shifted, ref) == pytest.approx(b, abs=1e-3)


def test_voice_kept_db_reads_a_gain_inside_active_frames():
    clean = speechlike(2 * SR, seed=4)
    assert common.voice_kept_db(clean, clean) == pytest.approx(0.0, abs=1e-6)
    assert common.voice_kept_db(0.5 * clean, clean) == pytest.approx(HALF_DB,
                                                                     abs=0.01)
    quiet = (clean.reshape(-1, 960).pow(2).mean(1)
             <= clean.reshape(-1, 960).pow(2).mean(1).median())
    touched = clean.clone().reshape(-1, 960)
    # damage confined to the frames below the median energy
    touched[quiet] *= 0.1
    assert common.voice_kept_db(touched.reshape(-1),
                                clean) == pytest.approx(0.0, abs=1e-6)


def test_band_kept_db_reports_per_band_gain_and_invented_high_frequency():
    assert (len(common.BANDS_HZ) == 7 and common.BANDS_HZ[0][0] == 0 and
            common.BANDS_HZ[-1][1] == SR // 2)
    assert all(common.BANDS_HZ[i][1] == common.BANDS_HZ[i + 1][0]
               for i in range(len(common.BANDS_HZ) - 1))
    ref = white(2 * SR, seed=5)
    assert all(abs(v) < 1e-6 for v in common.band_kept_db(ref, ref))
    assert all(v == pytest.approx(HALF_DB, abs=0.05)
               for v in common.band_kept_db(0.5 * ref, ref))
    t = torch.arange(2 * SR) / SR
    invented = common.band_kept_db(
        ref + 0.3 * torch.sin(2 * math.pi * 20000 * t), ref)
    assert invented[-1] > 6 and all(abs(v) < 0.5 for v in invented[:-1])
    dulled = common.band_kept_db(common.nb8k(ref), ref)
    assert dulled[-1] < -30 and abs(dulled[0]) < 0.5


def test_tercile_kept_db_gain_nans_and_explicit_mask():
    ref = speechlike(2 * SR, seed=6)  # 100 frames of 960 samples
    assert all(v == pytest.approx(0.0, abs=1e-6)
               for v in common.tercile_kept_db(ref, ref))
    assert all(v == pytest.approx(HALF_DB, abs=0.02)
               for v in common.tercile_kept_db(0.5 * ref, ref))
    # under one frame
    assert all(
        math.isnan(v) for v in common.tercile_kept_db(ref[:500], ref[:500]))
    # < 9 frames
    assert all(
        math.isnan(v)
        for v in common.tercile_kept_db(ref[:960 * 5], ref[:960 * 5]))
    act = torch.zeros(100, dtype=torch.bool)
    act[:30] = True
    out = ref.clone()
    # destroyed outside the mask: invisible when the mask says so
    out[30 * 960:] = 0
    assert all(v == pytest.approx(0.0, abs=1e-6)
               for v in common.tercile_kept_db(out, ref, act=act))
    # without the mask every tercile mixes intact and destroyed frames: a
    # mean-energy loss
    assert all(v < -3 for v in common.tercile_kept_db(out, ref))


def test_load48_resamples_to_48k_mono_float32_and_pads_or_trims(tmp_path):
    n16 = 16000
    mono = np.random.default_rng(0).standard_normal(n16)
    stereo = np.stack([mono] * 2, 1).astype(np.float32)
    p16 = str(tmp_path / "s16.wav")
    sf.write(p16, 0.1 * stereo, 16000)
    y = common.load48(p16)
    assert (y.dtype == torch.float32 and y.ndim == 1 and
            abs(len(y) - 3 * n16) <= 1)
    assert len(common.load48(p16, secs=0.5)) == SR // 2
    assert len(common.load48(p16, secs=2.0)) == 2 * SR
    x48 = white(SR, seed=7)
    p48 = str(tmp_path / "m48.wav")
    sf.write(p48, x48.numpy(), SR, subtype="FLOAT")
    assert torch.allclose(common.load48(p48), x48)


def test_set_id_and_manifest_resolve_paths_under_the_set_root(tmp_path):
    root = tmp_path / "set"
    root.mkdir()
    rows = [
        {
            "id": "a.wav",
            "condition": "babble@+4",
            "mix_path": "mix/a.wav",
            "clean_path": "ref/a.wav",
        },
        {
            "id": "r.wav",
            "condition": "real_x",
            "mix_path": "mix/r.wav",
            "clean_path": None,
        },
    ]
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1]) + "\n")
    json.dump({
        "set_id": "abc123",
        "n_items": 2
    }, open(root / "checksums.json", "w", encoding="utf-8"))
    assert common.set_id(str(root)) == "abc123"
    assert common.set_id(str(tmp_path / "nowhere")) is None
    m = common.manifest(str(root))
    assert [r["id"] for r in m] == ["a.wav", "r.wav"]
    assert m[0]["mix_path"] == str(root / "mix" / "a.wav")
    assert m[0]["clean_path"] == str(root / "ref" / "a.wav")
    assert m[1]["clean_path"] is None


def test_outputs_digest_reads_the_stamp_and_is_none_without_one(tmp_path):
    json.dump({
        "name": "sys",
        "outputs_sha256": "ab" * 32
    }, open(tmp_path / "sys.meta.json", "w", encoding="utf-8"))
    json.dump({"name": "INPUT"},
              open(tmp_path / "INPUT.meta.json", "w", encoding="utf-8"))
    assert common.outputs_digest("sys", str(tmp_path)) == "ab" * 32
    # the set's own mixtures, linked: nothing rendered, nothing to name
    assert common.outputs_digest("INPUT", str(tmp_path)) is None
    assert common.outputs_digest("unstamped", str(tmp_path)) is None


def test_speaker_embedder_caches_reference_embeddings(tmp_path, monkeypatch):
    calls = []

    class _Model:

        def __call__(self, x):
            calls.append(x.shape)
            return torch.ones(1, 4)

    emb = common.SpeakerEmbedder.__new__(common.SpeakerEmbedder)
    emb.m, emb.cache_dir, emb._mem = _Model(), str(tmp_path), {}
    wav = torch.randn(common.SR)
    a = emb.emb(wav, key="item.ref")
    b = emb.emb(wav, key="item.ref")  # memory hit
    emb._mem.clear()
    c = emb.emb(wav, key="item.ref")  # disk hit
    emb.emb(wav)  # uncached call
    assert len(calls) == 2 and torch.equal(a, b) and torch.equal(a, c)
    assert (tmp_path / "item.ref.pt").exists()
    assert emb.sim(wav, wav, key_b="item.ref") == pytest.approx(1.0)


def _row(cond, idx, lang="en", paired=True, family=None):
    r = {
        "id": f"{cond.replace('@', 'AT').replace('+', 'p')}_{idx:03d}.wav",
        "condition": cond,
        "family": family or cond.split("@")[0].split("_")[0],
        "language": lang,
    }
    if paired:
        r["clean_path"] = f"ref/{r['id']}"
    return r


def test_subset_splits_the_multilingual_core_from_the_english_extension():
    assert common.subset(_row("babble@+4", 7, "ta")) == "core"
    # English share of the base draw
    assert common.subset(_row("babble@+4", 23, "en")) == "core"
    # added above the limit
    assert common.subset(_row("babble@+4", 48, "en")) == "extension"
    # a new level on the base utterances
    assert common.subset(_row("babble@-5", 3, "de")) == "core"
    # English-only family
    assert common.subset(_row("stationary@-5", 3, "en")) == "extension"
    # the softly spoken retention family
    assert common.subset(_row("soft@+0", 3, "en")) == "extension"
    # and the whispered one
    assert common.subset(_row("whisper@+0", 3, "en")) == "extension"
    assert (common.subset(
        _row("nonverbal_clean", 3, "en",
             family="nonverbal_clean")) == "extension")
    assert (common.subset(_row("nonverbal@+0", 3, "en",
                               family="ambience")) == "extension")
    # real: core throughout
    assert common.subset(_row("real_multi", 90, "ko", paired=False)) == "core"
    assert common.subset(_row("real_noisy", 60, "en", paired=False)) == "core"


def test_manifest_filters_by_subset_from_the_argument_or_the_environment(
        tmp_path, monkeypatch):
    rows = [
        _row("clean", 1, "fr"),
        _row("clean", 50, "en"),
        _row("traffic@+0", 1, "en"),
    ]
    with open(tmp_path / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ids = [r["id"] for r in common.manifest(str(tmp_path))]
    assert ids == [r["id"] for r in rows]
    core = [r["id"] for r in common.manifest(str(tmp_path), which="core")]
    assert core == [rows[0]["id"]]
    assert len(common.manifest(str(tmp_path), which="extension")) == 2
    monkeypatch.setattr(common, "SUBSET", "core")
    assert len(common.manifest(str(tmp_path))) == 1
    assert common.manifest(str(tmp_path))[0]["clean_path"].startswith(
        str(tmp_path))


def test_coherent_kept_reads_zero_for_voice_and_very_negative_for_bed_alone():
    g = torch.Generator().manual_seed(3)
    n = 48000 * 2
    voice = torch.sin(torch.linspace(0, 2 * math.pi * 220 * 2,
                                     n)) * (torch.rand(n, generator=g) > 0.3)
    # louder than the voice
    bed = torch.randn(n, generator=g) * float(voice.pow(2).mean().sqrt()) * 1.8
    assert abs(float(common.coherent_kept_db(voice, voice).mean())) < 0.01
    # the voice is in there: ~0 dB
    assert float(common.coherent_kept_db(voice + bed, voice).mean()) < 0.5
    # a bed alone is not kept speech
    assert float(common.coherent_kept_db(bed, voice).max()) < -20
    kept = common.coherent_kept_db(0.1 * voice, voice)
    assert float(kept.mean()) == pytest.approx(-20.0, abs=0.1)
    # the energy ratio would have called it kept
    assert float(common.voice_kept_db(bed, voice)) > 0


def test_coherent_kept_follows_the_activity_mask_so_quiet_speech_counts():
    n = 48000 * 2
    voice = torch.sin(torch.linspace(0, 2 * math.pi * 200 * 2, n)) * 0.3
    voice[n // 2:] *= 0.05  # the second half is quiet speech, 26 dB down
    # a VAD that hears speech throughout
    act16 = torch.ones(n // 3, dtype=torch.bool)
    mask = common.block_mask(act16, n // 4800)
    assert bool(mask.all())
    gone = voice.clone()
    gone[n // 2:] = 0  # the quiet half deleted
    # seen with the mask
    assert float(common.coherent_kept_db(gone, voice, act=mask).min()) < -60
    faint = voice.clone()
    # 66 dB down: under the 40 dB floor, so not measurable either way
    faint[n // 2:] *= 0.01
    gone2 = faint.clone()
    gone2[n // 2:] = 0
    assert float(common.coherent_kept_db(gone2, faint, act=mask).min()) > -1
    kept = common.coherent_kept_db(voice, voice, act=mask)
    assert abs(float(kept.min())) < 0.01
    partial = torch.zeros(n // 3, dtype=torch.bool)
    partial[:n // 6] = True
    # half the blocks marked
    assert common.block_mask(partial, n // 4800).sum() == n // 4800 // 2


def test_coherent_blocks_split_the_voice_from_the_residual():
    g = torch.Generator().manual_seed(7)
    n = 48000 * 2
    voice = torch.sin(torch.linspace(0, 2 * math.pi * 180 * 2, n)) * 0.3
    # a bed 10 dB under the voice
    bed = torch.randn(n, generator=g) * 0.3 * 10**(-10 / 20)
    kept, resid, loud = common.coherent_blocks(0.5 * voice + bed, voice)
    assert bool(loud.all()) and len(kept) == len(resid) == n // 4800
    # the voice, halved
    assert float(kept.mean()) == pytest.approx(20 * math.log10(0.5), abs=0.1)
    bed_vs_voice = 10 * math.log10(
        float(bed.pow(2).mean()) / float(voice.pow(2).mean()))
    # the bed, as it is
    assert float(resid.mean()) == pytest.approx(bed_vs_voice, abs=0.5)
    k2, r2, _ = common.coherent_blocks(voice, voice)
    # nothing but the voice
    assert abs(float(k2.mean())) < 0.01 and float(r2.max()) < -70
    empty = common.coherent_blocks(torch.zeros(100), torch.zeros(100))
    assert all(len(t) == 0 for t in empty)
    # one primitive
    assert torch.equal(common.coherent_kept_db(0.5 * voice + bed, voice), kept)


def test_pick_rows_by_name_prefix_pairing_and_limit():
    rows = [
        {
            "id": "a",
            "condition": "babble@+4",
            "clean_path": "r/a"
        },
        {
            "id": "b",
            "condition": "babble@-5",
            "clean_path": "r/b"
        },
        {
            "id": "c",
            "condition": "real_noisy",
            "clean_path": None
        },
        {
            "id": "d",
            "condition": "real_wild",
            "clean_path": None
        },
    ]

    def ids(rs):
        return [r["id"] for r in rs]

    assert ids(common.pick_rows(rows, "babble@+4")) == ["a"]
    assert ids(common.pick_rows(rows, "babble@+4,real_wild")) == ["a", "d"]
    # a prefix when no exact name matches
    assert ids(common.pick_rows(rows, "real")) == ["c", "d"]
    assert ids(common.pick_rows(rows, "")) == ["a", "b", "c", "d"]
    assert ids(common.pick_rows(rows, "", paired=True)) == ["a", "b"]
    assert ids(common.pick_rows(rows, "", paired=False)) == ["c", "d"]
    assert ids(common.pick_rows(rows, "", limit=3)) == ["a", "b", "c"]


def test_model_tag_and_ops_path_stay_out_of_the_published_results():
    assert common.model_tag("voice-isolator-v1") == "voice-isolator-v1"
    assert common.model_tag("/x/y/export/model.safetensors") == "model"
    assert common.model_tag("staging/sweeps/candidate_a.pt") == "candidate_a"
    assert common.ops_path("m", "seams", "/tmp/x.json") == "/tmp/x.json"
    p = common.ops_path("staging/a/b.safetensors", "seams")
    assert (p.endswith(os.path.join("staging", "ops", "b.seams.json")) and
            "results" not in p)


def test_isolator_runner_routes_chunking_and_alpha(monkeypatch):
    calls = []

    class _Vi:
        device = "cpu"

        def isolate(self, x, sr, **kw):
            calls.append(("isolate", sr, kw))
            return x

        def enhance(self, x, sr, alpha, **kw):
            calls.append(("enhance", alpha, kw))
            return x

    seen = {}
    monkeypatch.setattr(
        voice_isolation.VoiceIsolator,
        "load",
        classmethod(lambda cls, m, **kw: seen.update(model=m, **kw) or _Vi()),
    )
    ap = argparse.ArgumentParser()
    common.add_model_args(ap)
    a = ap.parse_args([
        "--model", "ckpt.pt", "--precision", "fp16", "--mode", "streaming",
        "--lookahead", "2"
    ])
    _, run = common.isolator_runner(a)
    assert seen == {
        "model": "ckpt.pt",
        "stage2": None,
        "device": common.DEV,
        "precision": "fp16",
        "mode": "streaming",
        "lookahead_frames": 2,
    }
    x = torch.zeros(10)
    run(x, 4.0, 1.0)
    assert calls[-1] == (
        "isolate",
        common.SR,
        {
            "output_sample_rate": common.SR,
            "chunk_seconds": 4.0,
            "overlap_seconds": 1.0,
        },
    )
    a = ap.parse_args(["--alpha", "0.5"])
    _, run = common.isolator_runner(a)
    run(x)
    assert calls[-1][:2] == ("enhance", 0.5)


def test_the_voice_activity_detector_is_a_release_the_cohort_was_cut_with():
    """The frozen deletion cohort counts the detector's usable segments.

    silero-vad 6.2.3 segments two of the cohort's recordings differently from
    6.2.0 to 6.2.2, and a row whose counts differ from the cohort's is
    incomplete; the ``eval`` extra pins the admissible releases and an
    installed detector outside them must fail here rather than in a board.
    """
    pytest.importorskip("silero_vad")
    # pylint: disable-next=import-outside-toplevel
    import importlib.metadata
    # pylint: disable-next=import-outside-toplevel
    import re

    # pylint: disable-next=import-outside-toplevel
    from packaging import specifiers
    repo = os.path.dirname(os.path.dirname(os.path.dirname(common.__file__)))
    with open(os.path.join(repo, "pyproject.toml"), encoding="utf-8") as fh:
        pin = re.search(r'^eval = \[.*"silero-vad([^"]*)".*\]', fh.read(), re.M)
    assert pin, "the eval extra must pin silero-vad"
    installed = importlib.metadata.version("silero-vad")
    assert specifiers.SpecifierSet(pin.group(1)).contains(installed), (
        f"silero-vad {installed} is installed; the cohort was cut with"
        f" {pin.group(1)}")


def test_alignment_of_a_clip_shorter_than_the_lag_window_reads_its_lag():
    """A clip of at most 2048 samples read a lag of -4800 and aligned to
    nothing; the lag window is capped at the clip's length."""
    ref = white(2000, seed=3)

    for lag in (0, 37):
        out = delayed(ref, lag)
        got, sign = common.best_alignment(out, ref)
        aligned = common.apply_lag(out, got, len(ref))

        assert (got, sign) == (lag, 1)
        assert torch.allclose(aligned[:len(ref) - lag],
                              ref[:len(ref) - lag],
                              atol=1e-6)


def test_the_private_results_guard_sees_through_a_symlinked_checkout(
        tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    (repo / "bench" / "results").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(repo)
    monkeypatch.setattr(common, "REPO_DIR", str(link))
    monkeypatch.setattr(common, "RESULTS", str(repo / "bench" / "results"))
    monkeypatch.setattr(common, "OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(common, "is_sealed", lambda rows, root=None: True)

    with pytest.raises(ValueError, match="private"):
        common.guard_private_results([])
