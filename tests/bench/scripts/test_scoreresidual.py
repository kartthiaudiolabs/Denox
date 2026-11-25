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
"""The residual sidecar: known damage in, what its two rows say out."""

import json
import math

import numpy as np
import pytest
import torch

import anchors
import common
import observations
import scorebench
import scoreresidual

SR = common.SR
BANDS = len(scoreresidual.CENTRES_HZ)


class _Vad:
    """Speech from 1.0 to 2.0 s and from 2.6 to 3.0 s of a 4 s item."""

    def segments(self, x):
        del x
        return None, [{
            "start": 16000,
            "end": 32000
        }, {
            "start": 41600,
            "end": 48000
        }]


def _item(seed=0, floor_db=-40.0, voice_db=-10.0, seconds=4.0):
    """A voice burst over a steady floor, both white."""
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * SR)
    floor = torch.randn(n, generator=g) * 10**(floor_db / 20)
    voice = torch.zeros(n)
    for start, end in ((1.0, 2.0), (2.6, 3.0)):
        a, b = int(start * SR), min(int(end * SR), n)
        if b > a:
            voice[a:b] = torch.randn(b - a, generator=g) * 10**(voice_db / 20)
    return floor + voice


def _pause_scaled(x, db, vad=_Vad()):
    """``x`` with everything outside the widened speech lowered by ``db``."""
    y = x.clone()
    _, segs = vad.segments(x)
    keep = torch.zeros(len(x), dtype=torch.bool)
    for s in segs:
        a = int((s["start"] / 16000 - scoreresidual.PRE_S) * SR)
        b = int((s["end"] / 16000 + scoreresidual.POST_S) * SR)
        keep[max(a, 0):b] = True
    y[~keep] *= 10**(-db / 20)
    return y


ROW = {"id": "real_x_000.wav", "condition": "real_x", "mix_path": "unused"}


def _rows(out, mix):
    return scoreresidual.pause_metrics(
        scoreresidual.score_real(ROW, out, mix, _Vad()))


def test_weighting_follows_the_468_table_normalised_at_1_khz():
    assert scoreresidual.weighting(1000.0) == pytest.approx(1.0)
    for hz, db in ((100.0, -19.8), (6300.0, 12.2), (10000.0, 8.1), (20000.0,
                                                                    -22.2)):
        assert 20 * math.log10(scoreresidual.weighting(hz)) == pytest.approx(
            db, abs=0.2)
    assert BANDS == 27
    assert scoreresidual.NB_BANDS == 22
    # every bin between the lowest and highest band edge belongs to a band
    freqs = np.arange(scoreresidual.N_FFT // 2 + 1) * SR / scoreresidual.N_FFT
    inside = (freqs >= 50 * 2**(-1 / 6)) & (freqs < 20000 * 2**(1 / 6))
    assert (scoreresidual.BAND_OF_BIN[inside] >= 0).all()
    # the 62.5 Hz band holds no bin at this resolution: it is not read
    assert not scoreresidual.SUPPORTED[1] and scoreresidual.SUPPORTED[2:].all()


def test_the_unchanged_input_reads_zero_and_known_reductions_read_true():
    x = _item()
    identity = _rows(x, x)
    assert identity["pause_resid_db"] == pytest.approx(0.0, abs=1e-6)
    assert identity["pause_resid_db_nb"] == pytest.approx(0.0, abs=1e-6)
    assert identity["pause_gain_db"] == pytest.approx(0.0, abs=1e-6)
    for db in (3.0, 10.0):
        rows = _rows(_pause_scaled(x, db), x)
        assert rows["pause_resid_db"] == pytest.approx(-db, abs=0.3)
        assert rows["pause_resid_db_nb"] == pytest.approx(-db, abs=0.3)
    # the residual against the voice: the floor sits about 30 dB under the
    # voice's loudest frames before anything is removed
    assert -36 < identity["pause_level_db"] < -24
    # and saturates at the clip once inaudible, as the reduction saturates at
    # the item's bound: a pause silenced outright earns no more than 20 dB
    silent = _rows(_pause_scaled(x, 60.0), x)
    assert silent["pause_level_db"] == scoreresidual.LEVEL_CLIP_DB
    assert silent["pause_resid_db"] == -scoreresidual.ITEM_CLIP_DB
    assert _rows(_pause_scaled(x, 20.0),
                 x)["pause_resid_db"] == pytest.approx(-20.0, abs=0.3)


def test_the_items_own_gain_is_undone_and_turning_it_down_earns_nothing():
    x = _item()
    # a louder or quieter copy reads the same pauses
    assert _rows(x * 2.0, x)["pause_resid_db"] == pytest.approx(0.0, abs=1e-6)
    assert _rows(x * 0.5, x)["pause_resid_db"] == pytest.approx(0.0, abs=1e-6)
    assert _rows(x * 0.5, x)["pause_gain_db"] == pytest.approx(-6.02, abs=0.01)
    # the loophole the review measured: this recording turned down 6 dB while
    # the clean items stay as they are (so the system's gain reads 0 dB) buys
    # nothing here, whatever that gain says
    assert _rows(x * 10**(-6 / 20),
                 x)["pause_resid_db"] == pytest.approx(0.0, abs=1e-6)
    # the voice turned down 6 dB and the pauses 12 dB: 6 dB of cleanup
    both = _pause_scaled(x * 10**(-6 / 20), 6.0)
    assert _rows(both, x)["pause_resid_db"] == pytest.approx(-6.0, abs=0.3)
    # delay and polarity are undone too
    delayed = torch.cat([torch.zeros(960), x[:-960]])
    assert _rows(delayed, x)["pause_resid_db"] == pytest.approx(0.0, abs=0.2)
    assert _rows(-x, x)["pause_resid_db"] == pytest.approx(0.0, abs=1e-6)


def test_deleting_speech_or_its_edges_earns_no_pause_credit():
    x = _item()
    # the words gone, the pauses untouched: the voice is gone with them, so
    # the pauses are as loud as ever against it, the bound
    gated = x.clone()
    gated[SR:2 * SR] = 0.0
    gated[int(2.6 * SR):3 * SR] = 0.0
    assert _rows(gated, x)["pause_resid_db"] == scoreresidual.ITEM_CLIP_DB
    # the 200 ms after each word gone: inside the widened mask, not a pause
    tails = x.clone()
    tails[2 * SR:int(2.2 * SR)] = 0.0
    tails[3 * SR:int(3.2 * SR)] = 0.0
    assert _rows(tails, x)["pause_resid_db"] == pytest.approx(0.0, abs=1e-6)
    # the item's first and last 300 ms are nobody's: silencing them reads
    # nothing either
    edges = x.clone()
    edges[:int(0.3 * SR)] = 0.0
    edges[-int(0.3 * SR):] = 0.0
    assert _rows(edges, x)["pause_resid_db"] == pytest.approx(0.0, abs=1e-6)


def test_hiss_counts_more_than_hum_of_the_same_energy():
    x = _item()
    t = torch.arange(len(x)) / SR
    hum = torch.sin(2 * math.pi * 100 * t) * 10**(-30 / 20)
    g = torch.Generator().manual_seed(7)
    hiss = torch.randn(len(x), generator=g) * 10**(-30 / 20)
    hiss = torch.from_numpy(
        np.fft.irfft(
            np.fft.rfft(hiss.numpy()) *
            (np.fft.rfftfreq(len(x), 1 / SR) > 4000), len(x))).float()
    hiss *= hum.norm() / hiss.norm()
    with_hum = _rows(x + hum, x)["pause_resid_db"]
    with_hiss = _rows(x + hiss, x)["pause_resid_db"]
    assert with_hiss > with_hum + 6.0


def test_an_item_without_enough_pause_or_an_old_record_is_not_read():
    x = _item(seconds=0.9)  # the edges off, under 0.4 s of pause is left
    measured = scoreresidual.score_real(ROW, x, x, _Vad())
    assert measured["pause_seconds"] < scoreresidual.MIN_PAUSE_S
    assert not scoreresidual.pause_metrics(measured)
    # a record without the output's voice energy (an earlier scorer's)
    older = dict(scoreresidual.score_real(ROW, _item(), _item(), _Vad()))
    older.pop("speech_out")
    assert not scoreresidual.pause_metrics(older)


def _measurement(out_db, gain_db=0.0, voice_db=30.0, bands=None):
    """A real item's measurement: flat pauses ``out_db`` under the input's.

    The whole output, voice included, sits ``gain_db`` from the input; the
    input's voice is ``voice_db`` over its pauses.
    """
    out = np.ones(BANDS) if bands is None else np.array(bands, dtype=float)
    return {
        "pause_seconds": 1.0,
        "speech_seconds": 1.0,
        "pause_frames": 100,
        "loud_frames": 100,
        "pause_out": (out * 10**((out_db + gain_db) / 10)).tolist(),
        "pause_in": out.tolist(),
        "speech_in": (out * 10**(voice_db / 10)).tolist(),
        "speech_out": (out * 10**((voice_db + gain_db) / 10)).tolist(),
    }


def test_the_worst_band_skips_the_empty_band_and_the_bands_the_voice_lacks():
    # the review's construction: every populated band 20 dB under the voice,
    # the 62.5 Hz band structurally empty, which read 0 dB before
    rows = scoreresidual.pause_metrics(_measurement(-20.0, voice_db=0.0))
    assert rows["floor_db"] == pytest.approx(-20.0, abs=1e-6)
    assert rows["floor_worst_db"] == pytest.approx(-20.0, abs=1e-6)
    # a band the voice does not occupy does not set the worst band either
    bands = np.ones(BANDS)
    bands[-1] = 1e-6  # no voice at 20 kHz
    m = _measurement(-20.0, voice_db=0.0, bands=bands)
    m["pause_out"][-1] = 1.0  # the pause holds a full tone there
    assert scoreresidual.pause_metrics(m)["floor_worst_db"] == pytest.approx(
        -20.0, abs=1e-6)


def _energies(o, r, m):
    return [
        float((o * o).mean()),
        float((o * r).mean()),
        float((r * r).mean()),
        float(((m - r)**2).mean()),
    ]


def test_babble_residual_reads_interference_and_not_the_voices_level():
    g = torch.Generator().manual_seed(1)
    r = torch.randn(48000, generator=g)
    n = torch.randn(48000, generator=g) * 10**(-4 / 20)
    m = r + n
    # the unchanged mixture: every bit of the chatter, 0 dB
    assert scoreresidual.babble_metrics(_energies(m, r,
                                                  m)) == pytest.approx(0.0,
                                                                       abs=0.1)
    # the voice 3 dB down and the chatter gone: nothing left to charge, the
    # item's bound
    assert scoreresidual.babble_metrics(_energies(
        0.7 * r, r, m)) == -scoreresidual.ITEM_CLIP_DB
    # half the chatter left: -6 dB, whatever the voice's level
    half = r + 0.5 * n
    assert scoreresidual.babble_metrics(_energies(half, r,
                                                  m)) == pytest.approx(-6.0,
                                                                       abs=0.2)
    assert scoreresidual.babble_metrics(_energies(0.5 * half, r,
                                                  m)) == pytest.approx(-6.0,
                                                                       abs=0.2)
    # a deleted voice is charged, as SI-SDR charges it; silence reads the
    # bound, the worst value, never the floor
    gone = r.clone()
    gone[:24000] = 0.0
    assert scoreresidual.babble_metrics(_energies(gone, r, m)) > -3.0
    assert scoreresidual.babble_metrics(_energies(
        torch.zeros(48000), r, m)) == scoreresidual.ITEM_CLIP_DB
    assert scoreresidual.babble_metrics(_energies(
        -r, r, m)) == scoreresidual.ITEM_CLIP_DB


def test_attenuation_never_improves_the_chatter_row():
    # the review's counterexample: an output barely holding a tenth of the
    # voice read +23.9 dB, and 1 % quieter read the floor
    energies = [1.0, 0.1001, 1.0, 0.4]
    quieter = [0.99**2, 0.1001 * 0.99, 1.0, 0.4]
    assert scoreresidual.babble_metrics(energies) == scoreresidual.ITEM_CLIP_DB
    assert scoreresidual.babble_metrics(quieter) == scoreresidual.ITEM_CLIP_DB
    # and in general the row is the same at every output gain
    g = torch.Generator().manual_seed(2)
    r = torch.randn(48000, generator=g)
    n = torch.randn(48000, generator=g) * 10**(-4 / 20)
    out = r + 0.3 * n
    readings = [
        scoreresidual.babble_metrics(_energies(a * out, r, r + n))
        for a in (1.0, 0.5, 0.1, 0.01, 0.001)
    ]
    assert max(readings) - min(readings) < 1e-3


def test_item_contributions_are_bounded_before_pooling():
    # the review's construction: 13 of 121 items with their pauses silenced
    # and 108 untouched bought the whole axis; an item now gives at most the
    # bound, and the pool reads what it should
    rows = [{
        "id": str(i),
        "condition": "real_wild",
        "mix_path": "x"
    } for i in range(121)]
    records = [{
        "id": str(i),
        "output_status": "ok",
        "measurements": _measurement(-200.0 if i < 13 else 0.0),
    } for i in range(121)]
    pooled = scoreresidual.aggregate(records, rows,
                                     0.0)["_real_cleanup"]["pause_resid_db"]
    assert pooled == pytest.approx(-20.0 * 13 / 121, abs=1e-6)
    assert anchors.normalize(pooled, {
        "floor": 0.0,
        "ceiling": -20.0
    }) == pytest.approx(13 / 121, abs=1e-6)


def test_aggregate_pools_real_items_and_keeps_every_count():
    rows = [
        {
            "id": "a",
            "condition": "real_wild",
            "mix_path": "x"
        },
        {
            "id": "b",
            "condition": "real_wild",
            "mix_path": "x"
        },
        {
            "id": "c",
            "condition": "real_dirty",
            "mix_path": "x"
        },
        {
            "id": "e",
            "condition": "real_noisy",
            "mix_path": "x"
        },
        {
            "id": "d",
            "condition": "babble@+4",
            "mix_path": "x",
            "clean_path": "y"
        },
    ]

    def real(uid, out_db, gain_db=0.0, seconds=1.0):
        m = _measurement(out_db, gain_db)
        m["pause_seconds"] = seconds
        return {"id": uid, "output_status": "ok", "measurements": m}

    records = [
        real("a", -10.0),
        # a 6 dB louder output reads the same pauses
        real("b", -20.0, gain_db=6.0),
        real("c", -3.0, seconds=0.1),
        {
            "id": "e",
            "output_status": "missing"
        },
        {
            "id": "d",
            "output_status": "ok",
            "measurements": {
                "babble_energy": [[1.0, 1.0, 1.0, 0.5]],
                "babble_energy_nb": [[1.0, 1.0, 1.0, 0.5]],
            },
        },
    ]
    out = scoreresidual.aggregate(records, rows, 0.0)
    assert out["real_wild"]["pause_resid_db"] == pytest.approx(-15.0)
    assert out["real_wild"]["n"] == 2 and out["real_wild"]["n_missing"] == 0
    # too little pause on its only item: short, not missing, still counted
    assert out["real_dirty"] == {
        "n_expected": 1,
        "n_missing": 0,
        "n": 0,
        "n_short": 1
    }
    # an output the system did not return: missing, never a smaller mean
    assert out["real_noisy"] == {"n_expected": 1, "n_missing": 1, "n": 0}
    pooled = out["_real_cleanup"]
    assert pooled["pause_resid_db"] == pytest.approx(-15.0)
    assert (pooled["n"], pooled["n_expected"], pooled["n_missing"],
            pooled["n_short"]) == (2, 4, 1, 1)
    assert pooled["by_family"] == {"real_wild": pytest.approx(-15.0)}
    assert out["babble@+4"]["babble_resid_db"] == -scoreresidual.ITEM_CLIP_DB
    assert out["babble@+4"]["n"] == 1
    # the records carry the item's rows for a pairing
    assert records[1]["metrics"]["pause_resid_db"] == pytest.approx(-20.0)
    # the system's calibration is recorded, not applied
    assert scoreresidual.aggregate(records, rows, 6.0)["_level_cal_db"] == 6.0
    assert scoreresidual.aggregate(
        records, rows,
        6.0)["_real_cleanup"]["pause_resid_db"] == pytest.approx(-15.0)


def test_base_records_and_scores_must_be_this_render_and_this_scorer(
        tmp_path, monkeypatch):
    monkeypatch.setattr(common, "set_id", lambda root=None: "tiny")
    records = [{
        "id": "d",
        "condition": "babble@+4",
        "output_status": "ok",
        "identity": {
            "output": "o"
        },
        "measurements": {
            "babble_energy": [[1.0, 1.0, 1.0, 0.5]],
            "babble_energy_nb": [[1.0, 1.0, 1.0, 0.5]],
        },
    }]
    path = tmp_path / "half.items.jsonl"
    observations.sidecar(path,
                         records,
                         "half",
                         "tiny",
                         "all",
                         False,
                         scorebench.SCORER_SHA256,
                         render_identity={"outputs_sha256": "render-1"})
    held = scoreresidual.base_energies(path, {"d"}, "half", "render-1")
    assert held["d"][1] == [1.0, 1.0, 1.0, 0.5]
    with pytest.raises(SystemExit, match="another render"):
        scoreresidual.base_energies(path, {"d"}, "half", "render-2")
    with pytest.raises(SystemExit, match="another system"):
        scoreresidual.base_energies(path, {"d"}, "other", "render-1")
    observations.sidecar(path,
                         records,
                         "half",
                         "tiny",
                         "all",
                         False,
                         "an earlier scorer",
                         render_identity={"outputs_sha256": "render-1"})
    with pytest.raises(SystemExit, match="another scorer"):
        scoreresidual.base_energies(path, {"d"}, "half", "render-1")
    with pytest.raises(SystemExit, match="missing"):
        scoreresidual.base_energies(tmp_path / "none.jsonl", {"d"}, "half",
                                    "render-1")
    # the base scores the calibration is read from, likewise
    metrics = tmp_path / "half.metrics.json"
    metrics.write_text(
        json.dumps({
            "_level_cal_db": -0.5,
            "_scorer_sha256": scorebench.SCORER_SHA256,
            "_outputs_sha256": "render-1",
        }))
    assert scoreresidual.level_calibration(tmp_path, "half", "render-1") == -0.5
    with pytest.raises(SystemExit, match="another render"):
        scoreresidual.level_calibration(tmp_path, "half", "render-2")
    metrics.write_text(
        json.dumps({
            "_level_cal_db": -0.5,
            "_scorer_sha256": "older",
            "_outputs_sha256": "render-1",
        }))
    with pytest.raises(SystemExit, match="another base scorer"):
        scoreresidual.level_calibration(tmp_path, "half", "render-1")


def test_the_level_row_is_bounded_per_item_against_the_items_own_input():
    """The board's cleanup row gives an item at most ITEM_CLIP_DB of credit or
    penalty against its own input, then the audibility cap."""
    # the input's pauses 10 dB under its voice, silenced by the output
    silenced = scoreresidual.pause_metrics(_measurement(-200.0, voice_db=10.0))
    # a quiet recording whose pauses are already 30 dB down still saturates
    quiet = scoreresidual.pause_metrics(_measurement(-200.0, voice_db=30.0))
    # pauses made 60 dB louder lose at most the bound
    louder = scoreresidual.pause_metrics(_measurement(60.0, voice_db=10.0))

    assert silenced["pause_level_db"] == pytest.approx(
        -10.0 - scoreresidual.ITEM_CLIP_DB, abs=1e-6)
    assert quiet["pause_level_db"] == scoreresidual.LEVEL_CLIP_DB
    assert louder["pause_level_db"] == pytest.approx(-10.0 +
                                                     scoreresidual.ITEM_CLIP_DB,
                                                     abs=1e-6)


def test_the_pause_rows_pool_every_real_family_the_everyday_recordings_too():
    rows = [{
        "id": "e",
        "condition": "real_everyday",
        "mix_path": "x"
    }, {
        "id": "w",
        "condition": "real_wild",
        "mix_path": "x"
    }]
    records = [{
        "id": "e",
        "output_status": "ok",
        "measurements": _measurement(-10.0)
    }, {
        "id": "w",
        "output_status": "ok",
        "measurements": _measurement(0.0)
    }]

    pooled = scoreresidual.aggregate(records, rows, 0.0)["_real_cleanup"]

    assert "real_everyday" in scoreresidual.REAL
    assert pooled["n"] == 2
    assert pooled["pause_resid_db"] == pytest.approx(-5.0, abs=1e-6)


def test_items_without_speech_are_counted_apart_from_those_without_pause():
    rows = [{
        "id": i,
        "condition": "real_noisy",
        "mix_path": "x"
    } for i in ("short", "mute")]
    short = _measurement(0.0)
    short["pause_seconds"] = 0.1
    mute = _measurement(0.0)
    mute["loud_frames"] = 0
    records = [{
        "id": "short",
        "output_status": "ok",
        "measurements": short
    }, {
        "id": "mute",
        "output_status": "ok",
        "measurements": mute
    }]

    out = scoreresidual.aggregate(records, rows, 0.0)

    assert out["real_noisy"]["n_short"] == 1
    assert out["real_noisy"]["n_no_speech"] == 1
    assert out["_real_cleanup"]["n_short"] == 1
    assert out["_real_cleanup"]["n_no_speech"] == 1
