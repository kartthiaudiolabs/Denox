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
"""The control defects are what they claim to be, on a fixed slice.

Every control is held to a declared expectation.
"""

# pylint: disable=protected-access

import math

import torch

import anchors
import common
import controls
import ranking

SR = controls.SR


def _voice(seconds=3.0, seed=0):
    """Returns noise bursts with silence between them and a quiet floor.

    Enough structure for a defect to land on "speech" and for a floor to be
    removable.
    """
    gen = torch.Generator().manual_seed(seed)
    n = int(seconds * SR)
    x = 1e-4 * torch.randn(n, generator=gen)
    for start in range(int(0.2 * SR), n - SR // 2, SR):
        burst = 0.3 * torch.randn(SR // 2, generator=gen)
        x[start:start + SR // 2] += burst
    return x


def test_a_deletion_removes_exactly_its_span_from_speech_and_is_frozen_per_item(
):
    ref = _voice()
    for seconds in (0.020, 0.080, 0.320):
        out = controls.delete(ref, seconds, "item_a.wav")
        gone = torch.nonzero(out == 0).flatten()
        # the span less its edges
        assert abs(len(gone) - seconds * SR) <= 2 * int(0.002 * SR)
        # it fell on a burst, not on the floor
        assert float(ref[gone].pow(2).mean()) > 1e-3
        assert torch.equal(out, controls.delete(ref, seconds, "item_a.wav"))
        untouched = torch.ones(len(ref), dtype=torch.bool)
        untouched[int(gone.min()) - 200:int(gone.max()) + 200] = False
        assert torch.equal(out[untouched], ref[untouched])
    assert not torch.equal(controls.delete(ref, 0.080, "item_a.wav"),
                           controls.delete(ref, 0.080, "item_b.wav"))
    short = ref[:1000]
    # nothing to delete from: unchanged
    assert torch.equal(controls.delete(short, 0.320, "x"), short)


def test_the_invented_high_band_keeps_the_energy_and_loses_the_waveform():
    ref = _voice()
    out = controls.invent_high_band(ref, "item.wav")

    def high(x):
        return controls.cut_high_band(x) - x

    # below 8 kHz: the recording
    assert common.si_sdr(controls.cut_high_band(out),
                         controls.cut_high_band(ref)) > 30
    ratio = 10 * math.log10(
        float(high(out).pow(2).mean() / high(ref).pow(2).mean()))
    assert abs(ratio) < 1.0  # above: the same energy
    assert common.si_sdr(high(out), high(ref)) < 0  # and another signal
    cut = controls.cut_high_band(ref)
    assert float(
        high(cut).pow(2).mean()) < 1e-3 * float(high(ref).pow(2).mean())


def test_resynthesis_is_another_waveform_with_the_same_spectrogram():
    ref = _voice()
    out = controls.resynthesise(ref, "item.wav")
    assert 15 < common.si_sdr(out, ref) < 30

    def spec(x):
        return controls._stft(x).abs()

    err = float((spec(out) - spec(ref)).pow(2).sum() / spec(ref).pow(2).sum())
    assert err < 0.02
    assert torch.equal(out, controls.resynthesise(ref, "item.wav"))


def test_floor_removal_silences_the_gaps_and_leaves_the_speech():
    ref = _voice()
    out = controls.remove_floor(ref)
    burst = slice(int(0.25 * SR), int(0.6 * SR))
    gap = slice(int(0.8 * SR), int(1.1 * SR))
    assert torch.equal(out[burst], ref[burst])
    assert (float(out[gap].abs().max()) == 0.0 and
            float(ref[gap].abs().max()) > 0.0)


def test_delay_gain_and_polarity_are_the_named_transform():
    ref = _voice(1.0)
    late = controls.delay(ref, 0.020)
    assert (len(late) == len(ref) and torch.equal(late[960:], ref[:-960]) and
            not late[:960].any())

    def build(name):
        return controls.CONTROLS[name][1](ref, 2 * ref, "i")

    assert torch.allclose(build("gainup6db"), ref * 1.9953, atol=1e-4)
    assert torch.equal(build("polarity"), -ref) and torch.equal(
        build("bedonly"), ref)
    assert torch.equal(build("input"), 2 * ref) and not build("silence").any()


def test_the_slice_is_fixed_and_heard_across_languages_before_any_twice():
    rows = [{
        "id": f"c_{lang}_{i}.wav",
        "condition": "clean",
        "language": lang,
        "clean_path": "r",
    }
            for lang, count in (("en", 6), ("de", 2), ("ja", 1))
            for i in range(count)]
    rows += [{
        "id": "real.wav",
        "condition": "clean",
        "language": "en"
    }]  # no reference: never picked
    picked = controls.stratified(list(reversed(rows)), ("clean", "absent"), 5)
    assert [r["language"] for r in picked] == ["de", "en", "ja", "de", "en"]
    assert picked == controls.stratified(rows, ("clean", "absent"), 5)
    assert len(controls.stratified(rows, ("clean",), 50)) == 9


def test_every_control_is_held_to_an_expectation_over_known_columns():
    assert set(controls.EXPECTED) == set(controls.CONTROLS)
    for expectation in controls.EXPECTED.values():
        assert expectation and set(expectation) <= set(controls.COLUMNS)
        assert all(op in (">=", "<=") for op, _ in expectation.values())
    for harmless in (
            "reference",
            "polarity",
            "delay20ms",
            "resynth",
            "floorless",
    ):
        # inaudible: never read as damage
        assert controls.EXPECTED[harmless]["fidelity"] == (">=", 0.95)
    # stated, not hidden
    assert (set(controls.BLIND) <= set(controls.CONTROLS) and
            all(controls.BLIND.values()))


def test_check_reports_the_expectations_a_table_misses():
    good = {
        "fidelity": 1.0,
        "kept_coh_p10": 0.0,
        "lost_pct": 0.0,
        "nv_lost_pct": 0.0,
        "spk_sim": 1.0,
        "si_sdr": 100.0,
    }
    table = {
        "reference": good,
        "polarity": dict(good, kept_coh_p10=-80.0),
        "delete320ms": {
            "lost_pct": float("nan"),
            "fidelity": 0.5
        },
    }
    # a slice without the block holds no control to its columns
    failures = controls.check(table, block=False)
    assert set(failures) == {"polarity", "delete320ms"}
    held = controls.check(table)
    assert "event_resid_db nan" in held["reference"][0]
    assert failures["polarity"] == ["kept_coh_p10 -80.00 should be >= -1"]
    # unmeasured is not passed
    assert "lost_pct nan" in failures["delete320ms"][0]
    md = controls.markdown(
        {
            k: {
                c: v.get(c, float("nan")) for c in controls.COLUMNS
            } for k, v in table.items()
        },
        failures,
        "0123456789abcdef0123",
        240,
        ("clean",),
    )
    assert ("| `reference` |" in md and "| yes |" in md and
            "no: kept_coh_p10 -80.00" in md)
    assert "## Blind spots" in md and "- `hfinvented`:" in md


def test_summarise_reads_the_published_fidelity_axis_on_the_clean_items():
    policy = anchors.definitions()
    points = {}
    for cond, metric, sign in ranking.AXES_FULL["fidelity"]:
        rule = anchors.rule_for(policy, cond, metric)
        points[anchors.row_key(cond, metric)] = {
            "floor": rule["floor"]["value"],
            "ceiling": rule["ceiling"]["value"],
            "direction": sign,
        }
    clean = {
        "si_sdr": 99.0,
        "kept_coh_p10": -6.0,
        "kept_coh_lost_pct": 2.5,
        "spk_sim": 1.0,
        "nisqa_mos": 4.5,
        "out_in_rms_db": -6.0,
        "level_err_db": 6.0,  # the scorer's mean absolute level change
        "band_kept_db": [0, 0, 0, 0, 0.1, -0.1, 0.1],
        "hf_err_db": 0.1,  # and its high-band error, the gain undone
        "mag_snr": 20.0,
    }
    metrics = {
        "clean": clean,
        "nonverbal_clean": {
            "nv_lost_pct": 15.0
        },
        "_level_cal_db": -6.0,
        "babble@+4": {
            "babble_supp_db": -12.0
        },
    }
    row = controls.summarise(metrics, {"points": points}, ranking.AXES_FULL)
    assert row["lost_pct"] == 2.5
    # a gain is level, not deletion
    assert row["kept_coh_p10"] == 0.0 and row["level_err"] == 6.0
    assert (math.isclose(row["hf_err"], 0.1, abs_tol=1e-9) and
            row["babble_supp_db"] == -12.0)
    assert row["mag_snr"] == 20.0 and row["nv_lost_pct"] == 15.0
    # the block's rows, read from its pooled cell when the slice holds it
    assert all(math.isnan(row[c]) for c in controls.BLOCK_COLUMNS)
    metrics["_coroom"] = {
        "event_resid_db": -20.0,
        "floor_resid_db": -6.0,
        "voice_level_db": -0.3,
    }
    pooled = controls.summarise(metrics, {"points": points}, ranking.AXES_FULL)
    assert (pooled["event_resid_db"], pooled["floor_resid_db"],
            pooled["voice_level_db"]) == (-20.0, -6.0, -0.3)
    assert math.isclose(row["fidelity"], (1 + 0 + 1 + 1 + 1 + 0.5 + 0.5) / 7)
    # a slice without the recordings has no fidelity to state
    del metrics["nonverbal_clean"]
    sliced = controls.summarise(metrics, {"points": points}, ranking.AXES_FULL)
    assert math.isnan(sliced["fidelity"])
    bare = controls.summarise({"clean": {}}, {"points": points},
                              ranking.AXES_FULL)
    assert math.isnan(bare["fidelity"])


def test_the_block_controls_tell_the_transients_from_the_floor_and_duck():
    ref = _voice()
    n = len(ref)
    floor = torch.randn(
        n, generator=torch.Generator().manual_seed(3)) * 10**(-55 / 20)
    event = torch.zeros(n)
    a = int(1.2 * controls.SR)
    event[a:a + 4800] = torch.randn(
        4800, generator=torch.Generator().manual_seed(4)) * 10**(-15 / 20)
    mix = ref + floor + event
    kept = controls.transients(mix - ref)
    # the event's frames stay whole, the floor's frames are silenced
    assert torch.allclose(kept[a:a + 4800], (floor + event)[a:a + 4800],
                          atol=1e-6)
    assert float(kept[:a - 480].abs().max()) == 0.0
    assert torch.equal(controls.CONTROLS["transientsleft"][1](ref, mix, "i"),
                       ref + kept)
    assert torch.equal(controls.CONTROLS["floorleft"][1](ref, mix, "i"),
                       mix - kept)
    assert torch.allclose(controls.CONTROLS["halfnoise"][1](ref, mix, "i"),
                          ref + 0.5 * (floor + event),
                          atol=1e-6)
    # the voice ducks only where something was added
    assert torch.allclose(controls.ducked(ref, mix), ref * 10**(-3 / 20))
    assert torch.equal(controls.ducked(ref, ref), ref)
    assert controls.BLOCK_CONDS == tuple(
        c for c in controls.DEFAULT_CONDS if c.startswith("coroom"))


def test_controls_have_a_description_and_a_builder_and_notes_name_controls():
    for name, entry in controls.CONTROLS.items():
        assert isinstance(entry, tuple) and len(entry) == 2, name
        assert isinstance(entry[0], str) and callable(entry[1]), name
    assert set(controls.EXPECTED) <= set(controls.CONTROLS) and set(
        controls.BLIND) <= set(controls.CONTROLS)


def test_cached_control_is_trusted_only_under_the_same_scorers_slice_and_policy(
    monkeypatch,):
    rows = [{"id": "a.wav"}, {"id": "b.wav"}]
    metrics = {"_controls_identity": controls.identity(rows), "clean": {}}
    assert controls.cached(metrics, rows)
    assert not controls.cached(metrics, rows[:1])  # another slice
    # saved before the identity existed
    assert not controls.cached({"clean": {}}, rows)
    # pylint: disable-next=import-outside-toplevel
    import scorekept

    monkeypatch.setattr(scorekept, "SCORER_SHA256", "another sidecar")
    assert not controls.cached(metrics, rows)  # a scorer changed under it
    monkeypatch.undo()
    # pylint: disable-next=import-outside-toplevel
    import scoreevents

    monkeypatch.setattr(scoreevents, "SCORER_SHA256", "another sidecar")
    assert not controls.cached(metrics, rows)
    monkeypatch.undo()
    # pylint: disable-next=import-outside-toplevel
    import observations

    digest = controls.identity(rows)["controls_sha256"]
    assert digest == observations.implementation(controls.__file__)
    # a corrected label of a same-room item makes the saved control stale
    block = [{
        "id": "coroomATboth_000.wav",
        "condition": "coroom@both",
        "provenance": {
            "events": [[1.0, 1.1, "speech", "household", "src:a", -23.0]],
            "room_share_db": 0.8
        }
    }]
    saved = {"_controls_identity": controls.identity(block), "clean": {}}
    assert controls.cached(saved, block)
    moved = [
        dict(block[0],
             provenance=dict(block[0]["provenance"], room_share_db=1.2))
    ]
    assert not controls.cached(saved, moved)


def _bursts():
    """Half a second of silence, then speech-like bursts at known places."""
    ref = torch.zeros(int(3.2 * controls.SR))
    for start_s, end_s in ((0.5, 1.5), (1.9, 2.7)):
        a, b = int(start_s * controls.SR), int(end_s * controls.SR)
        t = torch.arange(b - a) / controls.SR
        ref[a:b] = 0.3 * torch.sin(2 * math.pi * 220 * t)
    return ref


def test_boundaries_find_the_word_starts_and_trailing_ends():
    onsets, offsets = controls.boundaries(_bursts())
    frame = controls.SR // 100
    assert [round(o / frame) for o in onsets] == [50, 190]
    assert [round(o / frame) for o in offsets] == [150, 270]


def test_onset_and_tail_cuts_sit_on_a_boundary_and_remove_only_their_span():
    ref = _bursts()
    onsets, offsets = controls.boundaries(ref)
    span = int(0.08 * controls.SR)
    edge = int(0.002 * controls.SR)
    cut, start = controls.cut_onset(ref, 0.08, "item")
    assert start in onsets
    assert cut[start + edge:start + span - edge].abs().max() == 0
    assert torch.equal(cut[start + span:], ref[start + span:])
    assert torch.equal(cut[:start], ref[:start])
    assert controls.cut_onset(ref, 0.08, "item")[1] == start  # frozen per item
    cut, start = controls.cut_tail(ref, 0.16, "item")
    span = int(0.16 * controls.SR)
    assert start + span in offsets
    assert cut[start + edge:start + span - edge].abs().max() == 0
    assert torch.equal(cut[start + span:], ref[start + span:])
    # nothing fits: the reference comes back unchanged, with no position
    same, none = controls.cut_onset(ref, 5.0, "item")
    assert none is None and torch.equal(same, ref)
