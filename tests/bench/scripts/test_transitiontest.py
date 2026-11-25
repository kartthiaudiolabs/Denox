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
"""Transition check: stimulus geometry, recovery counting, and fakes.

Each fake has a known dip and recovery.
"""

# pylint: disable=redefined-outer-name,unused-argument

import json
import math

import pytest
import soundfile as sf
import torch

import common
import transitiontest

SR = common.SR
FRAME = transitiontest.FRAME


def speech(seconds, seed=0, level=0.2, quiet_ends=0.5):
    """Speech-like noise with silent ends.

    The silent ends keep the speech centre well inside the clip.
    """
    n = int(seconds * SR)
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    env = 0.3 + 0.7 * (0.5 + 0.5 * torch.sin(2 * math.pi * 2.0 * t)).pow(2)
    x = torch.randn(n, generator=g) * env * level
    q = int(quiet_ends * SR)
    x[:q] = 0
    x[-q:] = 0
    return x


def bed_for(ref, snr_db, seed=99):
    g = torch.Generator().manual_seed(seed)
    b = torch.randn(len(ref), generator=g)
    active = ref[ref != 0]
    return (b * float(active.pow(2).mean().sqrt()) /
            float(b.pow(2).mean().sqrt()) * 10**(-snr_db / 20))


def test_speech_centre_is_the_middle_of_the_loud_span():
    ref = speech(6.0)  # speech in blocks 5..54: the centre is block 29
    assert transitiontest.speech_center_block(ref) == 29
    # nothing loud: the clip's middle
    assert transitiontest.speech_center_block(torch.zeros(3 * SR)) == 15


def test_build_stimulus_gates_the_bed_at_the_block_boundary():
    ref = speech(6.0)
    bed = bed_for(ref, 0.0)
    t, ramp = 30, 960
    on = transitiontest.build_stimulus(ref, bed, t, "onset", ramp=ramp)
    off = transitiontest.build_stimulus(ref, bed, t, "offset", ramp=ramp)
    edge = t * FRAME
    # nothing before the ramp
    assert torch.equal(on[:edge - ramp], ref[:edge - ramp])
    # the new state from the boundary on
    assert torch.allclose(on[edge:], (ref + bed)[edge:])
    assert torch.allclose(off[:edge - ramp], (ref + bed)[:edge - ramp])
    assert torch.equal(off[edge:], ref[edge:])
    # the two gates are complementary
    assert torch.allclose(on + off, 2 * ref + bed)
    with pytest.raises(ValueError):
        transitiontest.build_stimulus(ref, bed, t, "sideways")


def test_recovery_counts_blocks_until_the_level_holds():
    kept = torch.tensor([0.0] * 10 + [-10.0] * 5 + [0.0] * 20)
    loud = torch.ones(len(kept), dtype=torch.bool)
    assert transitiontest.recovery_blocks(kept, loud, 10, 0.0, 3.0) == 5
    # inside the tolerance from the start
    assert transitiontest.recovery_blocks(kept, loud, 10, 0.0, 12.0) == 0
    never = torch.tensor([0.0] * 10 + [-10.0] * 25)
    assert math.isnan(transitiontest.recovery_blocks(never, loud, 10, 0.0, 3.0))
    flicker = torch.tensor([0.0] * 10 + [-10.0, 0.0, -10.0, 0.0, 0.0, 0.0] +
                           [0.0] * 10)
    loud_cut = loud[:len(flicker)]
    # one good block does not count
    assert transitiontest.recovery_blocks(flicker, loud_cut, 10, 0.0, 3.0) == 3
    quiet = loud.clone()
    quiet[12:15] = False  # quiet blocks neither help nor hurt
    assert transitiontest.recovery_blocks(kept, quiet, 10, 0.0, 3.0) == 5


def test_oracle_and_identity_read_zero_everywhere():
    ref = speech(8.0, seed=1)
    bed = bed_for(ref, 0.0)
    mix = ref + bed
    # the oracle, and the unprocessed input
    for run in (lambda s: ref, lambda s: s):
        r = transitiontest.measure_item(run, mix, ref)
        for kind in ("onset", "offset"):
            k = r[kind]
            # the ramp block only
            assert abs(k["dip_db"]) < 0.3 and abs(k["pre_db"]) < 0.3
            assert k["recovery_ms"] == 0.0 and abs(k["leak_db"]) < 0.5
            assert abs(k["steady_delta_db"]) < 0.01
    ident = transitiontest.measure_item(lambda s: s, mix, ref)
    # the bed itself
    assert ident["onset"]["resid_new_state_db"] == pytest.approx(0.0, abs=2.5)
    assert ident["offset"]["resid_new_state_db"] < -70  # nothing but the voice
    assert abs(ident["onset"]["kept_new_state_db"]) < 0.5


def _bed_changes_at(stim, ref, t, frame=FRAME):
    """A fake's ear.

    Does the bed under the reference differ before and after block ``t``?
    """
    d = stim - ref
    e0 = float(d[max(0, (t - 10) * frame):t * frame].pow(2).mean())
    e1 = float(d[t * frame:(t + 10) * frame].pow(2).mean())
    return abs(10 * math.log10((e1 + 1e-12) / (e0 + 1e-12))) > 20


def test_a_slow_adapter_shows_its_dip_and_recovery_time():
    ref = speech(8.0, seed=2)
    bed = bed_for(ref, 0.0)
    mix = ref + bed
    t = transitiontest.speech_center_block(ref)
    # -10.5 dB for the 500 ms after a change, then the voice again
    gain, hold_blocks = 0.3, 5

    def adapter(stim):
        y = ref.clone()
        if _bed_changes_at(stim, ref, t):
            y[t * FRAME:(t + hold_blocks) * FRAME] *= gain
        return y

    r = transitiontest.measure_item(adapter, mix, ref, t_block=t)
    for kind in ("onset", "offset"):
        assert r[kind]["dip_db"] == pytest.approx(20 * math.log10(gain),
                                                  abs=0.1)
        assert r[kind]["recovery_ms"] == pytest.approx(hold_blocks * 100)
        assert (abs(r[kind]["pre_db"]) < 0.01 and
                abs(r[kind]["steady_delta_db"]) < 0.01)


def test_a_reaction_that_never_settles_is_a_steady_delta_without_recovery():
    ref = speech(8.0, seed=3)
    mix = ref + bed_for(ref, 0.0)
    t = transitiontest.speech_center_block(ref)

    def halves_after_a_change(stim):
        y = ref.clone()
        if _bed_changes_at(stim, ref, t):
            y[t * FRAME:] *= 0.5
        return y

    def onset(run):
        return transitiontest.measure_item(run, mix, ref, t_block=t)["onset"]

    r = onset(halves_after_a_change)
    assert r["dip_db"] == pytest.approx(-6.02, abs=0.05)
    assert r["steady_delta_db"] == pytest.approx(-6.02, abs=0.05)
    assert math.isnan(r["recovery_ms"])
    # the same halving
    steady = onset(lambda s: ref * 0.5)
    # on every input
    assert (abs(steady["dip_db"]) < 0.01 and
            abs(steady["steady_delta_db"]) < 0.01)


def test_a_model_that_reacts_before_the_change_shows_a_pre_echo():
    ref = speech(8.0, seed=4)
    mix = ref + bed_for(ref, 0.0)
    t = transitiontest.speech_center_block(ref)

    def anticipates(stim):
        y = ref.clone()
        if _bed_changes_at(stim, ref, t):
            y[(t - 3) * FRAME:t * FRAME] *= 0.1  # 300 ms of look-ahead damage
        return y

    r = transitiontest.measure_item(anticipates, mix, ref, t_block=t)["offset"]
    assert r["pre_db"] == pytest.approx(-20.0, abs=0.1)
    assert abs(r["dip_db"]) < 0.01 and r["recovery_ms"] == 0.0


def test_aggregate_reports_means_worst_and_never_recovered():
    a = {
        "onset": {
            k: 0.0 for k in transitiontest.RESPONSE_KEYS
        },
        "offset": {
            k: 0.0 for k in transitiontest.RESPONSE_KEYS
        },
    }
    assert "dip_control_db" not in transitiontest.RESPONSE_KEYS
    b = {
        "onset":
            dict({k: -2.0 for k in transitiontest.RESPONSE_KEYS},
                 recovery_ms=float("nan")),
        "offset":
            dict({k: -4.0 for k in transitiontest.RESPONSE_KEYS},
                 recovery_ms=800.0),
    }
    s = transitiontest.aggregate([a, b])
    assert s["n_items"] == 2
    assert s["onset"]["dip_db"] == -1.0 and s["onset"]["dip_worst_db"] == -2.0
    assert (s["onset"]["n_never_recovered"] == 1 and
            s["onset"]["recovery_ms"] == 0.0)
    assert (s["offset"]["recovery_max_ms"] == 800.0 and
            s["offset"]["n_never_recovered"] == 0)


@pytest.fixture
def paired_set(tmp_path, monkeypatch):
    root = tmp_path / "set"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    rows = []
    for i in range(2):
        ref = speech(5.0, seed=30 + i)
        mix = ref + bed_for(ref, 5.0, seed=40 + i)
        rid = f"ambienceATp0_{i:03d}.wav"
        sf.write(str(root / "mix" / rid), mix.numpy(), SR, subtype="FLOAT")
        sf.write(str(root / "ref" / rid), ref.numpy(), SR, subtype="FLOAT")
        rows.append({
            "id": rid,
            "condition": "ambience@+0",
            "mix_path": f"mix/{rid}",
            "clean_path": f"ref/{rid}",
        })
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump({
        "set_id": "tiny",
        "n_items": 2
    }, open(root / "checksums.json", "w", encoding="utf-8"))
    monkeypatch.setattr(common, "EVAL", str(root))
    return root


def test_main_runs_both_stimuli_through_the_tiny_model(identity_model,
                                                       paired_set, tmp_path,
                                                       monkeypatch):
    # pylint: disable-next=import-outside-toplevel
    from inworld.voice_isolation import isolator

    vi = isolator.VoiceIsolator(identity_model, device="cpu")
    monkeypatch.setattr(
        common,
        "isolator_runner",
        lambda a: (vi, lambda x, c=None, o=None: vi.isolate(x, SR)),
    )
    out = tmp_path / "transitions.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "transitiontest.py",
            "--model",
            "tiny",
            "--conds",
            "ambience@+0",
            "--out",
            str(out),
        ],
    )
    transitiontest.main()
    r = json.load(open(out, encoding="utf-8"))
    assert r["set_id"] == "tiny" and r["ramp_ms"] == 20.0
    s = r["conditions"]["ambience@+0"]["summary"]
    assert s["n_items"] == 2
    # identity
    assert (abs(s["onset"]["dip_db"]) < 0.5 and
            s["onset"]["n_never_recovered"] == 0)
    assert r["conditions"]["ambience@+0"]["items"][0]["onset"]["t_s"] > 0
