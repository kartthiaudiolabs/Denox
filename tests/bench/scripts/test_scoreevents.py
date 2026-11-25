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
"""The events sidecar: labelled events and floors in, what its rows say out."""

import json
import math
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

import common
import observations
import scorebench
import scoreevents

SR = common.SR


class _Vad:
    """Speech from 1.0 to 2.0 s and from 2.6 to 3.4 s of a 4 s item."""

    def segments(self, x):
        del x
        return None, [{
            "start": 16000,
            "end": 32000
        }, {
            "start": 41600,
            "end": 54400
        }]


def _burst(n, seed, db):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * 10**(db / 20)


def _same_room_item(seed=0, floor_db=-50.0):
    """A voice over a pause, an event in the pause and one under the voice."""
    n = 4 * SR
    ref = torch.zeros(n)
    for a, b in ((1.0, 2.0), (2.6, 3.4)):
        a, b = round(a * SR), round(b * SR)
        ref[a:b] = _burst(b - a, seed, -12.0)
    noise = _burst(n, seed + 1, floor_db)
    events = []
    for start, kind in ((2.2, "pause"), (1.4, "speech")):
        length = int(0.1 * SR)
        a = int(start * SR)
        noise[a:a + length] += _burst(length, seed + 2, -15.0)
        events.append([start, start + 0.1, kind, "household", "x", -15.0])
    row = {
        "id": "coroomATboth_000.wav",
        "condition": "coroom@both",
        "mix_path": "unused",
        "clean_path": "unused",
        "provenance": {
            "events": sorted(events),
            "pauses": [[2.0, 2.6]],
            "room_share_rule": scoreevents.ROOM_SHARE_RULE,
        },
    }
    return row, ref + noise, ref, noise


def _rows(it,
          out,
          mix,
          ref,
          out_ref=None,
          gain_db=0.0,
          lag=0,
          room_share_db=0.0):
    """The item's rows; the system renders the reference unchanged unless
    ``out_ref`` says otherwise."""
    out_ref = ref if out_ref is None else out_ref
    return scoreevents.same_room_metrics(
        scoreevents.score_same_room(it, out, mix, ref, out_ref, gain_db, lag,
                                    room_share_db), it["condition"])


def test_the_unchanged_mixture_reads_zero_and_the_reference_reads_the_bound():
    it, mix, ref, _ = _same_room_item()
    same = _rows(it, mix, mix, ref)
    assert same["event_resid_db"] == pytest.approx(0.0, abs=1e-6)
    assert same["floor_resid_db"] == pytest.approx(0.0, abs=1e-6)
    # the floor under the voice is -38 dB down: its cross term is below 0.01 dB
    assert same["voice_level_db"] == pytest.approx(0.0, abs=0.01)
    clean = _rows(it, ref, mix, ref)
    assert clean["event_resid_db"] == -scoreevents.ITEM_CLIP_DB
    assert clean["floor_resid_db"] == -scoreevents.ITEM_CLIP_DB
    assert clean["voice_level_db"] == pytest.approx(0.0, abs=1e-6)
    # the unbounded rows stand beside, floored where nothing is left
    assert same["event_resid_raw_db"] == pytest.approx(0.0, abs=1e-6)
    assert clean["event_resid_raw_db"] == scoreevents.RAW_FLOOR_DB
    assert clean["floor_resid_raw_db"] == scoreevents.RAW_FLOOR_DB


def test_a_known_reduction_of_the_added_noise_reads_true():
    it, mix, ref, noise = _same_room_item()
    for db in (6.0, 12.0):
        rows = _rows(it, ref + noise * 10**(-db / 20), mix, ref)
        assert rows["event_resid_db"] == pytest.approx(-db, abs=0.05)
        assert rows["floor_resid_db"] == pytest.approx(-db, abs=0.05)
        assert rows["voice_level_db"] == pytest.approx(0.0, abs=0.05)


def test_the_voice_turned_down_reads_as_lost_unless_it_is_the_systems_gain():
    it, mix, ref, _ = _same_room_item()
    # the voice ducked while the noise is there: the reference's render keeps
    # its level, the mixture's loses 6 dB
    quieter = _rows(it, mix * 10**(-6 / 20), mix, ref)
    assert quieter["voice_level_db"] == pytest.approx(-6.0, abs=0.1)
    # a system's gain applies to both renders and the base scores undo it
    gain = 10**(-6 / 20)
    undone = _rows(it, mix * gain, mix, ref, out_ref=ref * gain, gain_db=-6.0)
    assert undone["voice_level_db"] == pytest.approx(0.0, abs=0.01)
    assert undone["event_resid_db"] == pytest.approx(0.0, abs=1e-5)
    # the same gain unaccounted for reads as the noise reduced by it and,
    # against the reference, as the voice turned down: a render quieter on
    # the block than its base scores say is attenuation, not a gain
    kept = _rows(it, mix * gain, mix, ref, out_ref=ref * gain)
    # (the noise under the loudest frames adds a few hundredths of a dB)
    assert kept["voice_pair_db"] == pytest.approx(0.0, abs=0.1)
    assert kept["voice_level_db"] == pytest.approx(-6.0, abs=0.1)
    # and with the voice under the gate, nothing counts as removed
    assert kept["event_resid_db"] == 0.0 and kept["floor_resid_db"] == 0.0


def test_the_systems_delay_is_undone_and_its_polarity_cannot_matter():
    it, mix, ref, _ = _same_room_item()
    late = (-torch.roll(mix, 37), -torch.roll(ref, 37))
    exact = _rows(it, late[0], mix, ref, out_ref=late[1], lag=37)
    assert exact["event_resid_db"] == pytest.approx(0.0, abs=1e-6)
    assert exact["floor_resid_db"] == pytest.approx(0.0, abs=1e-6)
    # the floor's cross term on the loudest frames, as for the unchanged mixture
    assert exact["voice_level_db"] == pytest.approx(0.0, abs=0.01)
    # a delay not undone only moves the spans a little against the output
    rows = _rows(it, late[0], mix, ref, out_ref=late[1])
    assert rows["event_resid_db"] == pytest.approx(0.0, abs=0.2)
    assert rows["floor_resid_db"] == pytest.approx(0.0, abs=0.2)
    assert rows["voice_level_db"] == pytest.approx(0.0, abs=0.2)


def test_what_a_system_makes_of_the_room_is_charged_on_neither_row():
    """A system that removes the voice's room and leaves the noise reads the
    noise untouched: the room it removed is in both renders."""
    it, dry, _, noise = _same_room_item()
    tail = 0.5 * torch.roll(dry, 2400)  # 50 ms of room after the voice
    wet = dry + tail
    mix = wet + noise
    # the room's share of the reference's level on the frames the voice row
    # reads, as the set's provenance records it
    energies = scoreevents.score_same_room(it, dry, mix, wet, dry, 0.0, 0, 0.0)
    share = 10 * math.log10(energies["voice_ref"] / energies["voice"][1])
    paired = _rows(it, dry + noise, mix, wet, out_ref=dry, room_share_db=share)
    assert paired["event_resid_db"] == pytest.approx(0.0, abs=1e-6)
    assert paired["floor_resid_db"] == pytest.approx(0.0, abs=1e-6)
    assert paired["voice_level_db"] == pytest.approx(0.0, abs=0.05)
    # read against the reference alone, the room it removed would have
    # counted as noise left under the speech and in the pause
    against_ref = _rows(it, dry + noise, mix, wet, out_ref=wet)
    assert against_ref["event_resid_db"] > 1.0
    assert against_ref["floor_resid_db"] > 1.0


def test_each_event_is_bounded_before_the_item_reads():
    it, mix, ref, noise = _same_room_item()
    # one event removed outright, the other left as it was: half the credit
    gone = noise.clone()
    a = int(2.2 * SR)
    gone[a - 2 * scoreevents.FRAME:a + int((0.1 + scoreevents.DECAY_S) * SR) +
         2 * scoreevents.FRAME] = 0.0
    rows = _rows(it, ref + gone, mix, ref)
    assert rows["event_resid_db"] == pytest.approx(-scoreevents.ITEM_CLIP_DB /
                                                   2,
                                                   abs=0.05)
    assert rows["event_resid_raw_db"] == pytest.approx(
        scoreevents.RAW_FLOOR_DB / 2, abs=0.05)


def test_the_voice_row_is_the_lower_of_the_paired_and_the_anchored_read():
    it, mix, ref, _ = _same_room_item()
    # both renders 10 dB down: the noise reads as reduced by the scaling,
    # the voice against the reference as 10 dB down
    down = 10**(-10 / 20)
    both = _rows(it, mix * down, mix, ref, out_ref=ref * down)
    assert both["voice_pair_db"] == pytest.approx(0.0, abs=0.1)
    assert both["voice_kept_db"] == pytest.approx(-10.0, abs=0.1)
    assert both["voice_level_db"] == pytest.approx(-10.0, abs=0.1)
    # the room's share is allowed: a render quieter by just that much reads
    # as the voice kept
    allowed = _rows(it,
                    mix * down,
                    mix,
                    ref,
                    out_ref=ref * down,
                    room_share_db=10.0)
    assert allowed["voice_level_db"] == pytest.approx(0.0, abs=0.1)
    # a voice ducked with the noise there is read by the paired side,
    # whatever the allowance
    ducked = _rows(it, mix * down, mix, ref, room_share_db=10.0)
    assert ducked["voice_level_db"] == pytest.approx(-10.0, abs=0.1)
    # a render with no voice on its loudest frames reads the bound, not nothing
    silent = _rows(it,
                   torch.zeros_like(mix),
                   mix,
                   ref,
                   out_ref=torch.zeros_like(ref))
    assert silent["voice_level_db"] == -scoreevents.ITEM_CLIP_DB
    # and silence removes nothing: the noise rows read as left
    assert silent["event_resid_db"] == 0.0 and silent["floor_resid_db"] == 0.0


def test_speech_muted_in_both_renders_shows_against_the_reference():
    it, mix, ref, _ = _same_room_item()
    a = int(1.4 * SR)
    b = int(1.5 * SR) + int(scoreevents.DECAY_S * SR)
    out_mix, out_ref = ref.clone(), ref.clone()  # the noise removed...
    out_mix[a:b] = 0.0  # ...and the voice muted under the event, in both
    out_ref[a:b] = 0.0
    rows = _rows(it, out_mix, mix, ref, out_ref=out_ref)
    assert rows["voice_pair_db"] == pytest.approx(0.0, abs=0.01)
    assert rows["voice_level_db"] < -0.3


def test_removal_is_credited_only_while_the_voice_is_kept():
    it, mix, ref, noise = _same_room_item()
    # a mild loss keeps the credit, the declared floor takes it away
    mild = _rows(it, (ref + noise * 0.5) * 10**(-2 / 20),
                 mix,
                 ref,
                 out_ref=ref * 10**(-2 / 20))
    assert mild["voice_level_db"] == pytest.approx(-2.0, abs=0.1)
    assert mild["event_resid_db"] == pytest.approx(-6.0 - 2.0, abs=0.1)
    gone = _rows(it, (ref + noise * 0.5) * 10**(-4 / 20),
                 mix,
                 ref,
                 out_ref=ref * 10**(-4 / 20))
    assert gone["voice_level_db"] == pytest.approx(-4.0, abs=0.1)
    assert gone["event_resid_db"] == 0.0 and gone["floor_resid_db"] == 0.0
    # deleting the speech an event sits on, in both renders, is not removal:
    # that event reads as left while the event in the pause keeps its credit
    a, b = int(1.4 * SR), int(1.5 * SR) + int(scoreevents.DECAY_S * SR)
    out_mix, out_ref = ref.clone(), ref.clone()
    out_mix[a:b] = 0.0
    out_ref[a:b] = 0.0
    rows = _rows(it, out_mix, mix, ref, out_ref=out_ref)
    assert rows["event_resid_db"] == pytest.approx(-scoreevents.ITEM_CLIP_DB /
                                                   2,
                                                   abs=0.05)
    assert rows["event_resid_raw_db"] == pytest.approx(
        scoreevents.RAW_FLOOR_DB / 2, abs=0.05)


def _wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.numpy(), SR, subtype="FLOAT")
    return str(path)


def test_assemble_reads_the_pair_render_and_marks_one_that_is_missing(
        tmp_path, monkeypatch):
    it, mix, ref, noise = _same_room_item()
    it = dict(it,
              mix_path=_wav(tmp_path / "set" / "mix" / it["id"], mix),
              clean_path=_wav(tmp_path / "set" / "ref" / it["id"], ref))
    monkeypatch.setattr(common, "OUT_ROOT", str(tmp_path / "out"))
    _wav(tmp_path / "out" / "sys" / it["id"], ref + noise * 10**(-6 / 20))
    pair_out = tmp_path / "pair" / "out" / "sys"
    _wav(pair_out / it["id"], ref)
    records = scoreevents.assemble("sys", [it],
                                   None,
                                   0.0,
                                   pair_out=str(pair_out))
    assert records[0]["output_status"] == "ok"
    assert records[0]["identity"]["pair_output"]
    rows = scoreevents.same_room_metrics(
        observations.revive(records[0]["measurements"]), it["condition"])
    assert rows["event_resid_db"] == pytest.approx(-6.0, abs=0.05)
    assert scoreevents.pair_outputs_digest(records) == scoreevents.identity_of(
        [records[0]["identity"]["pair_output"]])
    # the record is reused only for the same pair of renders
    reuse = {
        scorebench.reuse_key(records[0]["identity"]):
            dict(records[0], gain_db=0.0)
    }
    again = scoreevents.assemble("sys", [it],
                                 None,
                                 0.0,
                                 reuse=reuse,
                                 pair_out=str(pair_out))
    assert again[0]["measurements"] == records[0]["measurements"]
    other_lag = scoreevents.assemble("sys", [it],
                                     None,
                                     0.0,
                                     reuse=reuse,
                                     pair_out=str(pair_out),
                                     lag=5)
    assert other_lag[0]["lag"] == 5
    assert other_lag[0]["measurements"] != records[0]["measurements"]
    _wav(pair_out / it["id"], ref * 0.5)
    fresh = scoreevents.assemble("sys", [it],
                                 None,
                                 0.0,
                                 reuse=reuse,
                                 pair_out=str(pair_out))
    assert fresh[0]["identity"]["pair_output"] != records[0]["identity"][
        "pair_output"]
    assert fresh[0]["measurements"] != records[0]["measurements"]
    missing = scoreevents.assemble("sys", [it],
                                   None,
                                   0.0,
                                   pair_out=str(tmp_path / "nowhere"))
    assert missing[0]["output_status"] == "missing_pair"
    assert scoreevents.pair_outputs_digest(missing) is None
    _wav(pair_out / it["id"], ref[:-480])
    short = scoreevents.assemble("sys", [it], None, 0.0, pair_out=str(pair_out))
    assert short[0]["output_status"] == "wrong_length_pair"
    # INPUT's pair render is the reference itself
    unchanged = scoreevents.assemble("INPUT", [it], None, 0.0)
    rows = scoreevents.same_room_metrics(
        observations.revive(unchanged[0]["measurements"]), it["condition"])
    assert rows["event_resid_db"] == pytest.approx(0.0, abs=1e-6)


def test_the_pair_links_the_references_and_its_renders_carry_the_board(
        tmp_path, monkeypatch):
    set_dir = tmp_path / "set"
    set_dir.mkdir()
    with open(set_dir / "checksums.json", "w", encoding="utf-8") as fh:
        json.dump({"set_id": "tiny"}, fh)
    monkeypatch.setattr(common, "EVAL", str(set_dir))
    it, mix, ref, _ = _same_room_item()
    rows = [
        dict(it,
             mix_path=_wav(set_dir / "mix" / it["id"], mix),
             clean_path=_wav(set_dir / "ref" / it["id"], ref)),
        _real_item()[0],
    ]
    root = tmp_path / "pair"
    assert scoreevents.pair_problem(str(root), rows) == "no probe.json"
    pair = scoreevents.write_pair(rows, str(root))
    assert pair["kind"] == scoreevents.PAIR_KIND and pair["set_id"] == "tiny"
    assert [e["id"] for e in pair["items"]] == [it["id"]]
    assert pair["identity"] == scoreevents.identity_of(
        [scorebench.sample_hash(ref)])
    assert scoreevents.pair_problem(str(root), rows) is None
    assert scoreevents.write_pair(rows, str(root)) == pair
    stray = root / "in" / "extra.wav"
    stray.write_bytes(b"")
    assert "not pair inputs" in scoreevents.pair_problem(str(root), rows)
    stray.unlink()
    with pytest.raises(SystemExit, match="link the pair again"):
        scoreevents.read_pair(str(tmp_path / "elsewhere"), rows)
    results = tmp_path / "results"
    results.mkdir()

    def problem():
        return scoreevents.pair_render_problem("sys", str(results), str(root),
                                               pair)

    assert "no stamp" in problem()
    with open(results / "sys.meta.json", "w", encoding="utf-8") as fh:
        json.dump({"render_environment_sha256": "env-board"}, fh)
    assert "no pair render" in problem()
    out = root / "out" / "sys"
    out.mkdir(parents=True)
    assert "no receipt" in problem()

    def receipt_of(environment, inputs):
        with open(out / ".receipt-1.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "environment_digest": environment,
                    "identity": {
                        "inputs": {
                            "probe": inputs
                        }
                    }
                }, fh)

    receipt_of("env-other", pair["identity"])
    assert "another environment" in problem()
    receipt_of("env-board", "other-inputs")
    assert "other inputs" in problem()
    receipt_of("env-board", pair["identity"])
    assert problem() is None


def test_only_the_conditions_with_a_layer_read_its_row():
    it, mix, ref, _ = _same_room_item()
    it_floor = dict(it, condition="coroom@floor")
    rows = _rows(it_floor, mix, mix, ref)
    assert "event_resid_db" not in rows and "floor_resid_db" in rows
    it_pause = dict(it, condition="coroom@pause")
    rows = _rows(it_pause, mix, mix, ref)
    assert "event_resid_db" in rows and "floor_resid_db" not in rows


def _real_item(seed=0, with_event=True):
    """A voice over a quiet floor with, by default, one impact in its pause."""
    n = 4 * SR
    x = _burst(n, seed, -55.0)
    for a, b in ((1.0, 2.0), (2.6, 3.4)):
        a, b = round(a * SR), round(b * SR)
        x[a:b] += _burst(b - a, seed + 1, -12.0)
    if with_event:
        a = int(2.25 * SR)
        x[a:a + int(0.05 * SR)] += _burst(int(0.05 * SR), seed + 2, -18.0)
    return {
        "id": f"real_x_{seed:03d}.wav",
        "condition": "real_everyday",
        "mix_path": "unused"
    }, x


def test_real_recordings_read_the_events_the_input_holds_outside_speech():
    it, x = _real_item()
    m = scoreevents.score_real(it, x, x, _Vad(), 0.0)
    assert len(m["events"]) == 1 and m["event_spans_s"][0][0] == pytest.approx(
        2.25, abs=0.02)
    same = scoreevents.real_metrics(m)
    assert same["n_events"] == 1
    assert same["event_att_db"] == pytest.approx(0.0, abs=1e-6)
    assert same["voice_level_db"] == pytest.approx(0.0, abs=1e-6)
    y = x.clone()
    a = int(2.2 * SR)
    y[a:a + int(0.3 * SR)] *= 10**(-10 / 20)
    reduced = scoreevents.real_metrics(
        scoreevents.score_real(it, y, x, _Vad(), 0.0))
    assert reduced["event_att_db"] == pytest.approx(-10.0, abs=0.3)
    quiet_it, quiet = _real_item(with_event=False)
    none = scoreevents.real_metrics(
        scoreevents.score_real(quiet_it, quiet, quiet, _Vad(), 0.0))
    assert none["n_events"] == 0 and "event_att_db" not in none


def test_the_detector_ignores_speech_onsets_and_the_spans_follow_the_definition(
):
    db = np.full(300, -60.0)
    db[100:110] = -30.0
    assert scoreevents.detect_events(db) == [(100, 110)]
    db[150] = -50.0
    assert scoreevents.detect_events(db) == [(100, 110)]


def _records(rows, out_by_id):
    records = []
    for it in rows:
        rec = observations.item(it)
        if it["id"] not in out_by_id:
            rec["output_status"] = "missing"
            records.append(rec)
            continue
        out, mix, ref = out_by_id[it["id"]]
        if ref is not None:
            rec["measurements"] = observations.finite(
                scoreevents.score_same_room(it, out, mix, ref, ref, 0.0))
        else:
            rec["measurements"] = observations.finite(
                scoreevents.score_real(it, out, mix, _Vad(), 0.0))
        records.append(rec)
    return records


def test_aggregate_pools_the_block_and_the_real_items_with_events():
    it, mix, ref, noise = _same_room_item()
    it_floor = dict(it, id="coroomATfloor_000.wav", condition="coroom@floor")
    real_with, x_with = _real_item(1)
    real_without, x_without = _real_item(2, with_event=False)
    missing = dict(it, id="coroomATboth_001.wav")
    rows = [it, it_floor, real_with, real_without, missing]
    outputs = {
        it["id"]: (ref + noise * 10**(-6 / 20), mix, ref),
        it_floor["id"]: (ref, mix, ref),
        real_with["id"]: (x_with, x_with, None),
        real_without["id"]: (x_without, x_without, None),
    }
    out = scoreevents.aggregate(_records(rows, outputs), rows, 0.0)
    assert out["coroom@both"]["n"] == 1 and out["coroom@both"]["n_missing"] == 1
    assert out["coroom@both"]["event_resid_db"] == pytest.approx(-6.0, abs=0.05)
    assert out["coroom@floor"]["floor_resid_db"] == -scoreevents.ITEM_CLIP_DB
    block = out["_coroom"]
    assert block["n"] == 2 and block["n_expected"] == 3 and block[
        "n_missing"] == 1
    assert block["event_resid_db"] == pytest.approx(-6.0, abs=0.05)
    assert block["floor_resid_db"] == pytest.approx(
        (-6.0 - scoreevents.ITEM_CLIP_DB) / 2, abs=0.05)
    assert block["metric_counts"] == {
        "event_resid_db": 1,
        "event_resid_raw_db": 1,
        "floor_resid_db": 2,
        "floor_resid_raw_db": 2,
        "voice_level_db": 2,
        "voice_pair_db": 2,
        "voice_kept_db": 2,
    }
    assert out["coroom@floor"]["metric_counts"]["floor_resid_db"] == 1
    real = out["_real_events"]
    assert real["n"] == 1 and real["n_without_events"] == 1 and real[
        "n_events"] == 1
    assert real["event_att_db"] == pytest.approx(0.0, abs=1e-6)
    assert out["_level_cal_db"] == 0.0


def test_a_view_re_reads_the_records_at_its_own_calibration():
    it, mix, ref, noise = _same_room_item()
    real_with, x_with = _real_item(1)
    rows = [it, real_with]
    outputs = {
        it["id"]: (ref + noise * 10**(-6 / 20), mix, ref),
        real_with["id"]: (x_with, x_with, None),
    }
    records = _records(rows, outputs)
    for rec in records:
        rec["gain_db"] = 0.0
    whole = scoreevents.aggregate(records, rows, 0.0)
    # 6 dB more gain undone: 6 dB less of the noise left, 6 dB less voice
    # against the reference, the paired voice read unchanged
    view = scoreevents.aggregate(records, rows, 6.0)
    # 6 dB of voice lost is under the gate: the view credits no removal
    assert view["coroom@both"]["event_resid_db"] == 0.0
    assert view["coroom@both"]["floor_resid_db"] == 0.0
    # at 2 dB the voice is kept and the rows move with the gain
    mild = scoreevents.aggregate(records, rows, 2.0)
    assert mild["coroom@both"]["event_resid_db"] == pytest.approx(
        whole["coroom@both"]["event_resid_db"] - 2.0, abs=0.05)
    assert mild["coroom@both"]["voice_level_db"] == pytest.approx(-2.0, abs=0.1)
    assert view["coroom@both"]["voice_level_db"] == pytest.approx(-6.0, abs=0.1)
    assert view["coroom@both"]["voice_pair_db"] == pytest.approx(
        whole["coroom@both"]["voice_pair_db"], abs=1e-6)
    assert view["_real_events"]["event_att_db"] == pytest.approx(-6.0, abs=1e-6)
    assert view["_real_events"]["voice_level_db"] == pytest.approx(-6.0,
                                                                   abs=1e-6)
    assert view["_level_cal_db"] == 6.0


def test_a_label_edit_makes_the_cached_measurement_stale():
    """Event spans, pauses, the floor and the room allowance are what a
    same-room measurement reads: changing any of them changes its identity."""
    row = {
        "id": "coroomATboth_000.wav",
        "condition": "coroom@both",
        "provenance": {
            "events": [[1.0, 1.1, "speech", "household", "src:a", -23.0]],
            "pauses": [[2.0, 3.0]],
            "floor": {
                "corruptor": "src:b",
                "floor_dbfs": -60.0
            },
            "split": "main",
            "room_share_db": 0.8,
            "room_share_rule": scoreevents.ROOM_SHARE_RULE,
            "scene": 0,
        },
    }
    base = scoreevents.labels_digest(row)
    assert base is not None
    for key, value in (("events",
                        [[1.05, 1.1, "speech", "household", "src:a",
                          -23.0]]), ("pauses", [[2.0, 3.1]]),
                       ("room_share_db", 0.9), ("floor", {
                           "corruptor": "src:b",
                           "floor_dbfs": -55.0
                       })):
        edited = dict(row, provenance=dict(row["provenance"], **{key: value}))
        assert scoreevents.labels_digest(edited) != base, key
    # a field no measurement reads leaves the identity alone
    other = dict(row, provenance=dict(row["provenance"], scene=7))
    assert scoreevents.labels_digest(other) == base
    assert scoreevents.labels_digest({"condition": "babble@+4"}) is None


def test_a_view_refuses_records_measured_under_other_labels(
        tmp_path, monkeypatch, capsys):
    """The aggregate-only entry path compares every same-room record's labels
    with the current manifest: a relabelled item stops the view instead of
    deriving it from the old observations."""
    row = {
        "id": "coroomATboth_000.wav",
        "condition": "coroom@both",
        "provenance": {
            "events": [[1.0, 1.1, "speech", "household", "src:a", -23.0]],
            "pauses": [[2.0, 3.0]],
            "floor": {
                "corruptor": "src:b",
                "floor_dbfs": -60.0
            },
            "split": "main",
            "room_share_db": 0.8,
            "room_share_rule": scoreevents.ROOM_SHARE_RULE,
        },
    }
    record = {
        "id": row["id"],
        "output_status": "ok",
        "identity": {
            "labels": scoreevents.labels_digest(row)
        },
        "measurements": {},
    }
    relabelled = json.loads(json.dumps(row))
    relabelled["provenance"]["events"] = []
    relabelled["provenance"]["room_share_db"] = 99.0
    written = []
    monkeypatch.setattr(common, "guard_private_results", lambda rows: None)
    monkeypatch.setattr(common, "RESULTS", str(tmp_path))
    monkeypatch.setattr(
        scorebench, "view_records", lambda path, name, ids, sha: ("", {
            "outputs_sha256": None,
            "lag": 0
        }, [record]))
    monkeypatch.setattr(scoreevents, "base_calibration", lambda *a: (0.0, 0))
    monkeypatch.setattr(scoreevents, "aggregate", lambda *a: {})
    monkeypatch.setattr(scoreevents, "write", lambda *a: written.append(a))
    monkeypatch.setattr(sys, "argv", [
        "scoreevents.py", "--systems", "half", "--aggregate-only",
        str(tmp_path)
    ])
    monkeypatch.setattr(common, "manifest", lambda: [relabelled])
    scoreevents.main()
    assert not written
    assert "measured under other labels" in capsys.readouterr().out
    monkeypatch.setattr(common, "manifest", lambda: [row])
    scoreevents.main()
    assert len(written) == 1


def test_turning_both_renders_down_earns_no_removal_credit():
    """Attenuating the output on both renders shrinks the measured residue,
    so without the voice gate it would read as noise removed; at and beyond
    the voice row's floor every such item reads its removal rows as left,
    whatever the attenuation, and the board cannot be gamed by ducking."""
    it, mix, ref, _ = _same_room_item()
    for gain_db in (-3.1, -6.0, -9.9):
        gain = 10**(gain_db / 20)
        rows = _rows(it, mix * gain, mix, ref, out_ref=ref * gain)
        assert rows["voice_level_db"] == pytest.approx(gain_db, abs=0.1)
        assert rows["event_resid_db"] == 0.0 and rows["floor_resid_db"] == 0.0
    # just above the floor the voice row charges the loss and the residue
    # rows read the attenuation as removal: the two move together
    mild = _rows(it, mix * 10**(-2 / 20), mix, ref, out_ref=ref * 10**(-2 / 20))
    assert mild["voice_level_db"] == pytest.approx(-2.0, abs=0.1)
    assert mild["event_resid_db"] == pytest.approx(-2.0, abs=0.1)


def test_shares_measured_under_another_allowance_rule_are_refused():
    """The set's provenance names the rule its shares were measured under; the
    scorer reads only shares measured under its own (v2, the room-free voice
    at the direct arrival), and the recorded rule is part of every label
    digest."""
    migrated = {
        "id": "coroomATpause_000.wav",
        "condition": "coroom@pause",
        "provenance": {
            "room_share_db": 1.2,
            "room_share_rule": scoreevents.ROOM_SHARE_RULE,
        },
    }
    unmigrated = dict(migrated, provenance={"room_share_db": 1.2})
    v1 = dict(migrated,
              provenance={
                  "room_share_db": 1.2,
                  "room_share_rule": "level-matched dry share on the "
                                     "reference's loudest fifth of speech "
                                     "frames, v1"
              })

    shares = scoreevents.room_shares([migrated])

    assert shares == {"coroomATpause_000.wav": 1.2}
    assert "direct arrival" in scoreevents.ROOM_SHARE_RULE
    for stale in (unmigrated, v1):
        with pytest.raises(SystemExit, match="measured under"):
            scoreevents.room_shares([stale])
        assert scoreevents.labels_digest(stale) != scoreevents.labels_digest(
            migrated)
