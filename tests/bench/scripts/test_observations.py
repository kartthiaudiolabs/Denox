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
"""Observation provenance, reproducible raw measurements, complete records."""

import json

import pytest

import observations


def test_windows_share_the_recording_group_without_disclosing_source_names():
    row = {
        "id": "a.wav",
        "condition": "babble@0",
        "clean_path": "ref/a.wav",
        "family": "babble",
        "provenance": {
            "speech": "private:utterance",
            "corruptor": "private:bed@0.0s",
        },
    }
    a = observations.item(row)
    row["provenance"]["corruptor"] = "private:bed@6.0s"
    b = observations.item(row)
    assert a["bed_source"] == b["bed_source"] and a["groups_verified"]
    assert "private:" not in json.dumps(a)
    del row["provenance"]
    assert not observations.item(row)["groups_verified"]


def test_rooms_group_by_response_and_bedless_conditions_by_recording_alone():
    room = {
        "id": "r.wav",
        "condition": "reverb_real",
        "clean_path": "ref/r.wav",
        "provenance": {
            "speech": "private:utterance",
            "rir": "private:hall-3"
        },
    }
    a, b = observations.item(room), observations.item(dict(room, id="s.wav"))
    assert (a["groups_verified"] and a["bed_source"] and
            a["bed_source"] == b["bed_source"])
    other = observations.item(
        dict(room, provenance=dict(room["provenance"], rir="private:hall-4")))
    assert other["bed_source"] != a["bed_source"]
    assert "private:" not in json.dumps(a)
    no_rir = dict(room, provenance={"speech": "private:utterance"})
    assert not observations.item(no_rir)["groups_verified"]
    for condition in observations.BEDLESS:
        rec = observations.item({
            "id": "c.wav",
            "condition": condition,
            "clean_path": "ref/c.wav",
            "provenance": {
                "speech": "private:utterance"
            },
        })
        assert rec["groups_verified"] and rec["bed_source"] is None
    # a condition that mixes a bed in cannot pass without naming it
    noisy = observations.item({
        "id": "n.wav",
        "condition": "nonverbal@+0",
        "clean_path": "ref/n.wav",
        "provenance": {
            "speech": "private:utterance"
        },
    })
    assert not noisy["groups_verified"]


def test_capture_keeps_reducer_inputs_and_missing_values():
    record = {}
    observations.capture(
        record,
        {
            "kept": [2.0, 3.0],
            "seg_drop": [0.0, -30.0, 0.0],
            "band": [[float("nan"), 0]],
        },
        {"kept": 1},
    )
    assert record["measurements"]["kept"] == [3.0]
    # deletions are read at a calibration, later
    assert record["metrics"]["n_segments"] == 3
    assert record["metrics"]["band_kept_db"] == [None, 0]
    json.dumps(record, allow_nan=False)


def test_record_writer_is_private_and_reader_refuses_truncation_and_duplicates(
    tmp_path,):
    path = tmp_path / "items.jsonl"
    meta = {"set_id": "set", "system": "system", "n_expected": 2}
    observations.write(path, [{"id": "a"}, {"id": "b"}], meta)
    assert path.stat().st_mode & 0o777 == 0o600
    assert len(observations.read(path)[1]) == 2
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]))
    with pytest.raises(ValueError, match="truncated"):
        observations.read(path)
    observations.write(path, [{"id": "a"}, {"id": "a"}], meta)
    with pytest.raises(ValueError, match="duplicated"):
        observations.read(path)
