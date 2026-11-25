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
"""Real-recording deletion sidecar: item selection and interval arithmetic."""

import pytest
import torch
import torchaudio

pytest.importorskip("scipy")
# pylint: disable=wrong-import-position
import common
import levels
import observations
import scorebench
import scorerealx

# pylint: enable=wrong-import-position

ROWS = [{
    "condition": c
} for c in ("real_noisy", "real_calls", "real_zh", "babble@+4", "clean")]


def _conds(rows):
    return [r["condition"] for r in rows]


def test_select_items_default_prefix_pools_every_real_condition():
    assert _conds(scorerealx.select_items(ROWS, "real")) == [
        "real_noisy",
        "real_calls",
        "real_zh",
    ]


def test_select_items_explicit_names_are_exact_matches():
    assert _conds(scorerealx.select_items(ROWS, "real_noisy,babble@+4")) == [
        "real_noisy",
        "babble@+4",
    ]
    # a name, not a prefix
    assert _conds(scorerealx.select_items(ROWS, "real_noisy")) == ["real_noisy"]
    # trailing comma
    trailing = scorerealx.select_items(ROWS, "real_noisy,")
    assert _conds(trailing) == ["real_noisy"]
    assert scorerealx.select_items(ROWS, "nothing") == []


def test_ci95_is_clopper_pearson_in_percent():
    lo, hi = scorerealx.ci95(0, 100)
    assert lo == 0.0 and hi == pytest.approx(100 * (1 - 0.025**(1 / 100)),
                                             abs=1e-6)
    lo, hi = scorerealx.ci95(100, 100)
    assert hi == 100.0 and lo == pytest.approx(100 * 0.025**(1 / 100), abs=1e-6)
    lo, hi = scorerealx.ci95(5, 100)
    assert lo == pytest.approx(1.643, abs=0.01) and hi == pytest.approx(
        11.283, abs=0.01)
    # every segment the detector reports counts; Silero reports none shorter
    assert scorerealx.MIN_SEG == 4000 and levels.DEL_DB == -10.0
    assert scorerealx.EDGE == 800


def _record(uid, cond, drops, prim, edge):
    return {
        "id": uid,
        "condition": cond,
        "output_status": "ok",
        "measurements": {
            "seg_drop": drops,
            "seg_primary": prim,
            "seg_edge": edge,
            "seg_start": [16000 * i for i in range(len(drops))],
            "seg_end": [16000 * i + 8000 for i in range(len(drops))],
            "anchor_sims": [[1.0] * len(drops)] if drops else [],
        },
    }


def test_aggregate_leaves_edge_fragments_out_of_every_pool():
    items = [{
        "id": "a",
        "condition": "real_x"
    }, {
        "id": "b",
        "condition": "real_x"
    }]
    records = [
        # the first segment is a word the recording cut off, and it is gone
        _record("a", "real_x", [-30.0, -1.0, -2.0], [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0]),
        # one inner deletion, one fragment kept
        _record("b", "real_x", [-1.0, -15.0, -1.0], [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]),
    ]
    out = scorerealx.aggregate(records, items, 0.85, 0.0)
    assert out["real_x"]["n_segments"] == 4 and out["real_x"]["n_deleted"] == 1
    assert out["_items_real"] == {"a": [0, 2], "b": [1, 2]}
    assert out["_items_primary"] == {"a": [0, 2], "b": [1, 2]}
    assert out["_pooled_primary"]["n_segments"] == 4
    assert out["_edge_segments"] == {
        "real_x": {
            "n_segments": 2,
            "n_deleted": 1
        }
    }
    assert out["_trimmed_segments"] == 0 and out["_widened_segments"] == 0
    # records of an older scorer carry no edge flags: every segment is inner
    legacy = _record("a", "real_x", [-30.0, -1.0], [1.0, 1.0], [])
    legacy["measurements"].pop("seg_edge")
    out = scorerealx.aggregate([legacy], items[:1], 0.85, 0.0)
    assert out["_items_real"] == {"a": [1, 2]} and out["_edge_segments"] == {}
    # the diagnostics count what the records say
    counted = _record("a", "real_x", [-1.0, -2.0], [1.0, 1.0], [0.0, 0.0])
    counted["measurements"].update(seg_trimmed=[1.0, 0.0],
                                   seg_widened=[0.0, 1.0])
    out = scorerealx.aggregate([counted], items[:1], 0.85, 0.0)
    assert out["_trimmed_segments"] == 1 and out["_widened_segments"] == 1


def test_item_metrics_keep_primary_drops_of_inner_segments_only():
    metrics = scorerealx.item_metrics({
        "seg_drop": [-30.0, -1.0, -20.0],
        "seg_primary": [1.0, 1.0, 1.0],
        "seg_edge": [1.0, 0.0, 0.0],
        "seg_trimmed": [0.0, 1.0, 0.0],
        "seg_widened": [1.0, 0.0, 0.0],
    })
    assert metrics == {
        "n_segments": 3,
        "n_edge_segments": 1,
        "n_trimmed_segments": 1,
        "n_widened_segments": 1,
        "primary_segments": 2,
        "primary_drops": [-1.0, -20.0],
    }


class _EdgeVad:
    """A 14 s item: speech from its start to 12.5 s, and its last 0.4 s."""

    def segments(self, x):
        x16 = torchaudio.functional.resample(x, 48000, 16000)
        return x16, [{
            "start": 0,
            "end": 200000
        }, {
            "start": len(x16) - 6400,
            "end": len(x16)
        }]


def test_a_segment_touching_an_edge_loses_the_edge_and_keeps_its_interior(
        tmp_path, monkeypatch):
    monkeypatch.setattr(common, "EVAL", str(tmp_path))
    g = torch.Generator().manual_seed(3)
    mix = torch.randn(14 * 48000, generator=g) * 0.1
    it = {"id": "a.wav", "condition": "real_x", "mix_path": "a.wav"}
    same = scorerealx.score_item(it, mix, mix, _EdgeVad(), [1.0, 1.0], 0.85)
    # the long utterance keeps its interior, read from 300 ms in; the 0.4 s
    # fragment at the end is reported, not read
    assert same["seg_start"] == [scorerealx.TRIM, 224000 - 6400]
    assert same["seg_end"] == [200000, 224000]
    assert same["seg_edge"] == [0.0, 1.0]
    assert same["seg_trimmed"] == [1.0, 0.0]
    assert same["seg_widened"] == [0.0, 1.0]
    assert same["seg_drop"] == pytest.approx([0.0, 0.0], abs=1e-6)
    # the review's case: the whole utterance deleted used to go uncharged
    out = mix.clone()
    out[:int(12.5 * 48000)] = 0.0
    out[-int(0.5 * 48000):] = 0.0
    gone = scorerealx.score_item(it, out, mix, _EdgeVad(), [1.0, 1.0], 0.85)
    assert gone["seg_drop"][0] < -60 and gone["seg_drop"][1] < levels.DEL_DB
    metrics = scorerealx.item_metrics(gone)
    assert metrics["primary_segments"] == 1 and metrics["n_edge_segments"] == 1
    assert levels.deleted(metrics["primary_drops"], 0.0) == [1.0]
    # the first 300 ms alone silenced is the cut-off word, nobody's
    head = mix.clone()
    head[:int(0.3 * 48000)] = 0.0
    assert scorerealx.score_item(it, head, mix, _EdgeVad(), [1.0, 1.0],
                                 0.85)["seg_drop"][0] == pytest.approx(0.0,
                                                                       abs=0.05)


def test_frozen_decisions_bind_reuse_and_require_inputs_boundaries(
        tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    (out_root / "half").mkdir(parents=True)
    (out_root / "half" / "a.wav").write_bytes(b"")
    mix = tmp_path / "a.wav"
    mix.write_bytes(b"")
    items = [{"id": "a.wav", "condition": "real_x", "mix_path": str(mix)}]
    monkeypatch.setattr(common, "OUT_ROOT", str(out_root))
    monkeypatch.setattr(common, "load48", lambda path: torch.ones(48000))
    monkeypatch.setattr(common, "file_hashes", lambda: (lambda path: "h"))
    calls = []

    def fake_score_item(it, out, mix, vad, sims, threshold):
        del it, out, mix, vad
        calls.append(sims)
        flags = [1.0 if s >= threshold else 0.0 for s in (sims or [0.0, 0.0])]
        return {
            "seg_drop": [-1.0, -2.0],
            "seg_primary": flags,
            "seg_edge": [0.0, 0.0],
            "seg_trimmed": [0.0, 0.0],
            "seg_widened": [0.0, 0.0],
            "seg_start": [1000, 9000],
            "seg_end": [5000, 13000],
            "anchor_sims": [list(sims)] if sims else [],
        }

    monkeypatch.setattr(scorerealx, "score_item", fake_score_item)
    decision = {
        "a.wav": {
            "sims": [0.1, 0.99],
            "start": [1000, 9000],
            "end": [5000, 13000]
        }
    }
    spk = object()  # anchoring on, the embedder never called
    first = scorerealx.assemble("half", items, None, spk, 0.85, frozen=decision)
    key = scorerealx.frozen_key(decision["a.wav"])
    assert first[0]["identity"]["semantics"]["frozen"] == key
    assert first[0]["measurements"]["seg_primary"] == [0.0, 1.0]
    # the same decision of INPUT: the record is reused
    index = {scorebench.reuse_key(r["identity"]): r for r in first}
    again = scorerealx.assemble("half",
                                items,
                                None,
                                spk,
                                0.85,
                                reuse=index,
                                frozen=decision)
    assert len(calls) == 1
    assert again[0]["measurements"] == first[0]["measurements"]
    # INPUT decided again and the flags swapped: the cached record is of
    # another decision and is not reused (the review's cache test)
    swapped = {"a.wav": dict(decision["a.wav"], sims=[0.99, 0.1])}
    fresh = scorerealx.assemble("half",
                                items,
                                None,
                                spk,
                                0.85,
                                reuse=index,
                                frozen=swapped)
    assert len(calls) == 2
    assert fresh[0]["measurements"]["seg_primary"] == [1.0, 0.0]
    assert fresh[0]["identity"]["semantics"]["frozen"] != key
    # the boundaries must be INPUT's to the sample, not merely as many
    moved = {"a.wav": dict(decision["a.wav"], start=[1000, 9100])}
    with pytest.raises(SystemExit, match="differ from INPUT's"):
        scorerealx.assemble("half", items, None, spk, 0.85, frozen=moved)
    # a pass without anchoring carries no decision
    own = scorerealx.assemble("half", items, None, None, 0.85)
    assert own[0]["identity"]["semantics"]["frozen"] is None


def test_frozen_sims_reads_inputs_flags_and_refuses_other_records(
        tmp_path, monkeypatch):
    monkeypatch.setattr(common, "set_id", lambda root=None: "tiny")
    items = [{"id": "a"}, {"id": "b"}]
    records = [
        _record("a", "real_x", [-1.0, -2.0], [1.0, 0.0], [0.0, 0.0]),
        _record("b", "real_x", [], [], []),
    ]
    records[0]["measurements"]["anchor_sims"] = [[0.9, 0.2]]
    path = tmp_path / "INPUT.realx.items.jsonl"
    observations.sidecar(path, records, "INPUT", "tiny", "all", False,
                         scorerealx.SCORER_SHA256)
    assert scorerealx.frozen_sims(path, items) == {
        "a": {
            "sims": [0.9, 0.2],
            "start": [0, 16000],
            "end": [8000, 24000]
        },
        "b": {
            "sims": None,
            "start": [],
            "end": []
        },
    }
    assert scorerealx.frozen_key({"sims": [0.9, 0.2]}) != scorerealx.frozen_key(
        {"sims": [0.2, 0.9]})
    with pytest.raises(SystemExit, match="lacks 1 items"):
        scorerealx.frozen_sims(path, items + [{"id": "c"}])
    observations.sidecar(path, records, "half", "tiny", "all", False,
                         scorerealx.SCORER_SHA256)
    with pytest.raises(SystemExit, match="not INPUT"):
        scorerealx.frozen_sims(path, items)
    observations.sidecar(path, records, "INPUT", "tiny", "all", False, "other")
    with pytest.raises(SystemExit, match="another deletion scorer"):
        scorerealx.frozen_sims(path, items)
    with pytest.raises(SystemExit, match="missing"):
        scorerealx.frozen_sims(tmp_path / "none.jsonl", items)
