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
"""The three sidecars keep per-item records.

The records are reused across passes and derive any view.
"""

# pylint: disable=redefined-outer-name,unused-argument

import json
import pathlib

import pytest
import soundfile as sf
import torch

import common
import levels
import observations
import scorebench
import scorenisqa
import scorerealx

SR = common.SR


def _write(path, x):
    sf.write(str(path), x.numpy(), SR, subtype="FLOAT")


@pytest.fixture
def tiny_real_set(tmp_path, monkeypatch):
    """Two babble items with references and three real recordings.

    The set comes with the outputs of a system 'half'.
    """
    root, out_root = tmp_path / "set", tmp_path / "out"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    (out_root / "half").mkdir(parents=True)
    n = 2 * SR
    rows = []
    for i in range(5):
        g = torch.Generator().manual_seed(i)
        speech = torch.randn(n, generator=g) * 0.1
        noise = (
            torch.randn(n, generator=torch.Generator().manual_seed(100 + i)) *
            0.05)
        real = i >= 2
        rid = f"{'real_x' if real else 'babbleATp4'}_{i:03d}.wav"
        mix = speech + noise
        _write(root / "mix" / rid, mix)
        row = {
            "id": rid,
            "condition": "real_x" if real else "babble@+4",
            "mix_path": f"mix/{rid}",
        }
        if not real:
            _write(root / "ref" / rid, speech)
            row["clean_path"] = f"ref/{rid}"
        rows.append(row)
        # the last one deleted
        _write(out_root / "half" / rid, mix * (0.5 if i != 4 else 0.02))
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump(
        {
            "set_id": "tiny",
            "n_items": len(rows)
        },
        open(root / "checksums.json", "w", encoding="utf-8"),
    )
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(out_root))
    monkeypatch.setattr(common, "RESULTS", str(tmp_path / "results"))

    def fake_nisqa(x):
        return float(3 + x.abs().mean()), 1.0, 2.0, 3.0, 4.0

    monkeypatch.setattr(common, "nisqa", fake_nisqa)
    return common.manifest(str(root))


class _Vad:

    def segments(self, x):
        # pylint: disable-next=import-outside-toplevel
        import torchaudio

        x16 = torchaudio.functional.resample(x, SR, 16000)
        # two inner segments: one touching an edge would be a cut-off word
        return x16, [
            {
                "start": len(x16) // 8,
                "end": len(x16) // 2
            },
            {
                "start": len(x16) // 2,
                "end": 7 * len(x16) // 8
            },
        ]


class _Spk:

    def emb(self, w):
        return torch.tensor([1.0, 0.0])


def _index(records, system, scorer):
    return {
        scorebench.reuse_key(r["identity"]): r
        for r in records
        if r.get("identity")
    }


def test_nisqa_records_reuse_and_the_mean_reads_the_first_items_of_a_view(
    tiny_real_set,):
    rows = tiny_real_set
    want = ["babble@+4", "real_x"]
    first = scorenisqa.assemble("half", rows)
    assert all(r["identity"]["output"] for r in first)
    whole = scorenisqa.aggregate(first, rows, want, 48)
    assert whole["real_x"]["n"] == 3 and whole["babble@+4"]["n"] == 2
    reuse = _index(first, "half", scorenisqa.SCORER_SHA256)
    again = scorenisqa.assemble("half", rows, reuse=reuse)
    assert [r["metrics"] for r in again] == [r["metrics"] for r in first]
    # the published sample: the first item(s)
    limited = scorenisqa.aggregate(first, rows, want, 1)
    assert limited["real_x"]["n"] == 1 and limited["real_x"]["n_expected"] == 1
    # a view of two real items
    view = [r for r in rows if r["condition"] == "real_x"][1:]
    derived = scorenisqa.aggregate(
        [r for r in first if r["id"] in {v["id"] for v in view}], view, want,
        48)
    direct_records = scorenisqa.assemble("half", view)
    direct = scorenisqa.aggregate(direct_records, view, want, 48)
    assert derived == direct and "babble@+4" not in derived


def test_realx_records_carry_segments_and_anchors_and_views_derive(
    tiny_real_set,):
    pytest.importorskip("scipy", reason="needs the eval extra")
    rows = tiny_real_set
    items = scorerealx.select_items(rows, "real")
    first = scorerealx.assemble("half", items, _Vad(), _Spk(), 0.85)
    out = scorerealx.aggregate(first, items, 0.85, 0.0)
    assert (out["_pooled_real"]["n_segments"] == 6 and
            out["_pooled_real"]["n_deleted"] == 2)
    assert out["_pooled_primary"]["n_segments"] == 6
    assert out["_items_primary"][items[-1]["id"]] == [2, 2]
    assert set(out["_anchor_sims"]) == {it["id"] for it in items}
    last = first[-1]["metrics"]
    assert last["n_segments"] == 2 and last["primary_segments"] == 2
    assert levels.deleted(last["primary_drops"], 0.0) == [1.0, 1.0]
    again = scorerealx.assemble(
        "half",
        items,
        _Vad(),
        _Spk(),
        0.85,
        reuse=_index(first, "half", scorerealx.SCORER_SHA256),
    )
    reused = [r["measurements"] for r in again]
    assert reused == [r["measurements"] for r in first]
    # the anchoring rule is part of the identity: another threshold scores
    # afresh
    other = scorerealx.assemble(
        "half",
        items,
        _Vad(),
        _Spk(),
        0.5,
        reuse=_index(first, "half", scorerealx.SCORER_SHA256),
    )
    assert all(
        r["identity"]["semantics"]["sim_threshold"] == 0.5 for r in other)
    view = items[1:]
    derived = scorerealx.aggregate(
        [r for r in first if r["id"] in {v["id"] for v in view}],
        view,
        0.85,
        0.0,
    )
    direct_records = scorerealx.assemble("half", view, _Vad(), _Spk(), 0.85)
    direct = scorerealx.aggregate(direct_records, view, 0.85, 0.0)
    assert derived == direct and derived["_pooled_real"]["n_segments"] == 4
    # without anchoring there is no primary pool and no anchor record
    plain_records = scorerealx.assemble("half", items, _Vad(), None, 0.85)
    plain = scorerealx.aggregate(plain_records, items, 0.85, 0.0)
    assert "_pooled_primary" not in plain and "_anchor_sims" not in plain


def test_nisqa_records_carry_the_published_cohort(tiny_real_set):
    """Only the first ``limit`` items of a condition feed the published mean.

    The others keep their values under measurements, so a paired comparison
    reads the board's cohort.
    """
    rows = tiny_real_set
    want = ["real_x"]
    records = scorenisqa.assemble("half", rows)
    members = scorenisqa.cohort(rows, want, 2)
    assert len(members) == 2
    scorenisqa.mark(records, members)
    inside = [r for r in records if r["id"] in members]
    outside = [
        r for r in records
        if r["condition"] == "real_x" and r["id"] not in members
    ]
    assert all(
        r["cohort"] and r["metrics"] and not r["measurements"] for r in inside)
    assert all(
        not r["cohort"] and not r["metrics"] and r["measurements"]["nisqa"]
        for r in outside)
    assert scorenisqa.aggregate(records, rows, want, 2)["real_x"]["n"] == 2
    # a view whose cohort includes an item outside the whole set's finds its
    # values again
    view = [r for r in rows if r["condition"] == "real_x"][1:]
    scorenisqa.mark(records, scorenisqa.cohort(view, want, 48))
    assert all(
        r["metrics"] for r in records if r["id"] in {v["id"] for v in view})
    assert scorenisqa.aggregate(records, view, want, 48)["real_x"]["n"] == 2


def test_primary_speaker_quality_reads_frozen_input_segments_one_at_a_time(
        tiny_real_set, monkeypatch):
    rows = tiny_real_set
    items = scorerealx.select_items(rows, "real")
    records = scorerealx.assemble("half", items, _Vad(), _Spk(), 0.85)
    m = records[0]["measurements"]
    # the input's segments
    assert m["seg_start"] == [4000, 16000] and m["seg_end"] == [16000, 28000]
    # the fake NISQA reads level: a segment of the loud half scores higher than
    # a quiet one
    out = torch.cat([torch.full((48000,), 0.5), torch.full((48000,), 0.05)])
    q = scorenisqa.primary_quality(out, [(0, 16000), (16000, 32000)],
                                   min_seconds=0.5)
    assert (q["nisqa_primary_eligible_s"] == 2.0 and
            q["nisqa_primary_coverage"] == 1.0)
    # duration-weighted mean
    assert abs(q["nisqa_mos_primary"] - (3.5 + 3.05) / 2) < 1e-6
    q = scorenisqa.primary_quality(out, [(0, 16000), (16000, 24000)],
                                   min_seconds=1.0)
    assert (q["nisqa_primary_eligible_s"] == 1.0 and
            abs(q["nisqa_primary_coverage"] - 2 / 3) < 1e-9)
    # the half-second segment is not scored on its own
    assert q["nisqa_mos_primary"] == 3.5
    assert "nisqa_mos_primary" not in scorenisqa.primary_quality(
        out, [(0, 8000)], min_seconds=1.0)
    # the frozen segments come from INPUT's deletion records, never from an
    # output
    path = str(pathlib.Path(common.RESULTS) / "INPUT.realx.items.jsonl")
    observations.sidecar(path, records, "INPUT", "tiny", "all", False,
                         scorerealx.SCORER_SHA256)
    segs = scorenisqa.primary_segments(common.RESULTS)
    assert segs[items[0]["id"]] == [(4000, 16000), (16000, 28000)]
    # and only INPUT's, of this set, of the whole set, by the current deletion
    # scorer, complete
    observations.sidecar(path, records, "half", "tiny", "all", False,
                         scorerealx.SCORER_SHA256)
    with pytest.raises(SystemExit, match="not INPUT"):
        scorenisqa.primary_segments(common.RESULTS)
    observations.sidecar(path, records, "INPUT", "previous", "all", False,
                         scorerealx.SCORER_SHA256)
    with pytest.raises(SystemExit, match="not the whole current set"):
        scorenisqa.primary_segments(common.RESULTS)
    observations.sidecar(path, records, "INPUT", "tiny", "gate", False,
                         scorerealx.SCORER_SHA256)
    with pytest.raises(SystemExit, match="not the whole current set"):
        scorenisqa.primary_segments(common.RESULTS)
    observations.sidecar(path, records, "INPUT", "tiny", "all", False,
                         "an earlier scorer")
    with pytest.raises(SystemExit, match="another deletion scorer"):
        scorenisqa.primary_segments(common.RESULTS)
    observations.sidecar(path, records[:-1], "INPUT", "tiny", "all", False,
                         scorerealx.SCORER_SHA256)
    with pytest.raises(SystemExit, match="covers 2 of 3 real items"):
        scorenisqa.primary_segments(common.RESULTS)
    # a pass without --primary-anchor: segments present, no speaker anchoring,
    # nothing primary
    unanchored = scorerealx.assemble("half", items, _Vad(), None, 0.85)
    assert all(not r["measurements"]["anchor_sims"] for r in unanchored)
    assert all(
        p == 0.0 for r in unanchored for p in r["measurements"]["seg_primary"])
    observations.sidecar(path, unanchored, "INPUT", "tiny", "all", False,
                         scorerealx.SCORER_SHA256)
    with pytest.raises(SystemExit, match="no speaker anchoring"):
        scorenisqa.primary_segments(common.RESULTS)
    # a recording without usable speech is not an unanchored pass
    silent = [dict(r, measurements=dict(r["measurements"])) for r in records]
    silent[0]["measurements"].update(seg_start=[],
                                     seg_end=[],
                                     seg_drop=[],
                                     seg_primary=[],
                                     anchor_sims=[])
    observations.sidecar(path, silent, "INPUT", "tiny", "all", False,
                         scorerealx.SCORER_SHA256)
    segs = scorenisqa.primary_segments(common.RESULTS)
    assert items[0]["id"] not in segs and items[1]["id"] in segs


def test_nisqa_and_deletion_files_name_the_render_they_read_and_views_keep_it(
        tiny_real_set, tmp_path, monkeypatch):
    # pylint: disable-next=import-outside-toplevel
    import sys

    results, view = tmp_path / "results", tmp_path / "view"
    results.mkdir()
    view.mkdir()
    rendered = "f" * 64
    (results / "half.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "half",
            "outputs_sha256": rendered
        }))
    for where, subset in ((results, "all"), (view, "gate")):
        (where / "half.metrics.json").write_text(
            json.dumps({
                "_set_id": "tiny",
                "_subset": subset,
                "_level_cal_db": 0.0
            }))
    monkeypatch.setattr(common, "Vad", _Vad)
    runs = (
        (scorenisqa, "nisqa", ["--conds", "babble@+4,real_x"]),
        (scorerealx, "realx", []),
    )
    for module, ext, extra in runs:
        monkeypatch.setattr(sys, "argv",
                            [f"score{ext}.py", "--systems", "half"] + extra)
        module.main()
        whole = json.loads((results / f"half.{ext}.json").read_text())
        with open(results / f"half.{ext}.items.jsonl", encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        assert whole["_outputs_sha256"] == rendered
        assert header["outputs_sha256"] == rendered
    # a view: the digest comes from the whole-set records, with no stamp beside
    members = {r["id"] for r in tiny_real_set[1:]}
    monkeypatch.setattr(common, "SUBSET", "gate")
    monkeypatch.setattr(common,
                        "panel_members",
                        lambda name, root=None: members)
    monkeypatch.setattr(common, "RESULTS", str(view))
    for module, ext, extra in runs:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                f"score{ext}.py", "--systems", "half", "--aggregate-only",
                str(results)
            ] + extra,
        )
        module.main()
        derived = json.loads((view / f"half.{ext}.json").read_text())
        assert derived["_outputs_sha256"] == rendered
