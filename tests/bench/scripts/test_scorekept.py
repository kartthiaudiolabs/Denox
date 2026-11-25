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
"""The kept sidecar reads the voice outside speech.

It also reads the spectrum of clean speech.
"""

# pylint: disable=redefined-outer-name

import json
import os
import pathlib

import numpy as np
import pytest
import soundfile as sf
import torch

import common
import scorebench
import scorekept

SR = common.SR
# one second of speech, a 400 ms laugh
SPEECH, LAUGH = (0, SR), (int(1.3 * SR), int(1.7 * SR))


class _Vad:
    """Calls the first second speech, whatever it is handed."""

    def segments(self, x):
        # pylint: disable-next=import-outside-toplevel
        import torchaudio

        return torchaudio.functional.resample(x, SR, 16000), [{
            "start": 0,
            "end": 16000
        }]


def _voice(seed):
    ref = torch.zeros(2 * SR)
    g = torch.Generator().manual_seed(seed)
    ref[SPEECH[0]:SPEECH[1]] = 0.1 * torch.randn(SR, generator=g)
    ref[LAUGH[0]:LAUGH[1]] = 0.05 * torch.randn(LAUGH[1] - LAUGH[0],
                                                generator=g)
    return ref


def _write(path, x):
    sf.write(str(path), x.numpy(), SR, subtype="FLOAT")


@pytest.fixture
def vocal_set(tmp_path, monkeypatch):
    """Returns recordings of laughs, under a bed and clean, and clean speech.

    Laughs under a quiet bed (one of them buried under a loud one), laughs on
    clean recordings and clean speech; a system that keeps everything and one
    that gates the laugh.
    """
    root, out_root = tmp_path / "set", tmp_path / "out"
    for d in (
            root / "mix",
            root / "ref",
            out_root / "keeps",
            out_root / "gates",
    ):
        d.mkdir(parents=True)
    rows = []
    for i, (cond, bed) in enumerate((
        ("nonverbal@+0", 0.02),
        ("nonverbal@+0", 0.02),
        ("nonverbal@+0", 0.5),
        ("nonverbal_clean", 0.0),
        ("nonverbal_clean", 0.0),
        ("clean", 0.0),
        ("clean", 0.0),
    )):
        rid = f"{cond.replace('@+', 'ATp')}_{i:03d}.wav"
        ref = _voice(i)
        mix = ref + bed * torch.randn(
            len(ref), generator=torch.Generator().manual_seed(50 + i))
        gated = mix.clone()
        gated[LAUGH[0]:LAUGH[1]] = 0.0
        _write(root / "ref" / rid, ref)
        _write(root / "mix" / rid, mix)
        _write(out_root / "keeps" / rid, mix)
        _write(out_root / "gates" / rid, gated)
        rows.append({
            "id": rid,
            "condition": cond,
            "mix_path": f"mix/{rid}",
            "clean_path": f"ref/{rid}",
        })
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
    return common.manifest(str(root))


def test_module_applies_the_identity_rule_of_scorebench():
    # one rule decides which renders a sidecar may read: scorebench's
    assert not hasattr(scorekept, "render_problem")
    assert "render_problem" in scorekept.main.__code__.co_names


def test_events_are_runs_of_consecutive_blocks():
    mask = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)
    assert scorekept.events(mask) == [(1, 3), (4, 5), (7, 8)]
    assert not scorekept.events(torch.zeros(4, dtype=torch.bool))


def test_vocal_blocks_are_loud_outside_speech_and_not_buried():
    ref = _voice(0)
    speech = torch.zeros(20, dtype=torch.bool)
    speech[:10] = True
    alone = scorekept.vocal_blocks(ref, speech, ref)
    # the laugh, not the speech
    assert torch.nonzero(alone).flatten().tolist() == [13, 14, 15, 16]
    quiet_bed = ref + 0.02 * torch.randn(
        len(ref), generator=torch.Generator().manual_seed(1))
    with_bed = scorekept.vocal_blocks(ref, speech, quiet_bed)
    assert with_bed.tolist() == alone.tolist()
    loud_bed = ref + 0.5 * torch.randn(
        len(ref), generator=torch.Generator().manual_seed(2))
    # 20 dB under the bed: nothing to keep
    assert not scorekept.vocal_blocks(ref, speech, loud_bed).any()
    whisper = ref.clone()
    # 46 dB under the loudest block: not a vocal sound
    whisper[scorekept.FRAME * 13:] *= 0.01
    assert not scorekept.vocal_blocks(whisper, speech, whisper).any()


def test_magnitude_ratio_ignores_gain_and_phase_and_reads_a_changed_spectrum():
    ref = _voice(3)
    assert scorekept.mag_snr(ref, ref) == 120.0
    assert scorekept.mag_snr(0.5 * ref, ref) > 100.0
    assert scorekept.mag_snr(-ref, ref) > 100.0
    g = torch.Generator().manual_seed(4)
    slight = ref + 0.001 * torch.randn(len(ref), generator=g)
    heavy = ref + 0.02 * torch.randn(len(ref), generator=g)
    heavy_snr = scorekept.mag_snr(heavy, ref)
    slight_snr = scorekept.mag_snr(slight, ref)
    assert 5.0 < heavy_snr < slight_snr < 120.0
    assert np.isnan(scorekept.mag_snr(ref[:100], ref[:100]))


def test_a_gated_laugh_is_lost_and_a_kept_one_is_not(vocal_set):
    rows = [r for r in vocal_set if scorekept.wanted(r)]
    assert len(rows) == 7
    kept_records = scorekept.assemble("keeps", rows, vad=_Vad())
    kept = scorekept.aggregate(kept_records, rows, 0.0)
    gated_records = scorekept.assemble("gates", rows, vad=_Vad())
    gated = scorekept.aggregate(gated_records, rows, 0.0)
    noisy, clean = kept["nonverbal@+0"], kept["nonverbal_clean"]
    # the buried laugh is nobody's to keep
    assert noisy["n"] == 3 and noisy["nv_items"] == 2
    assert noisy["nv_blocks"] == 8 and noisy["nv_events"] == 2
    assert noisy["nv_lost_pct"] == 0.0 and clean["nv_lost_pct"] == 0.0
    assert gated["nonverbal@+0"]["nv_lost_pct"] == 100.0
    assert gated["nonverbal@+0"]["nv_events_lost_pct"] == 100.0
    assert gated["nonverbal_clean"]["nv_lost_pct"] == 100.0
    by_id = {r["id"]: r for r in gated_records}
    assert by_id[rows[0]["id"]]["metrics"]["nv_lost"] == 4
    assert by_id[rows[0]["id"]]["metrics"]["nv_lost_pct"] == 100.0
    assert by_id[rows[2]["id"]]["metrics"]["nv_blocks"] == 0
    assert "nv_lost_pct" not in by_id[rows[2]["id"]]["metrics"]
    # gating a laugh changes the spectrum of nothing in the clean condition's
    # speech
    assert (kept["clean"]["mag_snr"] == 120.0 and
            "nv_lost_pct" not in kept["clean"])
    # the clean items hold the same laugh, now gone
    assert gated["clean"]["mag_snr"] < 30.0


def test_level_calibration_is_removed_before_a_block_is_called_lost(vocal_set):
    rows = [r for r in vocal_set if r["condition"] == "nonverbal_clean"]
    quiet = os.path.join(common.OUT_ROOT, "quiet")
    os.makedirs(quiet)
    for r in rows:
        _write(
            pathlib.Path(quiet) / r["id"], 0.25 * common.load48(r["mix_path"]))
    records = scorekept.assemble("quiet", rows, vad=_Vad())
    uncalibrated = scorekept.aggregate(records, rows, 0.0)["nonverbal_clean"]
    assert uncalibrated["nv_lost_pct"] == 100.0
    calibrated = scorekept.aggregate(records, rows, -12.0)["nonverbal_clean"]
    assert calibrated["nv_lost_pct"] == 0.0


def test_records_reuse_and_views_derive(vocal_set):
    rows = [r for r in vocal_set if scorekept.wanted(r)]
    first = scorekept.assemble("gates", rows, vad=_Vad())
    whole = scorekept.aggregate(first, rows, 0.0)
    index = {scorebench.reuse_key(r["identity"]): r for r in first}
    # no detector needed: nothing is scored
    again = scorekept.assemble("gates", rows, reuse=index)
    assert scorekept.aggregate(again, rows, 0.0) == whole
    reused = [r["measurements"] for r in again]
    assert reused == [r["measurements"] for r in first]
    view = rows[:2] + rows[5:6]
    derived = scorekept.aggregate(
        [r for r in first if r["id"] in {v["id"] for v in view}], view, 0.0)
    direct_records = scorekept.assemble("gates", view, vad=_Vad())
    direct = scorekept.aggregate(direct_records, view, 0.0)
    assert derived == direct and derived["nonverbal@+0"]["n"] == 2
    assert "nonverbal_clean" not in derived


def test_the_files_name_the_render_they_read_and_a_view_keeps_it(
        vocal_set, tmp_path, monkeypatch):
    # pylint: disable-next=import-outside-toplevel
    import sys

    results, view = pathlib.Path(common.RESULTS), tmp_path / "view"
    results.mkdir()
    view.mkdir()
    rendered = "e" * 64
    (results / "gates.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "gates",
            "outputs_sha256": rendered
        }))
    for where, subset in ((results, "all"), (view, "gate")):
        (where / "gates.metrics.json").write_text(
            json.dumps({
                "_set_id": "tiny",
                "_subset": subset,
                "_level_cal_db": 0.0
            }))
    monkeypatch.setattr(common, "Vad", _Vad)
    monkeypatch.setattr(sys, "argv", ["scorekept.py", "--systems", "gates"])
    scorekept.main()
    whole = json.loads((results / "gates.kept.json").read_text())
    with open(results / "gates.kept.items.jsonl", encoding="utf-8") as fh:
        header = json.loads(fh.readline())
    assert whole["_outputs_sha256"] == rendered == header["outputs_sha256"]
    members = {r["id"] for r in vocal_set[:2]}
    monkeypatch.setattr(common, "SUBSET", "gate")
    monkeypatch.setattr(common,
                        "panel_members",
                        lambda name, root=None: members)
    monkeypatch.setattr(common, "RESULTS", str(view))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scorekept.py", "--systems", "gates", "--aggregate-only",
            str(results)
        ],
    )
    scorekept.main()
    derived = json.loads((view / "gates.kept.json").read_text())
    assert derived["_outputs_sha256"] == rendered


def test_missing_outputs_are_counted_not_averaged_away(vocal_set):
    rows = [r for r in vocal_set if scorekept.wanted(r)]
    os.remove(os.path.join(common.OUT_ROOT, "keeps", rows[0]["id"]))
    records = scorekept.assemble("keeps", rows, vad=_Vad())
    out = scorekept.aggregate(records, rows, 0.0)
    assert records[0]["output_status"] == "missing"
    assert (out["nonverbal@+0"]["n"] == 2 and
            out["nonverbal@+0"]["n_missing"] == 1)
