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
"""The reference scorer end to end on a tiny synthetic set.

Heavy models are stubbed, so the tests run on CPU in seconds.
"""

# pylint: disable=redefined-outer-name,unused-argument

import json
import math
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

import common
import levels
import observations
import scorebench
import stamprender

SR = common.SR
SECS = 2
HALF_DB = 20 * math.log10(0.5)


def speechlike(n, seed, level=0.1):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, n)
    env = 0.2 + 0.8 * (0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t)).pow(2)
    return torch.randn(n, generator=g) * env * level


class _Spk:
    """Stands in for the WavLM speaker embedder.

    Its similarity is the cosine similarity of the raw waveforms.
    """

    def sim(self, a, b, key_b=None):
        n = min(len(a), len(b))
        return float(torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0))


class _Vad:
    """Stands in for Silero: the whole clip is one speech segment."""

    def segments(self, wav48):
        w16 = torchaudio.functional.resample(wav48, SR, 16000)
        return w16, [{"start": 0, "end": len(w16)}]


def _write(path, x):
    sf.write(str(path), x.numpy(), SR, subtype="FLOAT")


@pytest.fixture
def tiny_set(tmp_path, monkeypatch):
    """A four-condition set: clean, babble@+4, expressive_tail@-5 and real_x.

    The expressive_tail@-5 items have a score_from.
    """
    root, out_root = tmp_path / "set", tmp_path / "out"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    n = SECS * SR
    rows, seed = [], 0
    for cond, snr, count in (
        ("clean", None, 3),
        ("babble@+4", 4.0, 3),
        ("expressive_tail@-5", -5.0, 2),
        ("real_x", 0.0, 2),
    ):
        for i in range(count):
            seed += 1
            clean = speechlike(n, seed)
            noise = (torch.randn(
                n, generator=torch.Generator().manual_seed(1000 + seed)) * 0.05)
            if snr is None:
                mix, ref = clean, clean
            else:
                mix, ref = common.mix_at_snr(clean, noise, snr)
            rid = f"{cond.replace('@', 'AT')}_{i:03d}.wav"
            _write(root / "mix" / rid, mix)
            row = {
                "id": rid,
                "condition": cond,
                "mix_path": f"mix/{rid}",
                "clean_path": None,
            }
            if cond != "real_x":
                _write(root / "ref" / rid, ref)
                row["clean_path"] = f"ref/{rid}"
            if cond.endswith("_tail@-5"):
                row["score_from"] = n // 2
            rows.append(row)
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
    monkeypatch.setattr(scorebench, "NISQA_ON", False)
    return common.manifest(str(root)), out_root


def render(rows, out_root, name, fn):
    """Writes fn(row, mix, ref) for every row as system name.

    fn may return None to skip.
    """
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        mix = common.load48(r["mix_path"])
        ref = common.load48(r["clean_path"]) if r.get("clean_path") else None
        y = fn(r, mix, ref)
        if y is not None:
            _write(d / r["id"], y)


def conditions(metrics):
    return {k for k in metrics if not k.startswith("_")}


def test_unprocessed_input_scores_as_the_identity_system(tiny_set):
    rows, out_root = tiny_set
    render(rows, out_root, "INPUT", lambda r, mix, ref: mix)
    metrics = scorebench.score_system("INPUT", rows, _Spk(), _Vad(), None)
    assert metrics["_set_id"] == "tiny"
    assert metrics["_level_cal_db"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["_bands_hz"] == [list(b) for b in common.BANDS_HZ]
    assert conditions(metrics) == {
        "clean",
        "babble@+4",
        "expressive_tail@-5",
        "real_x",
    }
    clean = metrics["clean"]
    babble = metrics["babble@+4"]
    real = metrics["real_x"]
    assert (clean["n"] == 3 and clean["si_sdr"] > 60 and
            clean["spk_sim"] == pytest.approx(1.0, abs=1e-4))
    assert clean["kept_mean"] == pytest.approx(0.0, abs=1e-6)
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in clean["band_kept_db"])
    assert clean["lag_median"] == 0 and clean["lag_max_abs"] == 0
    assert babble["babble_supp_db"] == pytest.approx(0.0, abs=1e-6)
    assert babble["babble_supp_db_nb"] == pytest.approx(0.0, abs=1e-3)
    # the mix sits at its nominal SNR
    assert babble["si_sdr"] == pytest.approx(4.0, abs=0.3)
    assert babble["si_sdr_nb"] == pytest.approx(4.0, abs=0.6)
    # cosine of a +4 dB mix against its clean
    assert 0.8 < babble["spk_sim"] < 0.9
    assert (real["n"] == 2 and real["n_segments"] == 2 and
            real["deleted_rate_pct"] == 0.0)
    assert real["median_drop_db"] == pytest.approx(0.0, abs=1e-6)
    assert real["speech_rms_delta_db"] == pytest.approx(0.0, abs=1e-6)
    # the stub VAD leaves no unvoiced samples
    assert math.isnan(real["nonvoiced_supp_db"])
    assert len(real["band_vs_in_db"]) == 7 and all(
        abs(v) < 1e-6 for v in real["band_vs_in_db"])
    assert "dnsmos_ovr" not in real and "nisqa_mos" not in real
    assert "babble_supp_db" not in clean


def test_a_global_gain_shows_up_in_level_calibration_and_every_kept_metric(
    tiny_set,):
    rows, out_root = tiny_set
    render(rows, out_root, "half", lambda r, mix, ref: 0.5 * mix)
    metrics = scorebench.score_system("half", rows, _Spk(), _Vad(), None)
    assert metrics["_level_cal_db"] == pytest.approx(HALF_DB, abs=0.02)
    clean, real = metrics["clean"], metrics["real_x"]
    assert clean["out_in_rms_db"] == pytest.approx(HALF_DB, abs=0.02)
    assert clean["kept_mean"] == pytest.approx(HALF_DB, abs=0.02)
    assert clean["kept_p10"] == pytest.approx(HALF_DB, abs=0.02)
    # a gain, not more voice
    assert clean["kept_coh_p10"] == pytest.approx(HALF_DB, abs=0.05)
    assert clean["kept_coh_nb_p10"] == pytest.approx(HALF_DB, abs=0.05)
    for q in ("kept_q1", "kept_q2", "kept_q3"):
        assert clean[q] == pytest.approx(HALF_DB, abs=0.05)
    assert all(
        v == pytest.approx(HALF_DB, abs=0.05) for v in clean["band_kept_db"])
    assert clean["si_sdr"] > 60 and clean["spk_sim"] == pytest.approx(
        1.0,
        # scale-invariant
        abs=1e-4,
    )
    assert real["speech_rms_delta_db"] == pytest.approx(HALF_DB, abs=0.02)
    # -6 dB is a level change, not a deletion
    assert real["deleted_rate_pct"] == 0.0


def test_score_from_restricts_reference_metrics_to_the_tail(tiny_set):
    rows, out_root = tiny_set

    def tail_only(r, mix, ref):
        if r["condition"] != "expressive_tail@-5":
            return mix
        o = ref.clone()
        s = r["score_from"]
        # wrong before
        o[:s] = torch.randn(s, generator=torch.Generator().manual_seed(7)) * 0.1
        return o

    render(rows, out_root, "tailonly", tail_only)
    metrics = scorebench.score_system("tailonly", rows, _Spk(), _Vad(), None)
    tail = metrics["expressive_tail@-5"]
    assert (tail["n"] == 2 and tail["si_sdr"] > 60 and
            tail["kept_mean"] == pytest.approx(0, abs=1e-6))
    assert tail["lag_median"] == 0
    ignoring = scorebench.score_system("tailonly",
                                       [dict(r, score_from=0) for r in rows],
                                       _Spk(), _Vad(), None)
    # the wrong first half would count
    assert ignoring["expressive_tail@-5"]["si_sdr"] < 15


def test_algorithmic_delay_is_measured_and_undone(tiny_set):
    rows, out_root = tiny_set
    lag = 480  # 10 ms late

    def late(r, mix, ref):
        return torch.cat([torch.zeros(lag), mix])[:len(mix)]

    render(rows, out_root, "late", late)
    metrics = scorebench.score_system("late", rows, _Spk(), _Vad(), None)
    assert metrics["clean"]["lag_median"] == lag
    assert metrics["clean"]["lag_max_abs"] == lag
    # aligned before scoring: only the 10 ms the delay pushed out of the clip is
    # lost
    assert metrics["clean"]["si_sdr"] > 20
    first = next(r for r in rows if r["condition"] == "clean")
    unaligned = common.si_sdr(
        common.load48(str(out_root / "late" / first["id"])),
        common.load48(first["clean_path"]),
    )
    # what the same output would score without the alignment
    assert unaligned < 5
    assert metrics["babble@+4"]["babble_supp_db"] == pytest.approx(0.0, abs=0.2)


def test_a_polarity_flip_is_undone_like_a_delay_and_never_reads_as_damage(
    tiny_set,):
    rows, out_root = tiny_set
    lag = 240

    def upside_down_and_late(r, mix, ref):
        return -torch.cat([torch.zeros(lag), mix])[:len(mix)]

    render(rows, out_root, "flipped", upside_down_and_late)
    metrics = scorebench.score_system("flipped", rows, _Spk(), _Vad(), None)
    clean = metrics["clean"]
    assert (clean["lag_median"] == lag and
            clean["n_polarity_flipped"] == clean["n"])
    assert clean["si_sdr"] > 20 and clean["kept_coh_p10"] == pytest.approx(
        0.0, abs=0.1)
    assert clean["kept_coh_lost_pct"] == 0.0
    assert metrics["babble@+4"]["babble_supp_db"] == pytest.approx(0.0, abs=0.2)
    render(rows, out_root, "upright", lambda r, mix, ref: mix)
    upright = scorebench.score_system("upright", rows, _Spk(), _Vad(), None)
    assert upright["clean"]["n_polarity_flipped"] == 0


def test_silence_cannot_calibrate_its_deleted_voice_back(tiny_set):
    rows, out_root = tiny_set
    render(rows, out_root, "mute", lambda r, mix, ref: torch.zeros_like(mix))
    metrics = scorebench.score_system("mute", rows, _Spk(), _Vad(), None)
    # bounded: a gain convention, not an output
    assert metrics["_level_cal_db"] == -common.MAX_LEVEL_CAL_DB
    assert metrics["clean"]["kept_coh_p10"] - metrics["_level_cal_db"] < -40
    assert metrics["clean"]["kept_coh_lost_pct"] == 100.0


def test_lost_speech_blocks_are_counted_where_a_percentile_is_blind(tiny_set):
    rows, out_root = tiny_set

    def one_gap(r, mix, ref):  # 300 ms of the loudest second goes missing
        y = mix.clone()
        if ref is not None:
            start = int(ref.abs().argmax()) // 4800 * 4800
            y[start:start + 3 * 4800] = 0.0
        return y

    render(rows, out_root, "gap", one_gap)
    clean = scorebench.score_system("gap", rows, _Spk(), _Vad(), None)["clean"]
    assert clean["kept_coh_lost_pct"] > 0.0
    # -12 dB everywhere is a gain: calibrated away, nothing lost
    render(rows, out_root, "quieter", lambda r, mix, ref: 0.25 * mix)
    quieter = scorebench.score_system("quieter", rows, _Spk(), _Vad(), None)
    assert quieter["clean"]["kept_coh_lost_pct"] == 0.0


def test_deletion_is_counted_on_reference_free_items(tiny_set):
    rows, out_root = tiny_set
    render(
        rows,
        out_root,
        "muter",
        lambda r, mix, ref: 0.01 * mix if r["condition"] == "real_x" else mix,
    )
    metrics = scorebench.score_system("muter", rows, _Spk(), _Vad(), None)
    assert metrics["real_x"]["deleted_rate_pct"] == 100.0
    assert metrics["real_x"]["median_drop_db"] == pytest.approx(-40.0, abs=0.1)
    assert metrics["clean"]["si_sdr"] > 60  # untouched elsewhere


def test_conditions_without_outputs_are_left_out(tiny_set):
    rows, out_root = tiny_set
    render(
        rows,
        out_root,
        "partial",
        lambda r, mix, ref: mix if r["condition"] == "clean" else None,
    )
    metrics = scorebench.score_system("partial", rows, _Spk(), _Vad(), None)
    assert conditions(metrics) == {"clean"} and metrics["clean"]["n"] == 3


def test_rms_db_of_a_known_sine():
    t = torch.arange(SR) / SR
    tone = 0.5 * torch.sin(2 * math.pi * 440 * t)
    assert scorebench.rms_db(tone) == pytest.approx(
        20 * math.log10(0.5 / math.sqrt(2)), abs=0.01)
    assert scorebench.rms_db(torch.zeros(1000)) < -200


def test_systems_without_an_identity_on_this_set_are_not_scored(
        tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", str(tmp_path))
    monkeypatch.setattr(common, "set_id", lambda root=None: "current-set")
    (tmp_path / "old.meta.json").write_text(
        json.dumps({"set_id": "previous-set"}))
    (tmp_path / "same.meta.json").write_text(
        json.dumps({"set_id": "current-set"}))
    (tmp_path / "unstamped.meta.json").write_text(json.dumps({"mechanism": "x"
                                                             }))
    assert "previous-set"[:12] in scorebench.render_problem("old")
    assert scorebench.render_problem("same") is None
    assert "no set_id" in scorebench.render_problem("unstamped")
    assert "meta.json missing" in scorebench.render_problem(
        "missing")  # a bare output directory is not scored


def test_retention_items_report_kept_speech_and_bed_residual_but_no_si_sdr(
        tmp_path, monkeypatch):
    root, out_root = tmp_path / "wildset", tmp_path / "out"
    (root / "mix").mkdir(parents=True)
    (root / "ref").mkdir()
    n, rows = SECS * SR, []

    # speech in the first half of the recording, its own floor after it
    class _HalfVad(_Vad):

        def segments(self, wav48):
            w16 = torchaudio.functional.resample(wav48, SR, 16000)
            return w16, [{"start": 0, "end": len(w16) // 2}]

    vad = _HalfVad()
    for i in range(2):
        wild = speechlike(n, 50 + i) + 0.05 * torch.randn(
            n, generator=torch.Generator().manual_seed(i))
        wild[n // 2:] = 0.02 * torch.randn(
            n - n // 2, generator=torch.Generator().manual_seed(7 + i))
        bed = (
            torch.randn(n, generator=torch.Generator().manual_seed(900 + i)) *
            0.05)
        mix, ref = common.mix_at_snr(wild, bed, -5.0)
        rid = f"wildAT-5_{i:03d}.wav"
        _write(root / "mix" / rid, mix)
        _write(root / "ref" / rid, ref)
        rows.append({
            "id": rid,
            "condition": "wild@-5",
            "family": "wild",
            "mix_path": f"mix/{rid}",
            "clean_path": f"ref/{rid}",
            "snr": -5.0,
            "score_policy": "retention",
        })
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump({
        "set_id": "wildtiny",
        "n_items": 2
    }, open(root / "checksums.json", "w", encoding="utf-8"))
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(out_root))
    monkeypatch.setattr(scorebench, "NISQA_ON", False)
    rows = common.manifest(str(root))

    def wild_scores(name):
        return scorebench.score_system(name, rows, _Spk(), vad, None)["wild@-5"]

    render(rows, out_root, "perfect", lambda r, mix, ref: ref)
    metrics = wild_scores("perfect")
    assert (metrics["score_policy"] == "retention" and
            "si_sdr" not in metrics and "si_sdr_nb" not in metrics)
    # voice kept, bed gone
    assert abs(
        metrics["kept_coh_p10"]) < 0.5 and metrics["nonvoiced_supp_db"] < -10
    render(rows, out_root, "muter", lambda r, mix, ref: 0.01 * ref)
    muted = wild_scores("muter")
    # deleted speech shows up here
    assert muted["kept_coh_p10"] == pytest.approx(-40.0, abs=0.5)
    # returns the added bed alone
    render(rows, out_root, "bedonly", lambda r, mix, ref: mix - ref)
    bed_only = wild_scores("bedonly")
    # coherent: deleted; energy: fooled
    assert bed_only["kept_coh_p10"] < -20 and bed_only["kept_p10"] > 0
    # and the bed is still there
    assert bed_only["nonvoiced_supp_db"] > metrics["nonvoiced_supp_db"] + 10

    # the quieter half of the speech blocks zeroed
    def quiet_deleted(r, mix, ref):
        y = ref.clone()
        blocks = y[:len(y) // 4800 * 4800].reshape(-1, 4800)
        e = blocks.pow(2).mean(1)
        speech = torch.zeros(len(e), dtype=torch.bool)
        # the VAD stub marks the first half as speech
        speech[:len(e) // 2] = True
        order = e.clone()
        order[~speech] = float("inf")
        idx = order.argsort()[:int(speech.sum()) // 2]
        blocks[idx] = 0
        return y

    render(rows, out_root, "quietgone", quiet_deleted)
    quiet_gone = scorebench.score_system("quietgone", rows, _Spk(), vad, None)
    # deleted quiet speech is seen, the energy ratio barely moves
    assert quiet_gone["wild@-5"]["kept_coh_p10"] < -30


def test_missing_outputs_are_counted_per_condition_and_for_the_set(tiny_set):
    rows, out_root = tiny_set
    seen = []

    def two_of_three_clean(r, mix, ref):
        if r["condition"] != "clean":
            return None
        seen.append(r["id"])
        return mix if len(seen) < 3 else None

    render(rows, out_root, "flaky", two_of_three_clean)
    metrics = scorebench.score_system("flaky", rows, _Spk(), _Vad(), None)
    assert conditions(metrics) == {"clean"}
    assert metrics["clean"]["n"] == 2 and metrics["clean"]["n_missing"] == 1
    # the other conditions never rendered at all
    assert metrics["_n_missing"] == len(rows) - 2
    render(rows, out_root, "whole", lambda r, mix, ref: mix)
    whole = scorebench.score_system("whole", rows, _Spk(), _Vad(), None)
    assert whole["_n_missing"] == 0


def test_observations_reproduce_condition_reducers_and_level_calibration(
    tiny_set,):
    rows, out_root = tiny_set
    render(
        rows,
        out_root,
        "varied",
        lambda r, mix, ref: mix * (0.2 + int(r["id"][-7:-4]) * 0.1),
    )
    records = []
    scores = scorebench.score_system("varied",
                                     rows,
                                     _Spk(),
                                     _Vad(),
                                     None,
                                     records=records)
    assert len(records) == len(rows)
    for condition in conditions(scores):
        raw = {}
        for r in records:
            if r["condition"] == condition:
                for key, values in r["measurements"].items():
                    raw.setdefault(key, []).extend(values)
        cell = scores[condition]
        assert cell["out_in_rms_db"] == np.mean(raw["out_in_rms_db"])
        if "kept" in raw:
            assert cell["kept_mean"] == np.mean(raw["kept"])
            assert cell["kept_p10"] == np.percentile(raw["kept"], 10)
            assert cell["lag_median"] == np.median(raw["lag"])
            assert cell["band_kept_db"] == pytest.approx(
                np.nanmean(np.array(raw["band"], dtype=float), axis=0))
            assert cell["kept_coh_p10"] == np.mean(raw["kept_coh_p10"])
        else:
            assert cell["n_segments"] == len(raw["seg_drop"])
            assert cell["deleted_rate_pct"] == 100 * np.mean(
                levels.deleted(raw["seg_drop"], scores["_level_cal_db"]))
    clean_gains = [
        r["metrics"]["out_in_rms_db"]
        for r in records
        if r["condition"] == "clean"
    ]
    assert scores["_level_cal_db"] == np.median(clean_gains)


def test_invalid_audio_statuses_remain_in_per_item_inventory(tiny_set):
    rows, out_root = tiny_set
    render(rows, out_root, "broken", lambda r, mix, ref: mix)
    directory = out_root / "broken"
    (directory / rows[0]["id"]).write_bytes(b"invalid wav")
    (directory / rows[1]["id"]).unlink()
    _write(directory / rows[2]["id"], torch.zeros(12))
    _write(
        directory / rows[3]["id"],
        torch.full_like(common.load48(rows[3]["mix_path"]), float("nan")),
    )
    records = []
    scores = scorebench.score_system("broken",
                                     rows,
                                     _Spk(),
                                     _Vad(),
                                     None,
                                     records=records)
    statuses = {r["id"]: r["output_status"] for r in records}
    assert [statuses[r["id"]] for r in rows[:4]] == [
        "unreadable",
        "missing",
        "wrong_length",
        "nonfinite",
    ]
    assert scores["_n_missing"] == 4


def _same(a, b):
    """Equality that treats nan as equal to nan.

    A condition with no nonvoiced frames reads nan.
    """

    def canonical(x):
        return json.dumps(observations.finite(x), sort_keys=True)

    return canonical(a) == canonical(b)


def _records_file(path, records, system="varied", scorer=None):
    observations.write(
        path,
        records,
        {
            "system": system,
            "set_id": "tiny",
            "subset": "all",
            "anchors_version": "t",
            "anchors_sha256": "t",
            "scorer_sha256": scorer or scorebench.SCORER_SHA256,
            "n_expected": len(records),
            "sealed": False,
            "render_identity": {},
            "level_cal_db": 0.0,
        },
    )


def test_records_carry_a_content_identity_and_are_reused_only_while_it_holds(
        tiny_set, tmp_path):
    rows, out_root = tiny_set
    render(
        rows,
        out_root,
        "varied",
        lambda r, mix, ref: mix * (0.2 + int(r["id"][-7:-4]) * 0.1),
    )
    first = []
    scores = scorebench.score_system("varied",
                                     rows,
                                     _Spk(),
                                     _Vad(),
                                     None,
                                     records=first)
    assert all(r["identity"]["output"] and r["identity"]["mix"] for r in first)
    assert all((r["identity"]["ref"] is None) == (r["condition"] == "real_x")
               for r in first)
    _records_file(tmp_path / "varied.items.jsonl", first)
    index, why = scorebench.reuse_index(str(tmp_path / "varied.items.jsonl"),
                                        "varied")
    assert len(index) == len(rows) and "reusable" in why
    # nothing changed: every record is reused and the file is reproduced
    again = []
    assert _same(
        scorebench.score_system("varied",
                                rows,
                                _Spk(),
                                _Vad(),
                                None,
                                records=again,
                                reuse=index),
        scores,
    )
    assert [r["metrics"] for r in again] == [r["metrics"] for r in first]
    # one output changes: that item is scored afresh, the others keep their
    # records
    changed = rows[1]
    _write(out_root / "varied" / changed["id"],
           common.load48(changed["mix_path"]) * 0.5)
    third = []
    rescored = scorebench.score_system("varied",
                                       rows,
                                       _Spk(),
                                       _Vad(),
                                       None,
                                       records=third,
                                       reuse=index)
    by_id = {r["id"]: r for r in third}
    before = {r["id"]: r for r in first}[changed["id"]]
    after = by_id[changed["id"]]
    assert after["identity"]["output"] != before["identity"]["output"]
    assert after["metrics"] != before["metrics"]
    for r in first:
        if r["id"] != changed["id"]:
            assert by_id[r["id"]]["metrics"] == r["metrics"]
    assert (rescored["_level_cal_db"] != scores["_level_cal_db"] or
            changed["condition"] != "clean")
    # another scorer's records are never reused
    _records_file(tmp_path / "other.items.jsonl", first, scorer="someone-else")
    index, why = scorebench.reuse_index(str(tmp_path / "other.items.jsonl"),
                                        "varied")
    assert index == {} and "another scorer" in why


def test_a_view_aggregated_from_whole_set_records_equals_scoring_it_directly(
        tiny_set, monkeypatch):
    rows, out_root = tiny_set
    render(
        rows,
        out_root,
        "varied",
        lambda r, mix, ref: mix * (0.2 + int(r["id"][-7:-4]) * 0.1),
    )
    whole = []
    scorebench.score_system("varied", rows, _Spk(), _Vad(), None, records=whole)
    view = [
        r for i, r in enumerate(rows) if i % 2 == 0 or r["condition"] == "clean"
    ]
    assert 0 < len(view) < len(rows)
    direct = scorebench.score_system("varied", view, _Spk(), _Vad(), None)
    ids = {r["id"] for r in view}
    derived = scorebench.aggregate([r for r in whole if r["id"] in ids], view)
    # bit for bit, the lost-block share under the view's calibration
    assert _same(derived, direct)
    whole_cal = scorebench.aggregate(whole, rows)["_level_cal_db"]
    n_clean = len([r for r in rows if r["condition"] == "clean"])
    n_clean_view = len([r for r in view if r["condition"] == "clean"])
    assert derived["_level_cal_db"] != whole_cal or n_clean == n_clean_view


def test_aggregate_only_derives_a_subset_from_whole_set_records(
        tiny_set, tmp_path, monkeypatch):
    rows, out_root = tiny_set
    render(rows, out_root, "varied", lambda r, mix, ref: mix * 0.7)
    whole_dir, view_dir = tmp_path / "whole", tmp_path / "view"
    whole_dir.mkdir()
    # the stamp's digest of the outputs, which the scorer checks the bytes
    # against
    rendered = stamprender.outputs_digest([r["id"] for r in rows],
                                          str(out_root / "varied"))
    (whole_dir / "varied.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "varied",
            "outputs_sha256": rendered
        }))
    (whole_dir / "INPUT.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "INPUT"
        }))
    monkeypatch.setattr(common, "RESULTS", str(whole_dir))
    monkeypatch.setattr(common,
                        "SpeakerEmbedder",
                        lambda cache_dir=None: _Spk())
    monkeypatch.setattr(common, "Vad", _Vad)
    monkeypatch.setattr(sys, "argv", ["scorebench.py", "--systems", "varied"])
    scorebench.main()
    assert (whole_dir / "varied.items.jsonl").exists()
    assert (whole_dir / "varied.metrics.json").exists()
    with open(whole_dir / "varied.metrics.json", encoding="utf-8") as fh:
        whole_metrics = json.load(fh)
    assert whole_metrics["_outputs_sha256"] == rendered
    # the view: a panel of three items, selected the way common.manifest selects
    # one
    members = {rows[0]["id"], rows[3]["id"], rows[-1]["id"]}
    monkeypatch.setattr(common, "SUBSET", "gate")
    monkeypatch.setattr(common,
                        "panel_members",
                        lambda name, root=None: members)
    monkeypatch.setattr(common, "RESULTS", str(view_dir))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scorebench.py",
            "--systems",
            "varied",
            "--aggregate-only",
            str(whole_dir),
        ],
    )
    scorebench.main()
    derived = json.load(open(view_dir / "varied.metrics.json",
                             encoding="utf-8"))
    assert (derived["_subset"] == "gate" and sum(
        v["n"] for k, v in derived.items() if not k.startswith("_")) == 3)
    # the view names the render its records read, with no stamp beside it
    assert derived["_outputs_sha256"] == rendered
    stamp = (view_dir / "varied.metrics.json").stat().st_mtime_ns
    # a view is re-derived every time, never skipped as "already scored"
    scorebench.main()
    assert (view_dir / "varied.metrics.json").stat().st_mtime_ns > stamp
    # the whole render's stamp cannot vouch for three of its outputs scored
    # again; a render made and stamped on the panel's items can
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    whole_stamp = {
        "set_id": "tiny",
        "name": "varied",
        "outputs_sha256": rendered,
        "n_files": len(rows),
    }
    (direct_dir / "varied.meta.json").write_text(json.dumps(whole_stamp))
    monkeypatch.setattr(common, "RESULTS", str(direct_dir))
    monkeypatch.setattr(sys, "argv", ["scorebench.py", "--systems", "varied"])
    with pytest.raises(SystemExit, match="derived from its whole-set records"):
        scorebench.main()
    panel_stamp = dict(whole_stamp,
                       n_files=len(members),
                       outputs_sha256=stamprender.outputs_digest(
                           sorted(members), str(out_root / "varied")))
    (direct_dir / "varied.meta.json").write_text(json.dumps(panel_stamp))
    scorebench.main()
    with open(direct_dir / "varied.metrics.json", encoding="utf-8") as fh:
        direct = json.load(fh)
    # each names the render it read; apart from that the two are one
    assert direct["_outputs_sha256"] == panel_stamp["outputs_sha256"]
    direct["_outputs_sha256"] = derived["_outputs_sha256"]
    assert _same(direct, derived)
    header = json.loads(
        open(view_dir / "varied.items.jsonl", encoding="utf-8").readline())
    assert header["subset"] == "gate" and header["n_expected"] == 3


def test_a_system_is_never_skipped_as_scored(tiny_set, tmp_path, monkeypatch,
                                             capsys):
    """A changed output is re-scored and an unchanged one keeps its record.

    A metrics file that already exists is never a reason to skip.
    """
    rows, out_root = tiny_set
    render(rows, out_root, "varied", lambda r, mix, ref: mix * 0.7)
    results = tmp_path / "results"
    results.mkdir()
    (results / "varied.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "varied"
        }))
    (results / "INPUT.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "INPUT"
        }))
    monkeypatch.setattr(common, "RESULTS", str(results))
    monkeypatch.setattr(common,
                        "SpeakerEmbedder",
                        lambda cache_dir=None: _Spk())
    monkeypatch.setattr(common, "Vad", _Vad)
    monkeypatch.setattr(sys, "argv", ["scorebench.py", "--systems", "varied"])
    scorebench.main()
    with open(results / "varied.items.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    first = {r["id"]: r for r in records[1:]}
    changed = rows[2]
    _write(out_root / "varied" / changed["id"],
           common.load48(changed["mix_path"]) * 0.3)
    capsys.readouterr()
    scorebench.main()
    out = capsys.readouterr().out
    assert ("already scored" not in out and "reused" in out and
            "scored 1 items" in out)
    with open(results / "varied.items.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    second = {r["id"]: r for r in records[1:]}
    assert second[changed["id"]]["metrics"] != first[changed["id"]]["metrics"]
    assert all(second[i]["metrics"] == first[i]["metrics"]
               for i in first
               if i != changed["id"])


def test_scores_are_accepted_only_of_the_bytes_their_stamp_vouches_for(
        tiny_set, tmp_path, monkeypatch):
    """A render changed since its stamp is never scored under that stamp."""
    rows, out_root = tiny_set
    render(rows, out_root, "stamped", lambda r, mix, ref: mix * 0.7)
    results = tmp_path / "results"
    results.mkdir()
    digest = stamprender.outputs_digest([r["id"] for r in rows],
                                        str(out_root / "stamped"))
    stamp = {"set_id": "tiny", "name": "stamped", "outputs_sha256": digest}
    (results / "stamped.meta.json").write_text(json.dumps(stamp))
    monkeypatch.setattr(common, "RESULTS", str(results))
    monkeypatch.setattr(common,
                        "SpeakerEmbedder",
                        lambda cache_dir=None: _Spk())
    monkeypatch.setattr(common, "Vad", _Vad)
    monkeypatch.setattr(sys, "argv", ["scorebench.py", "--systems", "stamped"])
    scorebench.main()
    scores = json.loads((results / "stamped.metrics.json").read_text())
    assert scores["_outputs_sha256"] == digest
    _, records = observations.read(results / "stamped.items.jsonl")
    hashes = {r["id"]: r["output_sha256"] for r in records}
    assert stamprender.digest_of(hashes) == digest
    # one output's samples change after the stamp: the scores it vouched for
    # stay, and nothing is written under it
    changed = rows[2]
    _write(out_root / "stamped" / changed["id"],
           common.load48(changed["mix_path"]) * 0.3)
    with pytest.raises(SystemExit, match="not the render its stamp vouches"):
        scorebench.main()
    assert json.loads((results / "stamped.metrics.json").read_text()) == scores
    # a filtered look reads too few outputs to vouch for the render
    monkeypatch.setattr(
        sys, "argv",
        ["scorebench.py", "--systems", "stamped", "--conds", "clean"])
    scorebench.main()
    looked = json.loads((results / "stamped.metrics.json").read_text())
    assert looked["_selection"] and looked["_outputs_sha256"] is None


def test_an_output_gain_is_undone_everywhere_but_the_level_row(tiny_set):
    """A perfect output made 6 dB louder reads as perfect but for its level.

    It reads as the reference does on every row a gain is a convention for, and
    as 6 dB on the one row that scores the gain.
    """
    rows, out_root = tiny_set
    render(
        rows,
        out_root,
        "REF2",
        lambda r, mix, ref: 2 * (ref if ref is not None else mix),
    )
    render(
        rows,
        out_root,
        "REF1",
        lambda r, mix, ref: ref if ref is not None else mix,
    )
    loud = scorebench.score_system("REF2", rows, _Spk(), _Vad(), None)
    plain = scorebench.score_system("REF1", rows, _Spk(), _Vad(), None)
    six = 20 * math.log10(2)
    assert loud["_level_cal_db"] == pytest.approx(six, abs=1e-4)
    assert loud["clean"]["level_err_db"] == pytest.approx(six, abs=1e-4)
    assert plain["clean"]["level_err_db"] == pytest.approx(0.0, abs=1e-4)
    assert loud["clean"]["hf_err_db"] == pytest.approx(
        plain["clean"]["hf_err_db"], abs=1e-3)
    assert loud["babble@+4"]["babble_supp_db"] == pytest.approx(
        plain["babble@+4"]["babble_supp_db"], abs=1.0)
    assert (loud["babble@+4"]["babble_supp_raw_db"] > -10 >
            plain["babble@+4"]["babble_supp_db"])
    assert (loud["real_x"]["deleted_rate_pct"] ==
            plain["real_x"]["deleted_rate_pct"] == 0)
    assert loud["real_x"]["median_drop_db"] == pytest.approx(0.0, abs=1e-3)
    a = loud["real_x"]["nonvoiced_supp_db"]
    b = plain["real_x"]["nonvoiced_supp_db"]
    assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-3)


SPEECH_S = ((0.3, 0.6), (0.9, 1.9))  # a 0.3 s word, then a 1 s phrase


class _WordVad(_Vad):
    """Stands in for Silero: the two words of _real_set, the first short."""

    def segments(self, wav48):
        w16 = torchaudio.functional.resample(wav48, SR, 16000)
        return w16, [{
            "start": int(a * 16000),
            "end": int(b * 16000)
        } for a, b in SPEECH_S]


def _real_set(tmp_path, monkeypatch, floor, tone_hz=None):
    """One reference-free recording: two words over a floor of RMS ``floor``,
    white, or a tone at ``tone_hz`` (a white floor loses two thirds of its
    power in the scorer's 16 kHz reading; a tone keeps it)."""
    root, out_root = tmp_path / "realset", tmp_path / "out"
    (root / "mix").mkdir(parents=True)
    n = SECS * SR
    if tone_hz is None:
        mix = torch.randn(n,
                          generator=torch.Generator().manual_seed(11)) * floor
    else:
        t = torch.arange(n, dtype=torch.float64) / SR
        mix = (math.sqrt(2.0) * floor *
               torch.sin(2 * math.pi * tone_hz * t)).float()
    for a, b in SPEECH_S:
        a, b = int(a * SR), int(b * SR)
        mix[a:b] += speechlike(b - a, 5)
    _write(root / "mix" / "real_x_000.wav", mix)
    rows = [{
        "id": "real_x_000.wav",
        "condition": "real_x",
        "mix_path": "mix/real_x_000.wav",
        "clean_path": None,
    }]
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump({
        "set_id": "realtiny",
        "n_items": 1
    }, open(root / "checksums.json", "w", encoding="utf-8"))
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(out_root))
    monkeypatch.setattr(scorebench, "NISQA_ON", False)
    return common.manifest(str(root)), out_root


def _real_scores(rows, name):
    return scorebench.score_system(name, rows, _Spk(), _WordVad(),
                                   None)["real_x"]


def test_a_short_word_is_speech_and_never_a_floor_to_suppress(
        tmp_path, monkeypatch):
    """Every detected segment is speech; only the deletion rule needs half a
    second of it."""
    rows, out_root = _real_set(tmp_path, monkeypatch, floor=0.01)
    render(rows, out_root, "same", lambda r, mix, ref: mix)

    def drop_word(r, mix, ref):
        y = mix.clone()
        a, b = (int(t * SR) for t in SPEECH_S[0])
        y[a:b] = 0.0
        return y

    render(rows, out_root, "dropword", drop_word)

    same, dropped = _real_scores(rows, "same"), _real_scores(rows, "dropword")

    assert same["nonvoiced_supp_db"] == pytest.approx(0.0, abs=1e-6)
    # the word is speech: deleting it is not cleaning the floor
    assert dropped["nonvoiced_supp_db"] == pytest.approx(0.0, abs=0.3)
    # and the deletion rule reads the phrase alone, as before
    assert same["n_segments"] == 1


def test_the_non_voiced_floor_is_read_by_frame_with_the_delay_undone(
        tmp_path, monkeypatch):
    """A late passthrough and a quiet floor are not suppression; a halved floor
    is 6 dB of it."""
    rows, out_root = _real_set(tmp_path, monkeypatch, floor=10**(-65 / 20))
    render(rows, out_root, "late",
           lambda r, mix, ref: torch.cat([torch.zeros(480), mix[:-480]]))

    def halved(r, mix, ref):
        y = mix * 0.5
        for a, b in SPEECH_S:
            a, b = int(a * SR), int(b * SR)
            y[a:b] = mix[a:b]
        return y

    render(rows, out_root, "halved", halved)

    late, half = _real_scores(rows, "late"), _real_scores(rows, "halved")

    assert abs(late["nonvoiced_supp_db"]) < 0.5
    # the raw row: the published one undoes the system's gain, which for this
    # system is the halving itself
    assert half["nonvoiced_supp_raw_db"] == pytest.approx(-6.02, abs=0.4)


def test_a_filtered_pass_leaves_the_whole_set_files_of_input_alone(
        tiny_set, tmp_path, monkeypatch):
    rows, out_root = tiny_set
    render(rows, out_root, "sys", lambda r, mix, ref: mix * 0.7)
    results = tmp_path / "results"
    results.mkdir()
    digest = stamprender.outputs_digest([r["id"] for r in rows],
                                        str(out_root / "sys"))
    (results / "sys.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "sys",
            "outputs_sha256": digest
        }))
    # the unprocessed input's identity, as runsystem.py --input writes it
    (results / "INPUT.meta.json").write_text(
        json.dumps({
            "set_id": "tiny",
            "name": "INPUT",
            "mechanism": "unprocessed input"
        }))
    monkeypatch.setattr(common, "RESULTS", str(results))
    monkeypatch.setattr(common,
                        "SpeakerEmbedder",
                        lambda cache_dir=None: _Spk())
    monkeypatch.setattr(common, "Vad", _Vad)
    monkeypatch.setattr(sys, "argv", ["scorebench.py", "--systems", "sys"])
    scorebench.main()
    whole = (results / "INPUT.metrics.json").read_text()
    records = (results / "INPUT.items.jsonl").read_text()

    monkeypatch.setattr(
        sys, "argv", ["scorebench.py", "--systems", "sys", "--conds", "clean"])
    scorebench.main()

    assert (results / "INPUT.metrics.json").read_text() == whole
    assert (results / "INPUT.items.jsonl").read_text() == records
    looked = json.loads((results / "sys.metrics.json").read_text())
    assert looked["_selection"] == {"conds": "clean", "limit": 0}

    # the discovered INPUT no more than a named one: a filtered pass over
    # every output directory leaves it alone too
    monkeypatch.setattr(sys, "argv", ["scorebench.py", "--conds", "clean"])
    scorebench.main()

    assert (results / "INPUT.metrics.json").read_text() == whole
    assert (results / "INPUT.items.jsonl").read_text() == records


def test_strong_suppression_of_a_quiet_floor_is_read_in_full(
        tmp_path, monkeypatch):
    """A -78 dBFS floor attenuated by 40 dB reads -40 dB: the log's guard sits
    far below the floors the row admits (a 1e-9 guard read 12 dB here)."""
    rows, out_root = _real_set(tmp_path,
                               monkeypatch,
                               floor=10**(-78 / 20),
                               tone_hz=700)

    def cleaned(r, mix, ref):
        y = mix * 0.01
        for a, b in SPEECH_S:
            a, b = int(a * SR), int(b * SR)
            y[a:b] = mix[a:b]
        return y

    render(rows, out_root, "cleaned", cleaned)

    scores = _real_scores(rows, "cleaned")

    assert scores["nonvoiced_supp_raw_db"] == pytest.approx(-40.0, abs=0.5)
