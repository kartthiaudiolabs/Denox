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
"""Loading score files.

Sidecars are merged only when they belong to the base scores.
"""

# pylint: disable=protected-access

import json
import math

import pytest
import torch

import anchors
import common
import ranking
import scorefiles
import tiny_board


def test_level_calibration_is_subtracted_from_kept_energy(tmp_path):
    m = tiny_board.metrics(10.0)
    m["_level_cal_db"] = 6.0
    tiny_board.write(tmp_path, "loud", m, deletion_pct=1.0)
    scores = scorefiles.load(str(tmp_path))
    assert scores["loud"]["babble@+4"]["kept_p10"] == pytest.approx(1.0 - 6.0)
    # coherent keys too
    assert scores["loud"]["babble@+4"]["kept_coh_p10"] == pytest.approx(1.0 -
                                                                        6.0)
    assert scores["loud"]["babble@+4"]["kept_coh_nb_p10"] == pytest.approx(1.0 -
                                                                           6.0)
    assert (scores["loud"]["babble@+4"]["babble_supp_db"] == -10.0
           )  # suppression is not a gain-shift metric
    assert scores["loud"]["_level_cal_applied_db"] == 6.0


def test_load_merges_sidecars_applies_level_calibration_and_hf_error(tmp_path):
    base = {
        "_set_id": "abc",
        "_level_cal_db": -2.0,
        "babble@+4": {
            "voice_kept_db": -5.0,
            "kept_mean": -5.0,
            "hf_err_db": 1.25,
            "level_err_db": 0.5,
            "n": 48,
        },
    }
    (tmp_path / "x.metrics.json").write_text(json.dumps(base))
    (tmp_path / "x.nisqa.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "babble@+4": {
                "nisqa_mos": 4.0,
                "n": 12
            }
        }))
    (tmp_path / "x.realx.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": -2.0,
            "_items_primary": {
                "r1": [0, 3]
            },
        }))
    d = scorefiles.load(str(tmp_path))["x"]
    assert d["babble@+4"]["nisqa_mos"] == 4.0
    assert (d["babble@+4"]["_hf_err"] == 1.25 and
            d["babble@+4"]["_level_err"] == 0.5)
    assert d["_level_cal_applied_db"] == -2.0
    # raw minus the level offset
    assert d["babble@+4"]["kept_mean"] == pytest.approx(-3.0)
    assert d["_deletion"]["primary_anchored"] is True
    assert d["_deletion"]["items"] == {"r1": [0, 3]}
    # a sidecar read at another level calibration than the base scores is
    # refused, not mixed in
    (tmp_path / "x.realx.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "_items_primary": {
                "r1": [0, 3]
            },
        }))
    with pytest.raises(ValueError, match="another level calibration"):
        scorefiles.load(str(tmp_path))
    # a millionth of a decibel is the same read (float32 medians jitter)
    assert scorefiles.same_calibration(-0.4207916, -0.4207925)
    assert not scorefiles.same_calibration(-0.42, -0.4205)
    assert scorefiles.same_calibration(None, None)
    assert not scorefiles.same_calibration(None, 0.0)


def test_stale_sidecar_from_another_build_is_ignored(tmp_path):
    (tmp_path / "x.metrics.json").write_text(
        json.dumps({
            "_set_id": "new",
            "babble@+4": {
                "n": 48,
                "babble_supp_db": -12.0
            }
        }))
    (tmp_path / "x.realx.json").write_text(
        json.dumps({
            "_set_id": "old",
            "_items_primary": {
                "r": [1, 1]
            }
        }))
    (tmp_path / "x.nisqa.json").write_text(
        json.dumps({
            "_set_id": "new",
            "babble@+4": {
                "nisqa_mos": 4.0
            }
        }))
    d = scorefiles.load(str(tmp_path))["x"]
    assert ("babble_supp_db_nb" not in d["babble@+4"] and
            d["_deletion"]["items"] == {})
    assert (d["babble@+4"]["nisqa_mos"] == 4.0 and
            d["babble@+4"]["babble_supp_db"] == -12.0)


def test_a_sidecar_of_another_render_than_its_base_scores_is_ignored(
        tmp_path, capsys):
    (tmp_path / "x.metrics.json").write_text(
        json.dumps({
            "_set_id": "tiny",
            "_outputs_sha256": "new",
            "babble@+4": {
                "n": 48,
                "babble_supp_db": -12.0
            }
        }))
    for ext, digest, mos in ((".nisqa.json", "old", 4.0),):
        (tmp_path / f"x{ext}").write_text(
            json.dumps({
                "_set_id": "tiny",
                "_outputs_sha256": digest,
                "babble@+4": {
                    "nisqa_mos": mos
                }
            }))
    d = scorefiles.load(str(tmp_path))["x"]
    assert "nisqa_mos" not in d["babble@+4"]
    assert "x.nisqa.json read another render" in capsys.readouterr().err


def test_sidecar_from_another_subset_is_ignored_and_a_matching_one_merges(
    tmp_path,):
    (tmp_path / "x.metrics.json").write_text(
        json.dumps({
            "_set_id": "s",
            "_subset": "core",
            "babble@+4": {
                "n": 40,
                "nisqa_mos": 4.0,
                "babble_supp_db": -12.0,
            },
        }))
    # no stamp: legacy all-set data, never core coverage
    (tmp_path / "x.nisqa.json").write_text(
        json.dumps({
            "_set_id": "s",
            "babble@+4": {
                "nisqa_mos": 1.0,
                "n": 72
            }
        }))
    (tmp_path / "x.realx.json").write_text(
        json.dumps({
            "_set_id": "s",
            "_subset": "all",
            "_items_primary": {
                "r": [1, 1]
            }
        }))
    d = scorefiles.load(str(tmp_path))["x"]
    # the core value stands
    assert d["babble@+4"]["nisqa_mos"] == 4.0 and d["babble@+4"]["n"] == 40
    assert (d["_deletion"]["items"] == {} and
            d["_deletion"]["primary_anchored"] is False)


def test_sidecar_anchor_version_must_match_base_scores(tmp_path):
    data = {
        "_set_id": "tiny",
        "_subset": "all",
        **anchors.stamp(),
        "clean": {
            "n": 20
        },
    }
    (tmp_path / "model.metrics.json").write_text(json.dumps(data))
    side = dict(data, _anchors_version="different")
    (tmp_path / "model.nisqa.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="anchor version"):
        scorefiles.load(tmp_path)


def test_level_error_is_the_scorers_mean_absolute_level_change(tmp_path):
    quiet, loud = tiny_board.metrics(10.0), tiny_board.metrics(10.0)
    quiet["clean"]["level_err_db"], loud["clean"]["level_err_db"] = 2.5, 1.5
    del loud["music@-6"]["level_err_db"]
    tiny_board.write(tmp_path, "quiet", quiet, deletion_pct=1.0)
    tiny_board.write(tmp_path, "loud", loud, deletion_pct=1.0)
    scores = scorefiles.load(str(tmp_path))
    assert (scores["quiet"]["clean"]["_level_err"] == 2.5 and
            scores["loud"]["clean"]["_level_err"] == 1.5)
    # nothing measured, nothing derived
    assert "_level_err" not in scores["loud"]["music@-6"]
    assert ["clean", "_level_err",
            -1] in [list(r) for r in ranking.AXES_FULL["fidelity"]]
    assert not any(
        row[1].startswith("si_sdr") for row in ranking.AXES_FULL["fidelity"])


def test_load_merges_the_kept_sidecar_with_its_own_item_counts(tmp_path):
    (tmp_path / "x.metrics.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "clean": {
                "n": 72,
                "si_sdr": 30.0
            },
        }))
    (tmp_path / "x.kept.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "clean": {
                "n": 72,
                "n_missing": 0,
                "mag_snr": 27.5,
                "mag_snr_nb": 28.0,
            },
            "nonverbal@+0": {
                "n": 48,
                "n_missing": 0,
                "nv_lost_pct": 12.5,
                "nv_events_lost_pct": 10.0,
                "nv_blocks": 108,
                "nv_events": 50,
                "nv_items": 30,
            },
        }))
    d = scorefiles.load(str(tmp_path))["x"]
    assert d["clean"]["mag_snr"] == 27.5 and d["clean"]["si_sdr"] == 30.0
    assert (d["nonverbal@+0"]["nv_lost_pct"] == 12.5 and
            d["nonverbal@+0"]["nv_blocks"] == 108)
    # only the sidecar's own measurements are merged
    assert "n_missing" not in d["nonverbal@+0"]
    # the items it was scored on
    assert d["nonverbal@+0"]["metric_counts"]["nv_lost_pct"] == 48
    scores = {"x": d}
    assert ranking._covered(scores, "x", "nonverbal@+0", "nv_lost_pct")
    d["nonverbal@+0"]["metric_counts"]["nv_lost_pct"] = 12
    assert not ranking._covered(scores, "x", "nonverbal@+0", "nv_lost_pct")


def test_sidecar_coverage_reads_the_items_that_observe_and_refuses_missing_ones(
    tmp_path,):
    (tmp_path / "x.metrics.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "nonverbal@+0": {
                "n": 48,
                "n_missing": 0
            },
        }))
    cell = {
        "n": 48,
        "n_missing": 0,
        "nv_lost_pct": 12.5,
        "nv_events_lost_pct": 10.0,
        "nv_items": 22,
        "metric_counts": {
            "nv_lost_pct": 22,
            "nv_events_lost_pct": 22
        },
    }
    (tmp_path / "x.kept.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "nonverbal@+0": cell
        }))
    scores = scorefiles.load(str(tmp_path))
    # observing, not scored
    assert scores["x"]["nonverbal@+0"]["metric_counts"]["nv_lost_pct"] == 22
    assert ranking._covered(scores, "x", "nonverbal@+0", "nv_lost_pct")
    # a panel where fewer than MIN_ITEMS recordings carry a laugh cannot feed
    # the axis
    cell["metric_counts"] = {"nv_lost_pct": 12, "nv_events_lost_pct": 12}
    (tmp_path / "x.kept.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "nonverbal@+0": cell
        }))
    assert not ranking._covered(scorefiles.load(str(tmp_path)), "x",
                                "nonverbal@+0", "nv_lost_pct")
    # an item the sidecar did not get back makes the row incomplete, never a
    # smaller mean
    cell["metric_counts"] = {"nv_lost_pct": 22, "nv_events_lost_pct": 22}
    cell.update(n=47, n_missing=1)
    (tmp_path / "x.kept.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "nonverbal@+0": cell
        }))
    scores = scorefiles.load(str(tmp_path))
    assert scores["x"]["nonverbal@+0"]["metric_missing"]["nv_lost_pct"] == 1
    assert not ranking._covered(scores, "x", "nonverbal@+0", "nv_lost_pct")


def test_a_sidecar_measured_on_other_outputs_than_the_base_scores_is_ignored(
    tmp_path,):
    (tmp_path / "x.metrics.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "clean": {
                "n": 48
            }
        }))
    (tmp_path / "x.kept.json").write_text(
        json.dumps({
            "_set_id": "abc",
            "_level_cal_db": 0.0,
            "clean": {
                "n": 48,
                "mag_snr": 40.0
            },
        }))

    def records(path, outputs):
        lines = [json.dumps({"type": "records", "system": "x"})]
        lines += [
            json.dumps({
                "id": uid,
                "identity": {
                    "output": h
                }
            }) for uid, h in outputs.items()
        ]
        (tmp_path / path).write_text("\n".join(lines) + "\n")

    records("x.items.jsonl", {"a.wav": "h1", "b.wav": "h2"})
    records("x.kept.items.jsonl", {"a.wav": "h1"})
    # the same outputs: merged
    assert scorefiles.load(str(tmp_path))["x"]["clean"]["mag_snr"] == 40.0
    # a re-render since: stale
    records("x.kept.items.jsonl", {"a.wav": "h1", "b.wav": "old"})
    assert "mag_snr" not in scorefiles.load(str(tmp_path))["x"]["clean"]
    # no records to check against: accepted as before
    (tmp_path / "x.items.jsonl").unlink()
    assert scorefiles.load(str(tmp_path))["x"]["clean"]["mag_snr"] == 40.0


def test_energy_above_a_band_limited_input_marks_the_system_as_restoring():
    keeps = {
        "srsweep_8k": {
            "band_kept_db": [0, 0, 0, -24.0, -47.0, -47.0, -47.0]
        },
        "srsweep_16k": {
            "band_kept_db": [0, 0, 0, 0, -40.0, -47.0, -47.0]
        }
    }
    restores = {
        "srsweep_8k": {
            "band_kept_db": [0.6, 0.4, 0.1, 55.6, 48.3, 41.4, 31.2]
        },
        "srsweep_16k": {
            "band_kept_db": [0, 0, 0, 0, 54.5, 45.5, 33.7]
        }
    }
    leaks = {"srsweep_8k": {"band_kept_db": [0, 0, 0, 3.9, 0.4, -1.0, -1.9]}}
    above = ranking.ASSIGN["above_db"]
    assert scorefiles.band_above_input_db(keeps) <= above
    assert scorefiles.band_above_input_db(restores) > above
    # the bands inside the input's band are subtracted: a gain cancels
    assert scorefiles.band_above_input_db(restores) == pytest.approx(55.6 -
                                                                     1.1 / 3)
    # a few decibels of resampler leakage are not restoration
    assert scorefiles.band_above_input_db(leaks) <= above
    # a system without a band-limited row is unknown, not a keeper
    assert math.isnan(
        scorefiles.band_above_input_db({"clean": {
            "si_sdr": 50.0
        }}))
    assert scorefiles.band_above_input_db({}) != scorefiles.band_above_input_db(
        {})


def test_the_band_test_ignores_a_gain_and_reads_every_band_above_the_input():
    sr = common.SR
    time = torch.arange(sr, dtype=torch.float64) / sr
    window = 0.5 - 0.5 * torch.cos(2 * math.pi * time)
    # a gain everywhere, no new frequency: not restoration
    ref = (0.1 * window * torch.sin(2 * math.pi * 3800 * time)).float()
    bands = common.band_kept_db(ref * 10**(11 / 20), ref)
    assert scorefiles.band_above_input_db(
        {"srsweep_8k": {
            "band_kept_db": bands
        }}) <= ranking.ASSIGN["above_db"]
    # a tone far above the input's band, beyond the first band above it
    ref = (0.1 * window * torch.sin(2 * math.pi * 1000 * time)).float()
    out = ref + (0.02 * window * torch.sin(2 * math.pi * 15000 * time)).float()
    bands = common.band_kept_db(out, ref)
    assert scorefiles.band_above_input_db(
        {"srsweep_8k": {
            "band_kept_db": bands
        }}) > ranking.ASSIGN["above_db"]
