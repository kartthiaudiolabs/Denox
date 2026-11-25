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
"""Ranking rules: axes, deletion cohort, composite, ties.

The tests use synthetic scores and no audio.
"""

import json
import math
import sys

import pytest

import anchors
import common
import leaderboard
import ranking
import scorefiles
import tables
import tiny_board


def test_partial_entrant_is_not_ranked(tmp_path):
    tiny_board.write(tmp_path,
                     "full_a",
                     tiny_board.metrics(10.0),
                     deletion_pct=1.0)
    tiny_board.write(tmp_path,
                     "full_b",
                     tiny_board.metrics(5.0),
                     deletion_pct=2.0)
    only_music = {"music@-6": {"n": 24, "si_sdr": 99.0}, "_level_cal_db": 0.0}
    tiny_board.write(tmp_path, "partial", only_music)
    tiers = {n: ("full", n) for n in ("full_a", "full_b", "partial")}
    scores = scorefiles.load(str(tmp_path))
    axis_scores, comp, order = tiny_board.rank(scores, tiers)
    assert order["full"] == ["full_a", "full_b"]
    assert math.isnan(comp["partial"])
    incomplete = ranking.incomplete(scores, axis_scores, comp, tiers)
    # without the band-limited rows no board can be assigned, and it says so
    reasons = incomplete["full"]["partial"]
    assert reasons[0].startswith("no board: the band-limited rows")
    assert reasons[1].startswith("missing ")


def test_a_saturated_level_calibration_is_not_ranked(tmp_path):
    tiny_board.write(tmp_path,
                     "full_a",
                     tiny_board.metrics(10.0),
                     deletion_pct=1.0)
    tiny_board.write(tmp_path,
                     "full_b",
                     tiny_board.metrics(5.0),
                     deletion_pct=2.0)
    # the scorer's bound was hit
    loud = dict(tiny_board.metrics(8.0), _level_cal_db=common.MAX_LEVEL_CAL_DB)
    tiny_board.write(tmp_path, "loud", loud, deletion_pct=1.0)
    tiers = {n: ("full", n) for n in ("full_a", "full_b", "loud")}
    scores = scorefiles.load(str(tmp_path))
    axis_scores, comp, order = tiny_board.rank(scores, tiers)
    assert order["full"] == ["full_a", "full_b"] and math.isnan(comp["loud"])
    # every axis present, yet not ranked
    assert all(v == v for v in axis_scores["loud"].values())
    incomplete = ranking.incomplete(scores, axis_scores, comp, tiers)
    reasons = incomplete["full"]["loud"]
    assert len(reasons) == 1 and reasons[0].startswith(
        "level calibration saturated at 12 dB")
    within = dict(
        tiny_board.metrics(8.0),
        # inside the bound: ranked
        _level_cal_db=common.MAX_LEVEL_CAL_DB - 0.5,
    )
    tiny_board.write(tmp_path, "loud", within, deletion_pct=1.0)
    scores = scorefiles.load(str(tmp_path))
    axis_scores, comp, order = tiny_board.rank(scores, tiers)
    assert "loud" in order["full"]


def test_thin_condition_coverage_blocks_the_axis(tmp_path):
    tiny_board.write(tmp_path, "a", tiny_board.metrics(10.0), deletion_pct=1.0)
    thin = tiny_board.metrics(5.0)
    thin["music@-6"]["n"] = 3
    tiny_board.write(tmp_path, "b", thin, deletion_pct=1.0)
    tiers = {n: ("full", n) for n in ("a", "b")}
    scores = scorefiles.load(str(tmp_path))
    axis_scores, comp, order = tiny_board.rank(scores, tiers)
    assert math.isnan(axis_scores["b"]["music"]) and math.isnan(comp["b"])
    assert order["full"] == ["a"]


def test_deletion_comes_only_from_the_primary_anchored_sidecar(tmp_path):
    m = tiny_board.metrics(10.0)
    # unfiltered cohort: must not be used
    m.setdefault("real_noisy", {"n": 24})["deleted_rate_pct"] = 7.0
    tiny_board.write(tmp_path, "no_sidecar", m)
    tiny_board.write(tmp_path,
                     "with_sidecar",
                     tiny_board.metrics(10.0),
                     deletion_pct=2)
    scores = scorefiles.load(str(tmp_path))
    ranking.pool_deletion(scores, {f"r{i:03d}": 1 for i in range(100)})
    assert math.isnan(scores["no_sidecar"]["_deletion"]["pct"])
    assert scores["with_sidecar"]["_deletion"]["pct"] == 2.0
    assert scores["with_sidecar"]["_deletion"]["primary_anchored"]


def test_deletion_is_pooled_over_the_common_cohort(tmp_path):
    """A system with only part of the real items cannot be compared on them."""
    full = {f"r{i:03d}": [0, 1] for i in range(100)}
    # 1 % on all items
    tiny_board.write(tmp_path,
                     "a",
                     tiny_board.metrics(10.0),
                     items=dict(full, r000=[1, 1]))
    half = {f"r{i:03d}": [0, 1] for i in range(50)}
    # covers 50 % of the items
    tiny_board.write(tmp_path, "half", tiny_board.metrics(9.0), items=half)
    tiny_board.write(tmp_path,
                     "b",
                     tiny_board.metrics(8.0),
                     items=dict(full, r099=[1, 1], r098=[1, 1]))
    tiers = {n: ("full", n) for n in ("a", "half", "b")}
    scores = scorefiles.load(str(tmp_path))
    _, comp, order = tiny_board.rank(scores, tiers)
    assert math.isnan(scores["half"]["_deletion"]["pct"]) and math.isnan(
        comp["half"])
    assert (scores["a"]["_deletion"]["cohort_items"] ==
            scores["b"]["_deletion"]["cohort_items"] == 100)
    assert (scores["a"]["_deletion"]["pct"] == 1.0 and
            scores["b"]["_deletion"]["pct"] == 2.0)
    assert order["full"] == ["a", "b"]


def test_unranked_entrants_do_not_move_the_normalisation(tmp_path):
    tiny_board.write(tmp_path, "a", tiny_board.metrics(10.0), deletion_pct=1)
    tiny_board.write(tmp_path, "b", tiny_board.metrics(5.0), deletion_pct=2)
    tiers = {n: ("full", n) for n in ("a", "b")}
    scores = scorefiles.load(str(tmp_path))
    _, comp_before, _ = tiny_board.rank(scores, tiers)
    tiny_board.write(
        tmp_path,
        "partial",
        {
            "music@-6": {
                "n": 24,
                "si_sdr": 999.0
            },
            "_level_cal_db": 0.0
        },
    )
    tiers["partial"] = ("full", "partial")
    scores = scorefiles.load(str(tmp_path))
    _, comp_after, order = tiny_board.rank(scores, tiers)
    assert comp_after["a"] == pytest.approx(comp_before["a"])
    assert comp_after["b"] == pytest.approx(comp_before["b"])
    assert order["full"] == ["a", "b"]


def test_ranking_requires_a_frozen_calibration():
    with pytest.raises(ValueError, match="frozen"):
        ranking.rank({})


def test_incomplete_entrant_cannot_move_the_deletion_cohort():
    items_ab = {f"r{i}": [0, 1] for i in range(20)}

    def system(items, deleted_on=()):
        d = {}
        for c in tiny_board.FULL:
            d[c] = {
                "n": 48,
                "babble_supp_db": -10.0,
                "si_sdr": 10.0,
                "kept_p10": 0.0,
                "kept_coh_p10": 0.0,
                "spk_sim": 0.9,
                "nisqa_mos": 4.0,
                "nv_lost_pct": 10.0,
                "mag_snr": 20.0,
                "_hf_err": 0.5,
                "_level_err": 0.5,
                "babble_resid_db": -10.0,
                "event_resid_db": -10.0,
                "floor_resid_db": -10.0,
                "band_added_db": 10.0,
                # a denoiser's per-band levels: nothing above the input's band
                "band_kept_db": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }
        d["_real_cleanup"] = {
            "n": 48,
            "pause_resid_db": -10.0,
            "pause_level_db": -30.0
        }
        d["_coroom"] = {"n": 48, "voice_level_db": -0.5}
        d["_real_quality"] = {"n": 48, "nisqa_mos": 4.0}
        d["_deletion"] = {
            "items": {
                k: [1, 1] if k in deleted_on else [0, 1] for k in items
            }
        }
        return d

    # A deletes on r0..r4, B deletes on r15..r19: on the full cohort they tie
    scores = {
        "A": system(items_ab, deleted_on=[f"r{i}" for i in range(5)]),
        "B": system(items_ab, deleted_on=[f"r{i}" for i in range(15, 20)]),
    }
    tiers = {"A": ("full", "A"), "B": ("full", "B"), "C": ("full", "C")}
    _, comp1, _ = tiny_board.rank(scores, tiers)
    # C covers only r10..r19 (half the items) and lacks a metric: it must not
    # rank and must not shrink the cohort to the items where only B deletes
    third = system({f"r{i}": [0, 1] for i in range(10, 20)})
    del third["clean"]
    scores["C"] = third
    _, comp2, order = tiny_board.rank(scores, tiers)
    assert comp2["A"] == pytest.approx(
        comp1["A"]) and comp2["B"] == pytest.approx(comp1["B"])
    assert ("C" not in order["full"] and
            scores["A"]["_deletion"]["cohort_items"] == 20)


def test_core_view_drops_the_extension_conditions_and_needs_core_results(
        tmp_path, monkeypatch):
    try:
        assert ranking.select_view("core") == "core"
        for axes in ranking.AXES_BY_TIER.values():
            conds = {c for subs in axes.values() for c, _, _ in subs}
            # the registry's core axes leave out the whole set's rows of the
            # English-only families, and read crowd chatter at -5 dB
            assert conds.isdisjoint(("stationary@-5", "traffic@-5"))
            assert {"babble@+4", "babble@-5", "clean"} <= conds
        ranking.select_view("all")
        assert any(c == "stationary@-5"
                   for c, _, _ in ranking.AXES_FULL["suppression"])
        with pytest.raises(SystemExit, match="--view must be"):
            ranking.select_view("sideways")
        (tmp_path / "a.metrics.json").write_text(
            json.dumps({
                "_set_id": "s",
                "_subset": "all",
                "clean": {
                    "n": 48
                }
            }))
        tiers = tmp_path / "tiers.json"
        tiers.write_text(json.dumps({"a": ["full", "a"], "b": ["full", "b"]}))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "leaderboard.py",
                "--results",
                str(tmp_path),
                "--view",
                "core",
                "--tiers",
                str(tiers),
            ],
        )
        with pytest.raises(SystemExit, match="BENCH_SUBSET=core"):
            leaderboard.main()
        (tmp_path / "b.metrics.json").write_text(
            json.dumps({
                "_set_id": "s",
                "_subset": "core",
                "clean": {
                    "n": 48
                }
            }))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "leaderboard.py",
                "--results",
                str(tmp_path),
                "--tiers",
                str(tiers),
            ],
        )
        with pytest.raises(SystemExit, match="different subsets"):
            leaderboard.main()
    finally:
        ranking.select_view("all")


def test_missing_outputs_are_listed_per_system_and_condition(
        tmp_path, monkeypatch, capsys):
    full = tiny_board.metrics(10.0)
    full["babble@+4"]["n_missing"] = 3
    full["clean"]["n_missing"] = 0  # zero is not a report
    full["_n_missing"] = 3
    tiny_board.write(tmp_path, "full_a", full, deletion_pct=1.0)
    tiny_board.write(tmp_path,
                     "full_b",
                     tiny_board.metrics(5.0),
                     deletion_pct=2.0)
    scores = scorefiles.load(str(tmp_path))
    assert ranking.missing_outputs(scores) == {"full_a": {"babble@+4": 3}}
    anchors.atomic_json(tmp_path / anchors.RESOLVED,
                        tiny_board.calibration(scores))
    out = tmp_path / "lb.json"
    tiers = tmp_path / "tiers.json"
    tiers.write_text(
        json.dumps({
            "full_a": ["full", "full_a"],
            "full_b": ["full", "full_b"]
        }))
    # nobody registered it
    (tmp_path / "stray.metrics.json").write_text(
        json.dumps(tiny_board.metrics(1.0)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leaderboard.py",
            "--results",
            str(tmp_path),
            "--json-out",
            str(out),
            "--tiers",
            str(tiers),
        ],
    )
    leaderboard.main()
    text = capsys.readouterr().out
    assert "IGNORED (no tier registered" in text and "stray" in text
    assert "MISSING OUTPUTS" in text and "babble@+4 3" in text
    assert json.load(open(out, encoding="utf-8"))["missing_outputs"] == {
        "full_a": {
            "babble@+4": 3
        }
    }


def test_the_composite_is_the_mean_of_the_registrys_parts(tmp_path):
    assert ranking.PARTS == {
        "removes": ["suppression", "music", "cleanup", "events", "voices"],
        "keeps": ["preservation", "fidelity"],
        "sounds": ["timbre", "quality"],
        "restores": ["room"],
    }
    assert ranking.BOARDS == {
        "denoising": ["removes", "keeps", "sounds"],
        "restoration": ["restores"],
    }
    assert ranking.ranking_board() == "denoising"
    # the band axis is a diagnostic beside both boards
    assert ranking.OUTSIDE == ["band"]
    assert ranking.axis_order(
    ) == [a for part in ranking.PARTS.values() for a in part] + ["band"]
    assert sorted(ranking.axis_order()) == sorted(ranking.AXES_FULL)
    rule = ranking.composite_rule()
    assert rule.startswith("two scores, each ranking the systems of its own")
    assert "is ranked on restoration, every other on denoising" in rule
    assert ("denoising = removes (suppression, music, cleanup, events, voices)"
            in rule)
    assert "restoration = restores (room)" in rule
    assert rule.endswith("scored beside both, read by neither: band")
    tiny_board.write(tmp_path, "a", tiny_board.metrics(10.0), deletion_pct=1.0)
    tiny_board.write(tmp_path, "b", tiny_board.metrics(5.0), deletion_pct=2.0)
    scores = scorefiles.load(str(tmp_path))
    tiers = {"a": ("full", "a"), "b": ("full", "b")}
    axis_scores, comp, _ = tiny_board.rank(scores, tiers)
    for n in ("a", "b"):
        parts = anchors.part_scores(axis_scores[n], ranking.PARTS)
        boards = ranking.board_scores(axis_scores[n])
        # the table ranks by the denoising board: the mean of three parts
        assert comp[n] == pytest.approx(
            (parts["removes"] + parts["keeps"] + parts["sounds"]) / 3)
        assert boards["denoising"] == pytest.approx(comp[n])
        assert boards["restoration"] == pytest.approx(parts["restores"])
        assert parts["removes"] == pytest.approx(
            (axis_scores[n]["suppression"] + axis_scores[n]["music"] +
             axis_scores[n]["cleanup"] + axis_scores[n]["events"] +
             axis_scores[n]["voices"]) / 5)
        assert parts["restores"] == pytest.approx(axis_scores[n]["room"])
    # a system that leaves rooms alone loses half of the restoration board and
    # nothing of the denoising board
    scores["a"]["reverb_real"].update(spk_sim=0.86, nisqa_mos=2.0)
    axis_scores_after, comp2, _ = tiny_board.rank(scores, tiers)
    assert axis_scores_after["a"]["room"] == 0.0
    assert axis_scores["a"]["room"] > 0.0
    assert comp2["a"] == pytest.approx(comp["a"])
    before = ranking.board_scores(axis_scores["a"])["restoration"]
    after = ranking.board_scores(axis_scores_after["a"])["restoration"]
    assert before - after == pytest.approx(axis_scores["a"]["room"])
    # each board needs its own evidence: without the room rows a system still
    # ranks on denoising, and its restoration score is simply absent
    del scores["a"]["reverb_real"]
    axis_scores_3, comp3, order3 = tiny_board.rank(scores, tiers)
    assert comp3["a"] == pytest.approx(
        comp["a"]) and order3["full"] == ["a", "b"]
    assert math.isnan(ranking.board_scores(axis_scores_3["a"])["restoration"])
    assert "a" not in ranking.incomplete(scores, axis_scores_3, comp3,
                                         tiers).get("full", {})
    # without a denoising axis it is not ranked
    del scores["a"]["babble@+4"]
    _, comp4, order4 = tiny_board.rank(scores, tiers)
    assert math.isnan(comp4["a"]) and order4["full"] == ["b"]
    labels = {"a": ("full", "A"), "b": ("full", "B")}
    # the band axis is scored beside both boards: its own table, its own order;
    # a section can also list the board by any axis
    beside = tables.markdown_outside(axis_scores, {"full": ["a", "b"]}, labels)
    assert beside.startswith("| # | system | band |")
    by_voices = tables.markdown_outside(axis_scores, {"full": ["a", "b"]},
                                        labels,
                                        axes=["voices"])
    voices_table = by_voices[by_voices.index("| # | system | voices |"):]
    assert (voices_table.count("\n| ") == 2 and "| A |" in voices_table and
            "| B |" in voices_table)


def test_axis_ties_name_the_systems_inside_a_declared_band():
    assert ranking.TIE_BANDS == {"timbre": 0.05}
    axis_scores = {
        "a": {
            "timbre": 0.76
        },
        "b": {
            "timbre": 0.72
        },
        "c": {
            "timbre": 0.70
        },
        "d": {
            "timbre": float("nan")
        },
    }
    tiers = {n: ("full", n.upper()) for n in axis_scores}
    assert ranking.axis_ties(axis_scores, list(axis_scores), tiers) == {
        "timbre": ["A", "B"]
    }
    # one system inside the band is no tie
    assert not ranking.axis_ties(axis_scores, ["a", "c"], tiers)
    assert not ranking.axis_ties(axis_scores, [], tiers)


def test_pooled_rows_feed_an_axis_only_complete_and_with_their_counts():
    # pylint: disable=protected-access
    scores = {
        "s": {
            "_real_cleanup": {
                "pause_resid_db": -10.0,
                "n": 121,
                "n_missing": 0
            },
            "_real_quality": {
                "nisqa_mos": 3.0,
                "n": 23,
                "metric_counts": {
                    "nisqa_mos": 23
                },
            },
            "_deletion": {
                "pct": 0.2,
                "n_segments": 452,
                "cohort_items": 210,
                "complete": True
            },
        }
    }
    assert ranking._covered(scores, "s", "_real_cleanup", "pause_resid_db")
    assert ranking._covered(scores, "s", "_real_quality", "nisqa_mos")
    assert ranking._covered(scores, "s", "_deletion", "pct")
    # an item the sidecar did not get back: an incomplete row, not a mean
    scores["s"]["_real_cleanup"]["n_missing"] = 1
    assert not ranking._covered(scores, "s", "_real_cleanup", "pause_resid_db")
    # too few items behind the pooled row
    scores["s"]["_real_cleanup"].update(n_missing=0, n=ranking.MIN_ITEMS - 1)
    assert not ranking._covered(scores, "s", "_real_cleanup", "pause_resid_db")
    scores["s"]["_real_quality"]["metric_missing"] = {"nisqa_mos": 2}
    assert not ranking._covered(scores, "s", "_real_quality", "nisqa_mos")
    # the deletion row answers to its own completeness on the frozen cohort,
    # and to the segments that cohort holds
    scores["s"]["_deletion"]["n_segments"] = ranking.MIN_ITEMS - 1
    assert not ranking._covered(scores, "s", "_deletion", "pct")
    scores["s"]["_deletion"].update(n_segments=452, complete=False)
    assert not ranking._covered(scores, "s", "_deletion", "pct")
    assert ranking.MIN_ITEMS == 20


def test_a_system_that_restores_above_the_band_is_ranked_on_restoration(
        tmp_path):
    tiny_board.write(tmp_path, "a", tiny_board.metrics(10.0), deletion_pct=1.0)
    restorer = tiny_board.metrics(5.0)
    for cond, _ in scorefiles.BAND_LIMITED_ROWS:
        if cond in restorer:
            restorer[cond]["band_kept_db"] = [0, 0, 0, 40.0, 40.0, 40.0, 40.0]
    tiny_board.write(tmp_path, "r", restorer, deletion_pct=1.0)
    tiers = {"a": ("full", "a"), "r": ("full", "r")}
    scores = scorefiles.load(str(tmp_path))
    assert ranking.board_of(scores["a"]) == "denoising"
    assert ranking.board_of(scores["r"]) == "restoration"
    axis_scores, comp, order = tiny_board.rank(scores, tiers)
    # each board ranks its own systems by its own score; both are scored on
    # both
    assert order["full"] == ["a"] and order["restoration"] == ["r"]
    assert comp["a"] == pytest.approx(
        ranking.board_scores(axis_scores["a"])["denoising"])
    assert comp["r"] == pytest.approx(
        ranking.board_scores(axis_scores["r"])["restoration"])
    assert ranking.board_scores(axis_scores["r"])["denoising"] > 0
    assert "r" not in ranking.incomplete(scores, axis_scores, comp,
                                         tiers).get("full", {})


def test_a_system_without_band_limited_rows_is_ranked_on_no_board(tmp_path):
    tiny_board.write(tmp_path, "a", tiny_board.metrics(10.0), deletion_pct=1.0)
    blind = tiny_board.metrics(12.0)
    for cond, _ in scorefiles.BAND_LIMITED_ROWS:
        blind.pop(cond, None)
    tiny_board.write(tmp_path, "blind", blind, deletion_pct=1.0)
    tiers = {"a": ("full", "a"), "blind": ("full", "blind")}
    scores = scorefiles.load(str(tmp_path))
    axis_scores, comp, order = tiny_board.rank(scores, tiers)
    # complete and better on denoising, yet missing evidence of the band it
    # adds is not evidence that it adds none
    assert ranking.board_of(scores["blind"]) is None
    assert ranking.board_scores(axis_scores["blind"])["denoising"] > comp["a"]
    assert order["full"] == ["a"] and order["restoration"] == []
    assert math.isnan(comp["blind"])
    reasons = ranking.incomplete(scores, axis_scores, comp,
                                 tiers)["full"]["blind"]
    assert reasons[0] == ("no board: the band-limited rows (srsweep_8k,"
                          " srsweep_16k, srsweep_24k) that assign one are"
                          " missing")


def test_a_16k_system_the_rule_sends_to_restoration_is_reported_not_dropped(
        tmp_path):
    tiny_board.write(tmp_path, "a", tiny_board.metrics(10.0), deletion_pct=1.0)
    restorer = tiny_board.metrics(10.0)
    for cond, _ in scorefiles.BAND_LIMITED_ROWS:
        if cond in restorer:
            restorer[cond]["band_kept_db"] = [0, 0, 0, 40.0, 40.0, 40.0, 40.0]
    tiny_board.write(tmp_path, "narrow", restorer, deletion_pct=1.0)
    tiers = {"a": ("full", "a"), "narrow": ("nb", "N")}
    scores = scorefiles.load(str(tmp_path))
    axis_scores, comp, order = tiny_board.rank(scores, tiers)
    assert ranking.board_of(scores["narrow"]) == "restoration"
    assert "narrow" not in order["nb"] + order["restoration"]
    reasons = ranking.incomplete(scores, axis_scores, comp,
                                 tiers)["nb"]["narrow"]
    assert reasons[0] == ("assigned to restoration, which ranks the full-band"
                          " tier only")


def test_the_rule_that_assigns_a_board_is_the_registrys():
    assert ranking.ASSIGN == {
        "by": "band_above_input_db",
        "above_db": 10.0,
        "board": "restoration",
        "otherwise": "denoising",
    }
    assert ranking.AXES_REGISTRY["version"] == "0"
    rule = ranking.composite_rule()
    assert ("is ranked on restoration, every other on denoising, one without"
            " those rows on neither" in rule)
