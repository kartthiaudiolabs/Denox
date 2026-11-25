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
"""The anchor sensitivity sweep on synthetic systems.

The systems' dependence on one constant is known.
"""

import copy

import pytest

import anchor_rows
import anchors
import anchorsweep
import ranking


def _case():
    """Returns two ranked systems that differ on clean passthrough only.

    ``keeper`` returns the recording, ``rewriter`` audibly harms it, and
    ``rewriter`` is a little better everywhere else.
    """
    policy, scores = anchors.definitions(), anchor_rows.scores()
    systems = {k: scores[k] for k in ("INPUT", "REF")}
    systems["keeper"] = copy.deepcopy(scores["a"])
    systems["rewriter"] = copy.deepcopy(scores["c"])
    systems["keeper"]["clean"].update(
        kept_coh_p10=0.0,
        _level_err=0.0,
        _hf_err=0.05,
        spk_sim=1.0,
        nisqa_mos=4.5,
    )
    systems["rewriter"]["clean"].update(
        kept_coh_p10=-7.0,
        _level_err=4.0,
        _hf_err=2.0,
        spk_sim=0.96,
        nisqa_mos=4.1,
    )
    tiers = {
        "keeper": ("full", "Keeper"),
        "rewriter": ("full", "Rewriter"),
        "INPUT": ("ref", "in"),
        "REF": ("ref", "ref"),
    }
    calibration = anchors.resolve(policy, systems, ranking.AXES_BY_TIER)
    return policy, systems, tiers, calibration


def test_ranged_lists_every_declared_constant_with_a_range_and_nothing_else():
    policy = anchors.definitions()
    knobs = anchorsweep.ranged(policy)
    assert knobs and all(low <= policy["rules"][i][side]["value"] <= high
                         for i, side, low, high, _ in knobs)
    ranged_count = sum(1 for r in policy["rules"] for s in ("floor", "ceiling")
                       if "range" in r[s])
    assert len(knobs) == ranged_count
    # zero by definition: no range
    assert not any("babble_supp_db floor" in label for *_, label in knobs)


def test_with_constants_moves_only_the_named_rows_and_refuses_crossed_endpoints(
):
    policy, _, _, calibration = _case()
    i = next(k for k, r in enumerate(policy["rules"])
             if r["conditions"] == ["clean"] and "_level_err" in r["metrics"])
    cal = anchorsweep.with_constants(calibration, policy, {(i, "floor"): 8.0})
    key = anchors.row_key("clean", "_level_err")
    assert (cal["points"][key]["floor"] == 8.0 and
            calibration["points"][key]["floor"] == 6.0)
    other = anchors.row_key("babble@+4", "nisqa_mos")
    assert cal["points"][other] == calibration["points"][other]
    with pytest.raises(ValueError, match="cross"):
        # under the ceiling of 1 dB
        anchorsweep.with_constants(calibration, policy, {(i, "floor"): 0.5})


def test_the_sweep_finds_the_constant_the_order_depends_on():
    policy, systems, tiers, calibration = _case()
    report = anchorsweep.sweep(systems,
                               tiers,
                               calibration,
                               policy,
                               draws=300,
                               seed=1)
    base = report["joint"]
    assert set(base) == {"keeper", "rewriter"}
    assert (report["joint"]["keeper"]["p05"] <= base["keeper"]["declared"] <=
            report["joint"]["keeper"]["p95"])
    keeper_first = base["keeper"]["first_fraction"]
    rewriter_first = base["rewriter"]["first_fraction"]
    assert keeper_first + rewriter_first == pytest.approx(1.0)
    clean = [r for r in report["single"] if r["constant"].startswith("clean ")]
    # every clean constant bites
    assert len(clean) >= 5 and all(r["max_shift"] > 0 for r in clean)
    floor = next(
        r for r in clean if r["constant"] == "clean kept_coh_p10 floor")

    # a lower floor forgives the rewriter's deleted quiet voice, a higher one
    # punishes it
    def gap(e):
        return e["composites"]["keeper"] - e["composites"]["rewriter"]

    assert gap(floor["ends"]["high"]) > gap(floor["ends"]["low"])
    pair = report["pairs"]["keeper"]["rewriter"]
    assert (0.0 <= pair["above_fraction"] <= 1.0 and
            0.0 <= pair["tie_fraction"] <= 1.0)
    mirror = report["pairs"]["rewriter"]["keeper"]
    assert mirror["above_fraction"] == pytest.approx(1 - pair["above_fraction"])


def test_the_sweep_is_seeded_and_leaves_its_inputs_alone():
    policy, systems, tiers, calibration = _case()
    before = copy.deepcopy(calibration)
    a = anchorsweep.sweep(systems, tiers, calibration, policy, draws=50, seed=3)
    b = anchorsweep.sweep(systems, tiers, calibration, policy, draws=50, seed=3)
    c = anchorsweep.sweep(systems, tiers, calibration, policy, draws=50, seed=4)
    assert a["joint"] == b["joint"] and a["joint"] != c["joint"]
    assert calibration == before


def test_markdown_reports_the_joint_table_and_the_order_changing_constants():
    policy, systems, tiers, calibration = _case()
    report = anchorsweep.sweep(systems,
                               tiers,
                               calibration,
                               policy,
                               draws=50,
                               seed=0)
    md = anchorsweep.markdown(report, label=lambda n: n)
    assert "| system | declared | 5th to 95th percentile | ranked first |" in md
    assert md.count("\n| keeper |") + md.count("\n| rewriter |") == 2
    assert "Single constants whose range alone reverses a pair" in md
    quiet = copy.deepcopy(report)
    for r in quiet["single"]:
        for e in r["ends"].values():
            e["order_changed"] = False
    quiet_md = anchorsweep.markdown(quiet, label=lambda n: n)
    assert "None: no single constant" in quiet_md
    ends = [e for r in report["single"] for e in r["ends"].values()]
    assert all(e["order_changed"] == bool(e["reversed_pairs"]) for e in ends)


def test_axis_sensitivity_names_the_axis_a_lead_rests_on():
    policy, systems, tiers, calibration = _case()
    ax = anchorsweep.axis_sensitivity(systems,
                                      tiers,
                                      calibration,
                                      draws=400,
                                      seed=2)
    board = ranking.BOARDS[ranking.ranking_board()] if ranking.BOARDS else list(
        ranking.PARTS or {})
    inside = {
        a for part in board for a in ranking.PARTS[part]
        if a not in ranking.OUTSIDE
    }
    # an axis beside the composite, or on the other board, has no weight to
    # relax: the report describes the score that orders the table
    assert set(ax["leave_one_out"]) == inside
    assert set(ax["axis_scores"]["keeper"]) == inside
    with_fidelity = ax["leave_one_out"]["music"]["composites"]
    without = ax["leave_one_out"]["fidelity"]["composites"]
    # the keeper's only advantage is clean passthrough: leaving fidelity out can
    # only hurt it
    assert (without["keeper"] -
            without["rewriter"]) < (with_fidelity["keeper"] -
                                    with_fidelity["rewriter"])
    assert ax["leave_one_out"]["fidelity"]["order"][0] == "rewriter"
    fractions = ax["random_weights_first_fraction"]
    assert (sum(fractions.values()) == pytest.approx(1.0) and
            0 < fractions["keeper"] < 1)
    report = anchorsweep.sweep(systems,
                               tiers,
                               calibration,
                               policy,
                               draws=20,
                               seed=0)
    report["axes"] = ax
    md = anchorsweep.markdown(report, label=lambda n: n)
    assert ("| axis left out | order |" in md and
            "ranked first under random axis weights" in md)
