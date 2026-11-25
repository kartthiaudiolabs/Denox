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
"""Anchor endpoints, immutable identities, and roster-independent composites."""

# pylint: disable=redefined-outer-name

import copy
import json
import math

import pytest

import anchor_rows
import anchors
import ranking


def test_removing_every_row_including_input_and_reference_leaves_bitwise_scores(
):
    systems = anchor_rows.scores()
    ref_tier = ("INPUT", "REF")
    tiers = {n: ("ref" if n in ref_tier else "full", n) for n in systems}
    calibration = anchors.resolve(anchors.definitions(), systems,
                                  ranking.AXES_BY_TIER)
    _, before, _ = ranking.rank(copy.deepcopy(systems), tiers, calibration)
    for removed in systems:
        reduced = {
            n: copy.deepcopy(v) for n, v in systems.items() if n != removed
        }
        _, after, _ = ranking.rank(reduced, tiers, calibration)
        for name in ("a", "b", "c"):
            if name != removed:
                assert before[name].hex() == after[name].hex()


def test_endpoints_constants_and_monotonicity():
    policy, scores = anchors.definitions(), anchor_rows.scores()
    calibration = anchors.resolve(policy, scores, ranking.AXES_BY_TIER)
    for key, point in calibration["points"].items():
        assert anchors.normalize(point["floor"], point) == 0
        assert anchors.normalize(point["ceiling"], point) == 1
        condition, metric = json.loads(key)
        rule = anchors.rule_for(policy, condition, metric)
        for side, expected in (("floor", 0), ("ceiling", 1)):
            if "source" in rule[side] and side not in point["fallbacks"]:
                source = rule[side]["source"]
                assert (anchors.normalize(scores[source][condition][metric],
                                          point) == expected)
        middle = (point["floor"] + point["ceiling"]) / 2
        assert anchors.normalize(middle + point["direction"] * 0.01,
                                 point) >= anchors.normalize(middle, point)


def test_missing_deletion_item_cannot_shrink_the_cohort():
    systems = anchor_rows.scores()
    calibration = anchors.resolve(anchors.definitions(), systems,
                                  ranking.AXES_BY_TIER)
    tiers = {n: ("full", n) for n in ("a", "b", "c")}
    _, before, _ = ranking.rank(copy.deepcopy(systems), tiers, calibration)
    del systems["c"]["_deletion"]["items"]["r1"]
    _, after, order = ranking.rank(systems, tiers, calibration)
    assert math.isnan(after["c"]) and "c" not in order["full"]
    assert before["a"].hex() == after["a"].hex()
    assert before["b"].hex() == after["b"].hex()


def test_changed_segment_denominator_is_incomplete():
    systems = anchor_rows.scores()
    calibration = anchors.resolve(anchors.definitions(), systems,
                                  ranking.AXES_BY_TIER)
    systems["a"]["_deletion"]["items"]["r0"] = [0, 7]
    ranking.pool_deletion(systems, calibration["deletion_cohort"])
    assert math.isnan(systems["a"]["_deletion"]["pct"])


def test_version_definition_and_content_mismatches_are_refused(tmp_path):
    policy, systems = anchors.definitions(), anchor_rows.scores()
    calibration = anchors.resolve(policy, systems, ranking.AXES_BY_TIER)
    path = tmp_path / anchors.RESOLVED
    anchors.atomic_json(path, calibration)
    assert anchors.load(path, policy) == calibration
    changed = copy.deepcopy(policy)
    changed["rules"][0]["ceiling"]["value"] = -21
    with pytest.raises(ValueError, match="version/definition"):
        anchors.load(path, changed)
    systems["a"]["_anchors_version"] = "older"
    with pytest.raises(ValueError, match="version"):
        anchors.validate_scores(calibration, systems)
    calibration["deletion_cohort"]["r0"] = 9
    anchors.atomic_json(path, calibration)
    with pytest.raises(ValueError, match="digest"):
        anchors.load(path, policy)


def _measured_policy():
    """Returns the shipped policy with a speaker row measured, not declared.

    The speaker-similarity row is measured from INPUT and REF, as a policy may
    still declare it: the fallback path stays covered although version 2
    declares it.
    """
    policy = copy.deepcopy(anchors.definitions())
    for rule in policy["rules"]:
        if "spk_sim" in rule["metrics"]:
            rule["floor"] = {
                "source": "INPUT",
                "fallback": 0,
                "rationale": "measured",
            }
            rule["ceiling"] = {
                "source": "REF",
                "fallback": 1,
                "rationale": "measured",
            }
    return policy


def test_degenerate_gap_uses_only_declared_fallback_and_missing_reference_fails(
):
    policy, systems = _measured_policy(), anchor_rows.scores()
    for data in systems.values():
        data.update(anchors.stamp(policy))
    systems["INPUT"]["babble@+4"]["spk_sim"] = 1.0
    result = anchors.resolve(policy, systems, ranking.AXES_BY_TIER)
    point = result["points"][anchors.row_key("babble@+4", "spk_sim")]
    assert point["floor"] == 0 and point["fallbacks"] == ["floor"]
    del systems["REF"]["babble@+4"]["spk_sim"]
    with pytest.raises(ValueError, match="missing measured anchor"):
        anchors.resolve(policy, systems, ranking.AXES_BY_TIER)


def test_declared_endpoints_are_verified_against_the_set_and_never_moved_by_it(
):
    policy, systems = anchors.definitions(), anchor_rows.scores()
    base = anchors.resolve(policy, systems, ranking.AXES_BY_TIER)
    key = anchors.row_key("babble@+4", "nisqa_mos")
    declared = next(
        r for r in policy["rules"]
        if "nisqa_mos" in r["metrics"] and "babble@+4" in r["conditions"])
    assert base["points"][key]["floor"] == declared["floor"]["value"]
    tolerance = declared["floor"]["verify"]["tolerance"]
    systems["INPUT"]["babble@+4"]["nisqa_mos"] += (
        0.9 * tolerance)  # a build whose input differs a little
    moved = anchors.resolve(policy, systems, ranking.AXES_BY_TIER)
    # the scale does not follow the set
    assert moved["points"] == base["points"]
    systems["INPUT"]["babble@+4"]["nisqa_mos"] += (
        0.2 * tolerance)  # past the tolerance: a stale policy
    with pytest.raises(ValueError, match="declare a new anchor version"):
        anchors.resolve(policy, systems, ranking.AXES_BY_TIER)
    # a verified endpoint still needs its measurement
    del systems["REF"]["babble@+4"]["nisqa_mos"]
    systems["INPUT"]["babble@+4"]["nisqa_mos"] = declared["floor"]["value"]
    with pytest.raises(ValueError, match="missing measured anchor"):
        anchors.resolve(policy, systems, ranking.AXES_BY_TIER)


def test_reference_conditions_cover_measured_and_verified_endpoints(tmp_path):
    """The REF row is scored wherever the policy reads it.

    As a measured endpoint or as the check on a declared one. A policy that only
    declares values still needs REF for its verification.
    """
    assert ("babble@+4" in anchors.reference_conditions()
           )  # shipped policy: declared, verified against REF
    measured = _measured_policy()
    declared = copy.deepcopy(anchors.definitions())
    for rule in declared["rules"]:
        for side in ("floor", "ceiling"):
            if (rule[side].get("verify") or {}).get("source") == "REF":
                del rule[side]["verify"]
    for name, policy, expected in (
        ("measured", measured, True),
        ("declared", declared, False),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(policy))
        assert ("babble@+4" in anchors.reference_conditions(path)) is expected
    assert anchors.reference_conditions(tmp_path / "declared.json") == set()


def test_verify_and_range_declarations_are_validated(tmp_path):
    policy = anchors.definitions()
    assert all("source" not in side
               for r in policy["rules"]
               for side in (r["floor"], r["ceiling"]))

    def broken(edit):
        bad = copy.deepcopy(policy)
        edit(
            next(r for r in bad["rules"] if "nisqa_mos" in r["metrics"] and
                 "babble@+4" in r["conditions"]))
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            anchors.definitions(path)

    # a range that excludes the declared value
    broken(lambda r: r["floor"].update(range=[3, 4]))
    broken(lambda r: r["floor"].update(verify={
        "source": "BEST",
        "tolerance": 0.1
    }))
    broken(lambda r: r["floor"].update(verify={
        "source": "INPUT",
        "tolerance": 0
    }))
    broken(lambda r: r["floor"].update(range=[1, float("inf")]))


def test_migration_restamps_scores_and_item_headers_and_drops_old_calibration(
    tmp_path,):
    policy = anchors.definitions()
    # versions the live policy never carries, so the test holds at any version
    old, other = "test-previous", "test-other"
    previous = dict(_anchors_version=old, _anchors_sha256="old")
    (tmp_path / "a.metrics.json").write_text(
        json.dumps({
            "clean": {
                "si_sdr": 30.0
            },
            **previous
        }))
    # a missing measurement survives the re-stamp
    (tmp_path / "b.nisqa.json").write_text(
        json.dumps({
            "clean": {
                "nisqa_mos": float("nan")
            },
            **previous
        }, indent=1))
    (tmp_path / "a.nisqa.json").write_text(
        json.dumps({
            "x": 1,
            "_anchors_version": other,
            "_anchors_sha256": "older"
        }))
    (tmp_path / "a.meta.json").write_text(json.dumps({"mechanism": "m"}))
    (tmp_path / "a.items.jsonl").write_text(
        json.dumps({
            "anchors_version": old,
            "anchors_sha256": "old",
            "n_expected": 1
        }) + "\n" + json.dumps({
            "id": "x.wav",
            "measurements": {
                "si_sdr": 30.0
            }
        }) + "\n")
    (tmp_path / anchors.RESOLVED).write_text(
        json.dumps({"anchors_version": old}))

    migrated = anchors.migrate(tmp_path, old, policy)
    migrated_again = anchors.migrate(tmp_path, old, policy)

    assert migrated == 3
    assert migrated_again == 0  # idempotent
    kept = (tmp_path / "b.nisqa.json").read_text()
    assert math.isnan(json.loads(kept)["clean"]["nisqa_mos"])
    assert kept.startswith('{\n "clean"')
    m = json.loads((tmp_path / "a.metrics.json").read_text())
    assert m["_anchors_version"] == policy["version"]
    assert m["_anchors_sha256"] == anchors.digest(policy)
    assert m["_anchors_migrated_from"] == [old] and m["clean"] == {
        "si_sdr": 30.0
    }  # raw values untouched
    # another version: left
    nisqa = json.loads((tmp_path / "a.nisqa.json").read_text())
    assert nisqa["_anchors_version"] == other
    assert "_anchors_version" not in json.loads(
        (tmp_path / "a.meta.json").read_text())
    head, row = (tmp_path / "a.items.jsonl").read_text().splitlines()
    assert json.loads(head)["anchors_version"] == policy["version"]
    assert (json.loads(head)["anchors_migrated_from"] == [old] and
            json.loads(row)["id"] == "x.wav")
    assert not (tmp_path / anchors.RESOLVED).exists()


def test_input_and_ref_must_share_their_scorers_but_ref_has_no_deletion_sidecar(
        composite_case, tmp_path):
    scores, _, _ = composite_case(tmp_path)
    scores["INPUT"]["_scorers"] = {
        "metrics": "m1",
        "nisqa": "n1",
        "realx": "r1"
    }
    # the clean references hold no real recording, so no deletion sidecar
    scores["REF"]["_scorers"] = {"metrics": "m1", "nisqa": "n1"}
    resolved = anchors.resolve(anchors.definitions(), scores,
                               ranking.AXES_BY_TIER)
    assert resolved["scorers"] == scores["INPUT"]["_scorers"]
    scores["REF"]["_scorers"] = {"metrics": "m2", "nisqa": "n1"}
    with pytest.raises(ValueError,
                       match=r"different implementations \(metrics\)"):
        anchors.resolve(anchors.definitions(), scores, ranking.AXES_BY_TIER)


def test_private_scores_force_private_calibration_without_a_cli_flag(
        composite_case, tmp_path):

    scores, _, _ = composite_case(tmp_path)
    scores["INPUT"]["_sealed"] = True
    scores["REF"]["_sealed"] = True
    resolved = anchors.resolve(anchors.definitions(), scores,
                               ranking.AXES_BY_TIER)
    assert resolved["sealed"] is True
    scores["REF"]["_sealed"] = False
    with pytest.raises(ValueError, match="private/public scope"):
        anchors.resolve(anchors.definitions(), scores, ranking.AXES_BY_TIER)


def test_a_panel_inherits_the_whole_set_verification_instead_of_repeating_it():
    policy, systems = anchors.definitions(), anchor_rows.scores()
    parent = anchors.resolve(policy, systems, ranking.AXES_BY_TIER)
    panel = copy.deepcopy(systems)
    for data in panel.values():
        data["_subset"] = "gate"
    declared = next(
        r for r in policy["rules"]
        if "nisqa_mos" in r["metrics"] and "babble@+4" in r["conditions"])
    # a mean over the panel's 24 items sits further from the declared value than
    # the set-wide tolerance allows: sampling, not a stale policy
    panel["INPUT"]["babble@+4"]["nisqa_mos"] += (
        2 * declared["floor"]["verify"]["tolerance"])
    with pytest.raises(ValueError, match="declare a new anchor version"):
        anchors.resolve(policy, panel, ranking.AXES_BY_TIER)
    cal = anchors.resolve(policy, panel, ranking.AXES_BY_TIER, parent=parent)
    assert cal["points"] == parent["points"] and cal["subset"] == "gate"
    assert cal["inherits"] == parent["calibration_sha256"]
    assert cal["calibration_sha256"] == anchors.digest({
        k: v for k, v in cal.items() if k != "calibration_sha256"
    })
    # only the whole-set calibration of the same build, policy and scope can be
    # inherited
    for other in (
            dict(parent, set_id="another"),
            dict(parent, sealed=True),
            cal,
    ):
        with pytest.raises(ValueError, match="inherits only"):
            anchors.resolve(policy, panel, ranking.AXES_BY_TIER, parent=other)


def test_the_composite_is_the_mean_of_its_parts_or_of_its_axes():
    scores = {
        "suppression": 0.6,
        "music": 0.9,
        "room": 0.0,
        "preservation": 0.8,
        "quality": 0.4,
    }
    assert anchors.composite(scores, None) == pytest.approx(0.54)
    members = {
        "removes": ["suppression", "music", "room"],
        "keeps": ["preservation"],
        "sounds": ["quality"],
    }
    assert anchors.composite(scores, members) == pytest.approx(
        (0.5 + 0.8 + 0.4) / 3)
    assert anchors.part_scores(scores, members) == pytest.approx({
        "removes": 0.5,
        "keeps": 0.8,
        "sounds": 0.4
    })
    # an added removal axis takes nothing from what is kept
    wider = dict(scores, tones=0.5)
    more = dict(members, removes=members["removes"] + ["tones"])
    assert anchors.composite(wider, more) == pytest.approx(
        (0.5 + 0.8 + 0.4) / 3)
    assert math.isnan(
        anchors.composite(dict(scores, room=float("nan")), members))
    assert math.isnan(anchors.composite({}, members))
    # an axis beside the composite is present but not read
    aside = {
        "removes": ["suppression", "music"],
        "keeps": ["preservation"],
        "sounds": ["quality"],
    }
    assert anchors.composite(scores, aside, ("room",)) == pytest.approx(
        (0.75 + 0.8 + 0.4) / 3)
    assert anchors.composite(dict(scores, room=0.9), aside,
                             ("room",)) == pytest.approx((0.75 + 0.8 + 0.4) / 3)
    room_aside = anchors.composite({"room": 0.9}, None, ("room",))
    assert room_aside != anchors.composite({"room": 0.9}, None, ())
    assert math.isnan(anchors.composite({"room": 0.9}, None, ("room",)))
    assert math.isnan(
        anchors.composite(dict(scores, room=float("nan")), aside, ("room",)))
    parts = anchors.part_scores(dict(scores, room=float("nan")), members)
    assert math.isnan(parts["removes"])
    with pytest.raises(ValueError, match="parts"):
        anchors.composite(scores, {"removes": ["suppression"]})
