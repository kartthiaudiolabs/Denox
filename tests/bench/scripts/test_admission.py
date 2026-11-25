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
"""Known regressions, null controls and listening evidence for admission.

Admissions are replayable.
"""

# pylint: disable=redefined-outer-name

import copy
import math

import pytest

import admission
import anchors


@pytest.fixture
def panel():
    registry = {
        "schema_version": 1,
        "version": "1",
        "policy": {
            "min_items": 20,
            "min_source_groups": 2,
            "min_human_items": 20,
            "max_redundancy": 0.9,
        },
        "baseline": {
            "all": {
                "full": {
                    "suppression": [["speech", "old", 1]]
                }
            }
        },
        "history": [{
            "version": "1",
            "action": "baseline"
        }],
    }
    policy = anchors.definitions()
    policy["rules"].append({
        "conditions": ["speech"],
        "metrics": ["new"],
        "direction": 1,
        "floor": {
            "value": 0,
            "rationale": "test floor"
        },
        "ceiling": {
            "value": 10,
            "rationale": "test ceiling"
        },
    })
    systems = {}
    for name, shift in (("baseline", 0), ("regression", -1), ("null", 0)):
        metadata = {
            "set_id": "set",
            "subset": "all",
            "anchors_version": policy["version"],
            "anchors_sha256": anchors.digest(policy),
            "scorer_sha256": "same",
            "render_identity": {
                "weights_sha256": anchors.digest(name)
            },
        }
        rows = [{
            "id": str(i),
            "condition": "speech",
            "language": "en",
            "output_status": "ok",
            "speech_source": f"source-{i}",
            "bed_source": f"bed-{i % 5}",
            "groups_verified": True,
            "metrics": {
                "new": math.sin(i) + shift,
                "old": math.cos(i)
            },
        } for i in range(40)]
        systems[name] = metadata, rows
    config = {
        "baseline": "baseline",
        "regression": "regression",
        "null": "null",
        "ranked_systems": list(systems),
        "roster": {
            "sha256": "published",
            "systems": list(systems)
        },
    }
    proposal = {
        "view": "all",
        "tier": "full",
        "axis": "suppression",
        "condition": "speech",
        "metric": "new",
        "direction": 1,
    }
    study = {
        "set_id":
            "set",
        "condition":
            "speech",
        "baseline_sha256":
            anchors.digest("baseline"),
        "regression_sha256":
            anchors.digest("regression"),
        "ratings": [{
            "item_id": str(i),
            "listener": str(j),
            "preference": -1
        } for i in range(40) for j in range(3)],
    }
    return config, systems, proposal, registry, policy, study


def test_regression_null_and_human_evidence_admit_and_replay(panel):
    report = admission.evaluate(*panel, n_boot=100)
    assert report["admitted"] and all(report["checks"].values())
    registry = admission.record_change(panel[3], report, "2",
                                       "Validated additional sensitivity")
    assert admission.replay(registry)["all"]["full"]["suppression"] == [
        ["speech", "old", 1],
        ["speech", "new", 1],
    ]
    assert registry["history"][-1]["study_id"] == report["study_id"]


def test_e1_without_a_listening_study_cannot_admit(panel):
    report = admission.evaluate(*panel[:-1], study=None, n_boot=100)
    assert report["checks"]["E1"] and not report["checks"]["E2"]
    assert not report["admitted"]
    with pytest.raises(ValueError, match="not admitted"):
        admission.record_change(panel[3], report, "2", "No human study")


def test_null_separation_and_wrong_human_sign_block_admission(panel):
    for row in panel[1]["null"][1]:
        row["metrics"]["new"] += 0.5
    assert not admission.evaluate(*panel, n_boot=100)["checks"]["E1"]
    for rating in panel[-1]["ratings"]:
        rating["preference"] = 1
    assert not admission.evaluate(*panel, n_boot=100)["checks"]["E2"]


def test_redundant_rows_must_replace_instead_of_gaining_weight(panel):
    for _, rows in panel[1].values():
        for row in rows:
            row["metrics"]["old"] = row["metrics"]["new"]
    assert not admission.evaluate(*panel, n_boot=100)["checks"]["E3"]
    panel[2]["replaces"] = ["speech", "old", 1]
    report = admission.evaluate(*panel, n_boot=100)
    assert report["admitted"]
    registry = admission.record_change(panel[3], report, "2",
                                       "Replace a redundant row")
    assert admission.replay(registry)["all"]["full"]["suppression"] == [[
        "speech", "new", 1
    ]]


def test_unknown_groups_or_absent_anchors_block_admission(panel):
    for row in panel[1]["regression"][1]:
        row["groups_verified"] = False
    report = admission.evaluate(*panel, n_boot=100)
    assert not report["checks"]["E1"] and not report["checks"]["E4"]
    panel[4]["rules"].pop()
    for metadata, _ in panel[1].values():
        metadata["anchors_sha256"] = anchors.digest(panel[4])
    assert not admission.evaluate(*panel, n_boot=100)["checks"]["E4"]


def test_stale_study_and_modified_admission_evidence_are_refused(panel):
    report = admission.evaluate(*panel, n_boot=100)
    changed = copy.deepcopy(report)
    changed["human"] = None
    changed["evidence_sha256"] = anchors.digest({
        k: v for k, v in changed.items() if k != "evidence_sha256"
    })
    with pytest.raises(ValueError):
        admission.record_change(panel[3], changed, "2", "Invalid")
    panel[-1]["baseline_sha256"] = "other"
    with pytest.raises(ValueError, match="checkpoint identity"):
        admission.evaluate(*panel, n_boot=100)


def test_changelog_cannot_reuse_a_version(panel):
    report = admission.evaluate(*panel, n_boot=100)
    with pytest.raises(ValueError, match="unique version"):
        admission.record_change(panel[3], report, "1", "Changed membership")


def test_cherry_picked_roster_cannot_admit(panel):
    panel[0]["ranked_systems"] = ["baseline"]
    with pytest.raises(ValueError, match="entire pinned"):
        admission.evaluate(*panel, n_boot=100)


def test_removal_requires_two_intact_consecutive_failed_panels(panel):
    panel[2]["metric"] = "old"
    registry = panel[3]
    registry["baseline"]["all"]["full"]["suppression"].append(
        ["speech", "remaining", 1])
    panel[0]["sequence"] = 1
    first = admission.evaluate(*panel, n_boot=100, review_existing=True)
    panel[0]["sequence"] = 2
    second = admission.evaluate(*panel, n_boot=100, review_existing=True)
    result = admission.record_removal(
        registry,
        [first, second],
        "older-set",
        "2",
        "Repeated loss of sensitivity",
    )
    assert admission.replay(result)["all"]["full"]["suppression"] == [[
        "speech", "remaining", 1
    ]]
    second["sequence"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        admission.record_removal(registry, [first, second], "older-set", "2",
                                 "Invalid")


def test_changing_metric_coverage_between_panel_arms_blocks_admission(panel):
    panel[1]["regression"][1][0]["metrics"]["new"] = None
    report = admission.evaluate(*panel, n_boot=100)
    assert report["regression"]["ci95"][1] < 0
    assert not report["checks"]["E1"] and not report["admitted"]


def test_a_redefinition_is_recorded_as_the_maintainers_and_replays(panel):
    registry = panel[3]
    rows = {"full": [["speech", "kept", 1], ["speech", "level", -1]]}
    staged = admission.record_redefinition(
        registry,
        "suppression",
        rows,
        "2",
        "Score what is audible",
        "bench/METRICS.md",
    )
    replayed = admission.replay(staged)
    assert replayed["all"]["full"]["suppression"] == rows["full"]
    assert admission.replay(registry)["all"]["full"]["suppression"] == [[
        "speech", "old", 1
    ]]
    change = staged["history"][-1]
    assert change["decided_by"] == "maintainer" and change["study_id"] is None

    def broken(edit):
        bad = copy.deepcopy(staged)
        edit(bad["history"][-1])
        with pytest.raises(ValueError):
            admission.replay(bad)

    # never dressed up as evidence
    broken(lambda c: c.update(decided_by="panel"))
    broken(lambda c: c.update(note=""))
    broken(lambda c: c.update(axis="absent"))
    # changes nothing
    broken(lambda c: c.update(rows={"full": [["speech", "old", 1]]}))
    broken(lambda c: c.update(
        rows={"full": [["speech", "kept", 1], ["speech", "kept", -1]]}))
    broken(lambda c: c.update(rows={
        "full": rows["full"],
        "nb": rows["full"]
    }))  # a tier the views lack
    # made against another registry
    broken(lambda c: c.update(axis_sha256="0" * 64))
    with pytest.raises(ValueError, match="unique version"):
        admission.record_redefinition(registry, "suppression", rows, "1",
                                      "reason", "bench/METRICS.md")


def test_a_revision_adds_an_axis_states_every_view_and_replays(panel):
    registry = panel[3]
    axes = {
        "all": {
            "full": {
                "suppression": [["speech", "kept", 1]],
                "room": [["reverb", "spk", 1]],
            }
        }
    }
    members = {"removes": ["suppression", "room"]}
    staged = admission.record_revision(registry, axes, "2", "Cover rooms",
                                       "bench/METRICS.md", members)
    assert admission.replay(staged)["all"]["full"] == axes["all"]["full"]
    assert (admission.parts(staged) == members and
            admission.parts(registry) is None)
    change = staged["history"][-1]
    assert change["action"] == "revise" and change["decided_by"] == "maintainer"

    def broken(edit):
        bad = copy.deepcopy(staged)
        edit(bad["history"][-1])
        with pytest.raises(ValueError):
            admission.replay(bad)

    broken(lambda c: c.update(decided_by="panel"))
    broken(lambda c: c.update(note=""))
    broken(lambda c: c.update(reason=""))
    broken(lambda c: c.update(axis_sha256="0" * 64))
    # a view the registry lacks
    broken(lambda c: c.update(axes={"core": axes["all"]}))
    # and a tier
    broken(lambda c: c.update(axes={"all": {"nb": axes["all"]["full"]}}))
    broken(lambda c: c.update(
        axes={"all": {
            "full": {
                "suppression": [["speech", "old", 1]]
            }
        }}))
    broken(lambda c: c.update(
        axes={"all": {
            "full": {
                "room": [["reverb", "spk", 0]]
            }
        }}))
    # an axis left out
    broken(lambda c: c.update(parts={"removes": ["suppression"]}))
    broken(lambda c: c.update(parts={
        "removes": ["suppression", "room"],
        "keeps": ["room"]
    }))
    # beside the composite: a scored axis
    broken(lambda c: c.update(outside=["absent"]))
    # and never all of them
    broken(lambda c: c.update(outside=["suppression", "room"]))
    # the parts then hold the axes inside alone
    broken(lambda c: c.update(outside=["room"]))
    with pytest.raises(ValueError, match="unique version"):
        admission.record_revision(registry, axes, "1", "reason",
                                  "bench/METRICS.md")
    beside = admission.record_revision(
        registry,
        axes,
        "2",
        "Cover rooms",
        "bench/METRICS.md",
        {"removes": ["suppression"]},
        None,
        ["room"],
    )
    assert (admission.outside(beside) == ["room"] and
            not admission.outside(registry))
    assert admission.parts(beside) == {"removes": ["suppression"]}


def test_views_of_a_revision_score_the_same_axes(panel):
    registry = copy.deepcopy(panel[3])
    registry["baseline"]["core"] = copy.deepcopy(registry["baseline"]["all"])
    room = {"full": {"room": [["reverb", "spk", 1]]}}
    with pytest.raises(ValueError, match="same axes"):
        admission.record_revision(
            registry,
            {
                "all": room,
                "core": {
                    "full": {
                        "suppression": [["speech", "kept", 1]]
                    }
                },
            },
            "2",
            "reason",
            "bench/METRICS.md",
        )
    staged = admission.record_revision(registry, {
        "all": room,
        "core": room
    }, "2", "reason", "note.md")
    core = admission.replay(staged)["core"]["full"]
    assert set(core) == {"suppression", "room"}
    # a revision of one view's rows leaves the other view as it was
    wider = {
        "full": {
            "suppression": [["speech", "kept", 1], ["crowd", "kept", 1]]
        }
    }
    narrowed = admission.record_revision(staged, {"core": wider}, "3", "reason",
                                         "note.md")
    views = admission.replay(narrowed)
    assert views["core"]["full"]["suppression"] == wider["full"]["suppression"]
    assert views["all"] == admission.replay(staged)["all"]


def test_bench_axes_selects_another_registry_file(tmp_path, monkeypatch):
    # pylint: disable-next=import-outside-toplevel
    import importlib
    live = admission.DEFAULT
    copied = tmp_path / "axes.json"
    copied.write_text(live.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("BENCH_AXES", str(copied))
    try:
        reloaded = importlib.reload(admission)
        assert reloaded.DEFAULT == copied
        assert reloaded.load()["version"] == admission.load(copied)["version"]
    finally:
        # the environment as it was (the live file, or the registry BENCH_AXES
        # named for this run), then the module as it was: a reload under
        # another file would leave every later test on a registry of its own
        monkeypatch.undo()
        importlib.reload(admission)
    assert admission.DEFAULT == live and live != copied


def _beside(panel):
    """The fixture's registry with a new axis scored beside the composite."""
    registry = admission.record_revision(
        panel[3], {"all": {
            "full": {
                "voices": [["speech", "new", 1]]
            }
        }}, "2", "Show the voices beside the board", "bench/METRICS.md",
        {"removes": ["suppression"]}, None, ["voices"])
    proposal = dict(panel[2], axis="voices")
    return registry, proposal


def test_an_axis_beside_the_composite_is_promoted_on_a_passing_review(panel):
    registry, proposal = _beside(panel)
    config, systems, _, _, policy, study = panel
    report = admission.evaluate(config,
                                systems,
                                proposal,
                                registry,
                                policy,
                                study,
                                n_boot=100,
                                review_existing=True,
                                redundancy_rows=[["speech", "old", 1]])
    assert not report["admitted"] and all(report["checks"].values())
    staged = admission.record_promotion(registry, report, "removes", "3",
                                        "Listeners prefer the removal",
                                        "bench/METRICS.md")
    assert not admission.outside(staged)
    assert admission.parts(staged) == {"removes": ["suppression", "voices"]}
    assert admission.replay(staged)["all"]["full"]["voices"] == [[
        "speech", "new", 1
    ]]
    assert staged["history"][-1]["study_id"] == report["study_id"]


def test_a_promotion_without_passing_or_intact_evidence_is_refused(panel):
    registry, proposal = _beside(panel)
    config, systems, _, _, policy, study = panel
    # reviewed against its own axis alone, the row has nothing to be compared
    # with: redundancy (E3) does not pass
    alone = admission.evaluate(config,
                               systems,
                               proposal,
                               registry,
                               policy,
                               study,
                               n_boot=100,
                               review_existing=True)
    assert not alone["checks"]["E3"]
    with pytest.raises(ValueError, match="every check"):
        admission.record_promotion(registry, alone, "removes", "3", "r",
                                   "bench/METRICS.md")
    report = admission.evaluate(config,
                                systems,
                                proposal,
                                registry,
                                policy,
                                study,
                                n_boot=100,
                                review_existing=True,
                                redundancy_rows=[["speech", "old", 1]])
    # evidence edited after the review is not intact
    changed = dict(report, human=dict(report["human"], ci95=[-0.5, -0.1]))
    with pytest.raises(ValueError, match="every check"):
        admission.record_promotion(registry, changed, "removes", "3", "r",
                                   "bench/METRICS.md")
    with pytest.raises(ValueError, match="existing part"):
        admission.record_promotion(registry, report, "keeps", "3", "r",
                                   "bench/METRICS.md")
    with pytest.raises(ValueError, match="note"):
        admission.record_promotion(registry, report, "removes", "3", "r", "")
    # a hand-edited promotion that also moves another axis does not replay
    staged = admission.record_promotion(registry, report, "removes", "3", "r",
                                        "bench/METRICS.md")
    staged["history"][-1]["parts"] = {"removes": ["voices"]}
    with pytest.raises(ValueError, match="changes nothing else"):
        admission.replay(staged)


def test_a_revision_moves_an_axis_into_the_composite_without_restating_rows(
        panel):
    registry, _ = _beside(panel)
    staged = admission.record_revision(registry, {}, "3", "Count the voices",
                                       "bench/METRICS.md",
                                       {"removes": ["suppression", "voices"]},
                                       None, [])
    assert not admission.outside(staged)
    assert admission.parts(staged) == {"removes": ["suppression", "voices"]}
    assert admission.replay(staged)["all"]["full"]["voices"] == [[
        "speech", "new", 1
    ]]
    assert staged["history"][-1]["action"] == "revise"


def test_a_revision_that_restates_nothing_and_moves_nothing_is_refused(panel):
    registry, _ = _beside(panel)
    with pytest.raises(ValueError, match="moves"):
        admission.record_revision(registry, {}, "3", "r", "bench/METRICS.md",
                                  {"removes": ["suppression"]}, None,
                                  ["voices"])
    with pytest.raises(ValueError, match="moves"):
        admission.record_revision(registry, {}, "3", "r", "bench/METRICS.md")


def test_a_revision_that_moves_an_axis_beside_states_where_the_rest_sit(panel):
    registry, _ = _beside(panel)
    # clearing `outside` with the parts inherited from before would leave
    # voices in neither: no place on the board
    with pytest.raises(ValueError, match="every axis inside it"):
        admission.record_revision(registry, {}, "3", "Count the voices",
                                  "bench/METRICS.md", None, None, [])
    # moving suppression beside the composite with parts that still list it
    # and leave voices out
    with pytest.raises(ValueError, match="every axis inside it"):
        admission.record_revision(registry, {}, "3", "r", "bench/METRICS.md",
                                  None, None, ["suppression"])


def test_a_promotion_sees_the_same_measurement_in_another_axis(panel):
    registry, proposal = _beside(panel)
    # the composite already reads the voices row inside suppression
    doubled = admission.record_revision(
        registry, {
            "all": {
                "full": {
                    "suppression": [["speech", "old", 1], ["speech", "new", 1]]
                }
            }
        }, "3", "Read the new row in suppression too", "bench/METRICS.md")
    config, systems, _, _, policy, study = panel
    against = [[axis, row]
               for part in admission.parts(doubled).values()
               for axis in part
               for row in admission.replay(doubled)["all"]["full"][axis]]
    report = admission.evaluate(config,
                                systems,
                                proposal,
                                doubled,
                                policy,
                                study,
                                n_boot=100,
                                review_existing=True,
                                redundancy_rows=against)
    # identical to a row the composite reads: redundancy 1, E3 fails
    assert not report["checks"]["E3"]
    assert any(r["abs_correlation"] == pytest.approx(1.0)
               for r in report["redundancy"])
    with pytest.raises(ValueError, match="every check"):
        admission.record_promotion(doubled, report, "removes", "4", "r",
                                   "bench/METRICS.md")


def test_a_revision_declares_the_boards_over_every_part_exactly_once(panel):
    registry = panel[3]
    axes = {
        "all": {
            "full": {
                "suppression": [["speech", "kept", 1]],
                "room": [["reverb", "spk", 1]],
            }
        }
    }
    members = {"removes": ["suppression"], "restores": ["room"]}
    # a part in no board
    with pytest.raises(ValueError, match="boards"):
        admission.record_revision(registry, axes, "2", "r", "n.md", members,
                                  None, None, {"denoising": ["removes"]})
    staged = admission.record_revision(registry, axes, "2", "Two boards",
                                       "n.md", members, None, None, {
                                           "denoising": ["removes"],
                                           "restoration": ["restores"]
                                       })
    assert admission.boards(staged) == {
        "denoising": ["removes"],
        "restoration": ["restores"]
    }
    assert admission.boards(registry) is None
    # a part twice, with the parts inherited
    with pytest.raises(ValueError, match="boards"):
        admission.record_revision(staged, {}, "3", "r", "n.md", None, None,
                                  None, {
                                      "denoising": ["removes", "removes"],
                                      "restoration": ["restores"]
                                  })
    # a hand-edited board that drops a part does not replay
    broken = copy.deepcopy(staged)
    broken["history"][-1]["boards"] = {"denoising": ["removes"]}
    with pytest.raises(ValueError, match="boards"):
        admission.replay(broken)


def test_a_revision_states_the_rule_that_assigns_each_system_a_board(panel):
    axes = {
        "all": {
            "full": {
                "suppression": [["speech", "kept", 1]],
                "room": [["reverb", "spk", 1]],
            }
        }
    }
    members = {"removes": ["suppression"], "restores": ["room"]}
    staged = admission.record_revision(panel[3], axes, "2", "Two boards",
                                       "n.md", members, None, None, {
                                           "denoising": ["removes"],
                                           "restoration": ["restores"]
                                       })
    rule = {
        "by": "band_above_input_db",
        "above_db": 10.0,
        "board": "restoration",
        "otherwise": "denoising",
    }
    assert admission.assignment(staged) is None
    assigned = admission.record_revision(staged, {},
                                         "3",
                                         "Each board ranks its own systems",
                                         "n.md",
                                         assign=rule)
    assert admission.assignment(assigned) == rule
    assert assigned["version"] == "3"
    # the same rule again changes nothing
    with pytest.raises(ValueError, match="moves"):
        admission.record_revision(assigned, {},
                                  "4",
                                  "r",
                                  "n.md",
                                  assign=dict(rule))
    # a board not in force, an undeclared diagnostic, one board for both, a
    # flag for a threshold, a missing field
    for bad in (dict(rule, board="enhancement"), dict(rule, by="nisqa_mos"),
                dict(rule,
                     otherwise="restoration"), dict(rule, above_db=True), {
                         k: v for k, v in rule.items() if k != "above_db"
                     }):
        with pytest.raises(ValueError, match="assignment"):
            admission.record_revision(staged, {}, "3", "r", "n.md", assign=bad)
    # boards renamed under an inherited rule that no longer names them
    with pytest.raises(ValueError, match="assignment"):
        admission.record_revision(assigned, {}, "4", "r", "n.md", None, None,
                                  None, {
                                      "isolation": ["removes"],
                                      "restoration": ["restores"]
                                  })


def test_the_live_registry_is_one_published_baseline():
    live = admission.load()
    assert live["version"] == "0" and len(live["history"]) == 1
    # the baseline states the layout, and is checked as a revision would be
    assert admission.assignment(live)["board"] == "restoration"
    assert admission.boards(live) and admission.parts(live)
    broken = copy.deepcopy(live)
    broken["history"][0]["assign"] = dict(admission.assignment(live),
                                          otherwise="enhancement")
    with pytest.raises(ValueError, match="assignment"):
        admission.replay(broken)


def test_the_command_line_needs_the_board_named(tmp_path, monkeypatch):
    # the published board is an input, never a path the tool assumes
    monkeypatch.setattr("sys.argv", [
        "admission.py", "--panel",
        str(tmp_path / "panel.json"), "--row",
        str(tmp_path / "row.json"), "--out",
        str(tmp_path / "decision.json")
    ])
    with pytest.raises(SystemExit):
        admission.main()
    assert not (tmp_path / "decision.json").exists()
