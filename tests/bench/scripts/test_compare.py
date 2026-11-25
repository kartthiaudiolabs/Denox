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
"""Paired effects, source dependence, and nonlinear composite aggregation."""

import copy
import math

import numpy as np
import pytest

import compare
import scorefiles


def dataset(shift=0):
    rows = [{
        "id": str(i),
        "condition": "speech",
        "language": "en",
        "output_status": "ok",
        "speech_source": f"source-{i // 3}",
        "bed_source": f"bed-{i % 4}",
        "groups_verified": True,
        "metrics": {
            "quality": i / 10 + shift
        },
    } for i in range(60)]
    return {
        "set_id": "same",
        "subset": "all",
        "anchors_version": "1",
        "anchors_sha256": "same",
        "scorer_sha256": "same",
        "level_cal_db": 0,
    }, rows


def test_known_regression_and_null_have_expected_paired_intervals():
    baseline, regression, null = dataset(), dataset(-1), dataset()
    bad = compare.paired(regression, baseline, "speech", "quality", n_boot=200)
    unchanged = compare.paired(null, baseline, "speech", "quality", n_boot=200)
    assert bad["ci95"] == pytest.approx([-1, -1])
    assert bad["excludes_zero"] and bad["cluster_counts"] == [20, 4]
    assert unchanged["ci95"] == [0, 0] and not unchanged["excludes_zero"]


def test_repeated_source_observations_do_not_create_fake_precision():
    left, right = dataset(), dataset()
    for i, row in enumerate(left[1]):
        row["metrics"]["quality"] += (i // 3) / 10
    before = compare.paired(left, right, "speech", "quality", n_boot=400)
    for data in (left, right):
        repeated = []
        for record in data[1]:
            for j in range(10):
                repeated.append({**record, "id": f"{record['id']}-{j}"})
        data[1][:] = repeated
    after = compare.paired(left, right, "speech", "quality", n_boot=400)
    assert before["ci95"] == pytest.approx(after["ci95"], abs=1e-12)


def test_different_inventories_versions_and_source_groups_are_refused():
    for mutation in ("inventory", "version", "group"):
        left, right = dataset(), dataset()
        if mutation == "inventory":
            right[1].pop()
        elif mutation == "version":
            right[0]["anchors_version"] = "other"
        else:
            right[1][0]["bed_source"] = "other"
        with pytest.raises(ValueError):
            compare.paired(left, right, "speech", "quality", n_boot=100)


def test_nonfinite_values_have_explicit_paired_coverage():
    left, right = dataset(), dataset()
    left[1][0]["metrics"]["quality"] = None
    result = compare.paired(left, right, "speech", "quality", n_boot=100)
    assert result["n_paired"] == 59 and result["n_expected"] == 60


def test_single_source_cannot_claim_an_uncertainty_interval():
    with pytest.raises(ValueError, match="independent"):
        list(compare.weights([["one"] * 30, [None] * 30], 100))


def test_clean_items_do_not_disappear_when_no_bed_is_sampled():
    for weights in compare.weights(
        [["a", "b", "c", "d"], [None, "x", "y", "z"]], 100):
        assert weights.shape == (4,)


def test_composite_matches_leaderboard_and_pools_deletion(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking
    case = composite_case(tmp_path)
    scores, calibration, systems = case
    _, expected, _ = ranking.rank(
        copy.deepcopy(scores),
        {n: ("full", n) for n in ("candidate", "baseline")},
        calibration,
    )
    result = compare.composite_delta(
        systems["candidate"],
        systems["baseline"],
        calibration,
        ranking.AXES_FULL,
        n_boot=100,
    )
    assert result["left"] == pytest.approx(expected["candidate"], abs=1e-12)
    assert result["right"] == pytest.approx(expected["baseline"], abs=1e-12)
    assert math.isfinite(result["delta"])


def test_hf_error_is_the_mean_of_per_item_absolute_errors(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking

    scores, calibration, systems = composite_case(tmp_path)
    for i, row in enumerate(
            r for r in systems["candidate"][1] if r["condition"] == "clean"):
        row["metrics"]["band_kept_db"] = [0] * 4 + [(-1)**i] * 3
    scores["candidate"]["clean"]["_hf_err"] = 1.0  # opposite signs never cancel
    _, expected, _ = ranking.rank(
        copy.deepcopy(scores),
        {n: ("full", n) for n in ("candidate", "baseline")},
        calibration,
    )
    result = compare.composite_delta(
        systems["candidate"],
        systems["baseline"],
        calibration,
        ranking.AXES_FULL,
        n_boot=100,
    )
    assert result["left"] == pytest.approx(expected["candidate"], abs=1e-12)


def test_unused_diagnostics_do_not_enter_composite_uncertainty(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking

    _, calibration, systems = composite_case(tmp_path)
    left = copy.deepcopy(systems["candidate"])
    right = copy.deepcopy(systems["baseline"])
    expected = compare.composite_delta(left,
                                       right,
                                       calibration,
                                       ranking.AXES_FULL,
                                       n_boot=100)
    for _, rows in (left, right):
        rows.append({
            "id": "unused",
            "condition": "long",
            "language": "en",
            "output_status": "ok",
            "groups_verified": False,
            "speech_source": "unknown",
            "bed_source": None,
            "metrics": {},
        })
    assert (compare.composite_delta(left,
                                    right,
                                    calibration,
                                    ranking.AXES_FULL,
                                    n_boot=100) == expected)


def test_an_item_missing_a_high_band_is_left_out_as_on_the_board(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import levels
    # pylint: disable-next=import-outside-toplevel
    import ranking

    scores, calibration, systems = composite_case(tmp_path)
    for name in ("candidate", "baseline"):
        clean = [r for r in systems[name][1] if r["condition"] == "clean"]
        for i, row in enumerate(clean):
            bands = row["metrics"]["band_kept_db"]
            if i < 4:
                bands[4] = None
                bands[5] = 0.9
        per_item = [
            levels.hf_error_db(
                [
                    float("nan") if v is None else v
                    for v in r["metrics"]["band_kept_db"]
                ],
                0.0,
            )
            for r in clean
        ]
        scores[name]["clean"]["_hf_err"] = float(np.nanmean(per_item))
    _, expected, _ = ranking.rank(
        copy.deepcopy(scores),
        {n: ("full", n) for n in ("candidate", "baseline")},
        calibration,
    )
    result = compare.composite_delta(
        systems["candidate"],
        systems["baseline"],
        calibration,
        ranking.AXES_FULL,
        n_boot=100,
    )
    assert result["left"] == pytest.approx(expected["candidate"], abs=1e-12)
    assert result["right"] == pytest.approx(expected["baseline"], abs=1e-12)


def test_a_saturated_level_calibration_has_no_composite_and_no_calibrated_pairs(
        composite_case, saturate, tmp_path):
    composite_case(tmp_path)
    saturate(tmp_path, "candidate")
    left = compare.load_system(tmp_path, "candidate")
    right = compare.load_system(tmp_path, "baseline")
    assert compare.level_saturated(12.0) and compare.level_saturated(-12.5)
    assert not compare.level_saturated(11.9) and not compare.level_saturated(
        None)
    assert compare.saturated_systems(left[0], right[0]) == ["candidate"]
    # pylint: disable-next=import-outside-toplevel
    import anchors
    # pylint: disable-next=import-outside-toplevel
    import ranking

    cal = anchors.load(tmp_path / anchors.RESOLVED, anchors.definitions())
    with pytest.raises(ValueError, match="saturated"):
        compare.composite_delta(left, right, cal, ranking.AXES_FULL)
    with pytest.raises(ValueError, match="saturated"):
        compare.paired(left, right, "babble@-10", "kept_coh_p10", +1)
    # an uncalibrated metric of the same records stays comparable: the level
    # does not enter it
    rec = next(r for r in left[1] if any(
        k not in compare.CALIBRATED and isinstance(v, (int, float))
        for k, v in r["metrics"].items()))
    metric = next(
        k for k, v in rec["metrics"].items()
        if k not in compare.CALIBRATED and isinstance(v, (int, float)))
    assert math.isfinite(
        compare.paired(left, right, rec["condition"], metric, +1)["delta"])


def test_the_voice_outside_speech_is_pooled_over_blocks_as_on_the_board(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking

    scores, calibration, systems = composite_case(tmp_path)
    for name in ("candidate", "baseline"):
        rows = [r for r in systems[name][1] if r["condition"] == "nonverbal@+0"]
        lost = total = 0
        # long events and short ones: a mean of shares would differ
        for i, row in enumerate(rows):
            blocks = 1 + 3 * (i % 5)
            gone = blocks if (name == "baseline" and i % 4 == 0) else (i % 2)
            row["metrics"].update(
                nv_blocks=blocks,
                nv_lost=gone,
                nv_lost_pct=100.0 * gone / blocks,
            )
            lost, total = lost + gone, total + blocks
        scores[name]["nonverbal@+0"]["nv_lost_pct"] = 100.0 * lost / total
    _, expected, _ = ranking.rank(
        copy.deepcopy(scores),
        {n: ("full", n) for n in ("candidate", "baseline")},
        calibration,
    )
    result = compare.composite_delta(
        systems["candidate"],
        systems["baseline"],
        calibration,
        ranking.AXES_FULL,
        n_boot=100,
    )
    assert result["left"] == pytest.approx(expected["candidate"], abs=1e-12)
    assert result["right"] == pytest.approx(expected["baseline"], abs=1e-12)
    # the blocks are the reference's: two systems that disagree on them were not
    # scored alike
    row = next(
        r for r in systems["baseline"][1] if r["condition"] == "nonverbal@+0")
    row["metrics"]["nv_blocks"] += 1
    with pytest.raises(ValueError, match="pooled cohorts"):
        compare.composite_delta(
            systems["candidate"],
            systems["baseline"],
            calibration,
            ranking.AXES_FULL,
            n_boot=10,
        )
    del row["metrics"]["nv_blocks"]
    with pytest.raises(ValueError, match="missing pooled counts"):
        compare.composite_delta(
            systems["candidate"],
            systems["baseline"],
            calibration,
            ranking.AXES_FULL,
            n_boot=10,
        )


def test_axis_subsets_read_their_plain_mean_and_the_whole_reads_the_parts(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking

    scores, calibration, systems = composite_case(tmp_path)
    axis_scores, comp, _ = ranking.rank(
        copy.deepcopy(scores),
        {n: ("full", n) for n in ("candidate", "baseline")},
        calibration,
    )
    pair = systems["candidate"], systems["baseline"]
    whole = compare.composite_delta(*pair,
                                    calibration,
                                    ranking.AXES_FULL,
                                    n_boot=100)
    plain = compare.composite_delta(*pair,
                                    calibration,
                                    ranking.AXES_FULL,
                                    n_boot=100,
                                    parts=None)
    one = compare.composite_delta(*pair,
                                  calibration,
                                  {"room": ranking.AXES_FULL["room"]},
                                  n_boot=100)
    assert compare.registry_parts() == (ranking.PARTS, ["band"])
    # the board's own composite
    assert whole["left"] == pytest.approx(comp["candidate"])
    axes = axis_scores["candidate"]
    # every axis, room too
    assert plain["left"] == pytest.approx(sum(axes.values()) / len(axes))
    # seven equal axes are another policy
    assert plain["left"] != pytest.approx(whole["left"])
    # one axis is no composite of parts
    assert one["left"] == pytest.approx(axes["room"])


def test_load_system_merges_the_kept_sidecar(composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import observations

    composite_case(tmp_path)
    metadata, rows = observations.read(tmp_path / "candidate.items.jsonl")
    side = [
        dict(r,
             metrics={
                 "nv_blocks": 7,
                 "nv_lost": 1,
                 "nv_lost_pct": 100 / 7
             },
             measurements={"nv_kept_db": [-20.0] + [-1.0] * 6})
        for r in rows
        if r["condition"] == "nonverbal@+0"
    ]
    header = {
        k: v for k, v in metadata.items() if k not in ("type", "schema_version")
    }
    observations.write(
        tmp_path / "candidate.kept.items.jsonl",
        side,
        dict(header, scorer_sha256="kept", n_expected=len(side)),
    )
    loaded, merged = compare.load_system(tmp_path, "candidate")
    assert loaded["sidecar_scorers"] == {"kept": "kept"}
    picked = [r for r in merged if r["condition"] == "nonverbal@+0"]
    assert picked and all(r["metrics"]["nv_blocks"] == 7 for r in picked)
    assert compare.value(picked[0], "nv_lost_pct",
                         loaded) == pytest.approx(100 / 7)


def test_lost_blocks_are_counted_again_at_the_base_scores_calibration(
        composite_case, tmp_path):
    """A lost-block count read at another gain never reaches a pairing."""
    # pylint: disable-next=import-outside-toplevel
    import observations

    composite_case(tmp_path)
    base = tmp_path / "candidate.items.jsonl"
    metadata, rows = observations.read(base)
    header = {
        k: v for k, v in metadata.items() if k not in ("type", "schema_version")
    }
    nonverbal = [r for r in rows if r["condition"] == "nonverbal@+0"]
    # the sidecar counted at 0 dB, where blocks 5 dB under the recording are
    # kept; it stores none of them as lost
    side = [
        dict(r,
             metrics={
                 "nv_blocks": 2,
                 "nv_lost": 0,
                 "nv_lost_pct": 0.0
             },
             measurements={"nv_kept_db": [-5.0, -5.0]}) for r in nonverbal
    ]
    observations.write(
        tmp_path / "candidate.kept.items.jsonl",
        side,
        dict(header, scorer_sha256="kept", n_expected=len(side)),
    )
    # the base scores were made again and found a 6 dB gain: those blocks sit
    # 11 dB under the recording once it is undone, so all of them are lost
    observations.write(base, rows, dict(header, level_cal_db=6.0))
    loaded, merged = compare.load_system(tmp_path, "candidate")
    picked = [r for r in merged if r["condition"] == "nonverbal@+0"]
    assert picked and all(r["metrics"]["nv_lost"] == 2 for r in picked)
    assert compare.value(picked[0], "nv_lost_pct", loaded) == 100.0
    # a record without its block levels cannot be counted again
    del side[0]["measurements"]
    observations.write(
        tmp_path / "candidate.kept.items.jsonl",
        side,
        dict(header, scorer_sha256="kept", n_expected=len(side)),
    )
    with pytest.raises(ValueError, match="without its block levels"):
        compare.load_system(tmp_path, "candidate")


def test_a_sidecar_measured_on_other_audio_is_refused(composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import observations

    composite_case(tmp_path)
    metadata, rows = observations.read(tmp_path / "candidate.items.jsonl")
    side = [
        dict(r, metrics={"nisqa_mos": 3.0})
        for r in rows
        if r["condition"] == "babble@+4"
    ]
    header = {
        k: v for k, v in metadata.items() if k not in ("type", "schema_version")
    }
    # a sidecar may add inputs of its own to the semantics (the segments it
    # scored)
    for r in side:
        r["identity"] = dict(r["identity"],
                             semantics=dict(r["identity"]["semantics"],
                                            segs="x"))
    observations.write(
        tmp_path / "candidate.nisqa.items.jsonl",
        side,
        dict(header, scorer_sha256="nisqa", n_expected=len(side)),
    )
    compare.load_system(tmp_path, "candidate")
    # an earlier render of the same item is other audio
    side[0]["identity"] = dict(side[0]["identity"], output="an earlier render")
    observations.write(
        tmp_path / "candidate.nisqa.items.jsonl",
        side,
        dict(header, scorer_sha256="nisqa", n_expected=len(side)),
    )
    with pytest.raises(ValueError, match="measured on other audio"):
        compare.load_system(tmp_path, "candidate")
    # a changed reference too, and a semantics field the base states differently
    side[0]["identity"] = dict(rows[0]["identity"], ref="another reference")
    with pytest.raises(ValueError, match="measured on other audio"):
        compare.same_audio({rows[0]["id"]: rows[0]},
                           [dict(side[0], id=rows[0]["id"])], "x")
    other = dict(
        rows[0],
        identity=dict(rows[0]["identity"], semantics={"condition": "clean"}),
    )
    with pytest.raises(ValueError, match="measured on other audio"):
        compare.same_audio({rows[0]["id"]: rows[0]}, [other], "x")


def test_the_band_row_is_derived_per_item_as_the_loader_derives_it():
    record = {
        "output_status": "ok",
        "condition": "srsweep_8k",
        "metrics": {
            "band_kept_db": [1.0, 1.0, 1.0, 30.0, 21.0, 10.0, 0.0]
        },
    }
    # the highest band above the input's less the mean inside: 30 - 1
    assert compare.value(record, "band_added_db", {}) == pytest.approx(29.0)
    assert math.isnan(
        compare.value(dict(record, condition="clean"), "band_added_db", {}))


def test_the_band_row_is_pooled_as_the_board_pools_it():
    # the two systems' pooled per-band levels are the same, so the board reads
    # the same band value for both: no difference, although every item's own
    # maximum sits 15 dB higher on one side
    left_meta, left = dataset()
    right_meta, right = dataset()
    for i, (a, b) in enumerate(zip(left, right)):
        a["condition"] = b["condition"] = "srsweep_8k"
        above = [30.0, 0.0, 0.0, 0.0] if i % 2 else [0.0, 30.0, 0.0, 0.0]
        a["metrics"] = {"band_kept_db": [1.0, 1.0, 1.0] + above}
        b["metrics"] = {"band_kept_db": [1.0, 1.0, 1.0, 15.0, 15.0, 0.0, 0.0]}
    result = compare.paired((left_meta, left), (right_meta, right),
                            "srsweep_8k",
                            "band_added_db",
                            n_boot=200)
    assert result["delta"] == pytest.approx(0.0)
    assert result["n_paired"] == 60 and result["matched_cohort"]
    # an item without its bands leaves both sides
    left[0]["metrics"]["band_kept_db"][5] = float("nan")
    result = compare.paired((left_meta, left), (right_meta, right),
                            "srsweep_8k",
                            "band_added_db",
                            n_boot=200)
    assert result["n_paired"] == 59 and not result["matched_cohort"]


def test_requested_axes_are_all_read(composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking

    scores, calibration, systems = composite_case(tmp_path)
    axis_scores, _, _ = ranking.rank(
        copy.deepcopy(scores),
        {n: ("full", n) for n in ("candidate", "baseline")},
        calibration,
    )
    pair = systems["candidate"], systems["baseline"]
    axes = axis_scores["candidate"]
    asked = {a: ranking.AXES_FULL[a] for a in ("preservation", "room")}
    # one axis of each board: both are read, as the parts they belong to
    two = compare.composite_delta(*pair, calibration, asked, n_boot=100)
    assert two["left"] == pytest.approx(
        (axes["preservation"] + axes["room"]) / 2)
    # a custom grouping naming an axis that was not requested reads the
    # requested ones
    custom = compare.composite_delta(*pair,
                                     calibration,
                                     asked,
                                     n_boot=100,
                                     parts={
                                         "keeps": ["preservation", "fidelity"],
                                         "restores": ["room"],
                                     })
    assert custom["left"] == pytest.approx(two["left"])


def _restorer(system):
    """The system with the per-band levels of one that restores above every
    band-limited input's band."""
    out = copy.deepcopy(system)
    for record in out[1]:
        first = dict(scorefiles.BAND_LIMITED_ROWS).get(record["condition"])
        if first is not None:
            record["metrics"]["band_kept_db"] = ([0.0] * first + [40.0] *
                                                 (7 - first))
    return out


def test_the_compared_board_is_named_or_assigned_never_assumed(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking

    _, calibration, systems = composite_case(tmp_path)
    whole = compare.composite_delta(systems["candidate"],
                                    systems["baseline"],
                                    calibration,
                                    ranking.AXES_FULL,
                                    n_boot=100)
    # two denoisers: the whole registry reads the denoising board, and says so
    assert whole["board"] == "denoising" and whole["delta"] > 0
    # two restorers whose room falls to the input's while every other axis
    # improves: the comparison reads the restoration board, and the loss
    left, right = _restorer(systems["candidate"]), _restorer(
        systems["baseline"])
    floor = {r["id"]: r for r in systems["INPUT"][1]}
    for record in left[1]:
        if record["condition"] == "reverb_real":
            for metric in ("spk_sim", "nisqa_mos"):
                record["metrics"][metric] = floor[
                    record["id"]]["metrics"][metric]
    lost = compare.composite_delta(left,
                                   right,
                                   calibration,
                                   ranking.AXES_FULL,
                                   n_boot=100)
    assert lost["board"] == "restoration"
    assert lost["delta"] < 0 and lost["ci95"][1] < 0
    named = compare.composite_delta(left,
                                    right,
                                    calibration,
                                    ranking.AXES_FULL,
                                    n_boot=100,
                                    board="denoising")
    assert named["board"] == "denoising" and named["delta"] > 0
    # a restorer against a denoiser: no board is assumed
    with pytest.raises(ValueError, match="different boards"):
        compare.composite_delta(left,
                                systems["baseline"],
                                calibration,
                                ranking.AXES_FULL,
                                n_boot=100)
    # records without the band-limited rows, as on a development panel,
    # cannot tell the board: it must be named
    limited = dict(scorefiles.BAND_LIMITED_ROWS)

    def panel(system):
        return system[0], [
            r for r in system[1] if r["condition"] not in limited
        ]

    with pytest.raises(ValueError, match="name the board"):
        compare.composite_delta(panel(left),
                                panel(right),
                                calibration,
                                ranking.AXES_FULL,
                                n_boot=100)
    assert compare.composite_delta(
        panel(left),
        panel(right),
        calibration,
        ranking.AXES_FULL,
        n_boot=100,
        board="restoration")["delta"] == pytest.approx(lost["delta"])
    with pytest.raises(ValueError, match="no board"):
        compare.composite_delta(left,
                                right,
                                calibration,
                                ranking.AXES_FULL,
                                n_boot=100,
                                board="enhancement")


def test_an_axis_beside_every_part_is_read_when_requested(
        composite_case, tmp_path):
    # pylint: disable-next=import-outside-toplevel
    import ranking

    scores, calibration, systems = composite_case(tmp_path)
    axis_scores, _, _ = ranking.rank(
        copy.deepcopy(scores),
        {n: ("full", n) for n in ("candidate", "baseline")},
        calibration,
    )
    pair = systems["candidate"], systems["baseline"]
    asked = {a: ranking.AXES_FULL[a] for a in ("room", "band")}
    two = compare.composite_delta(*pair, calibration, asked, n_boot=100)
    axes = axis_scores["candidate"]
    # band sits beside every part: requested, it is read as a group of its own
    assert two["left"] == pytest.approx((axes["room"] + axes["band"]) / 2)
    assert two["board"] is None
    # a custom grouping that drops or repeats a requested axis is refused
    for grouping in ({
            "restores": ["room"]
    }, {
            "restores": ["room", "band"],
            "beside": ["band"]
    }):
        with pytest.raises(ValueError, match="exactly once"):
            compare.composite_delta(*pair,
                                    calibration,
                                    asked,
                                    n_boot=100,
                                    parts=grouping)
