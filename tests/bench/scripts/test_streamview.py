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
"""The streaming probe on the board: which sidecars are shown, and how."""

import json

import pytest

import streamview


def _stream_side(verdict, set_id="abc", **extra):
    # pylint: disable-next=import-outside-toplevel
    import scorestream

    side = {
        "_set_id": set_id,
        "_subset": "all",
        "_rule_version": scorestream.RULE_VERSION,
        "_scorer_sha256": scorestream.SCORER_SHA256,
        "verdict": verdict,
        "causal": verdict == "yes",
        "holds": verdict == "yes",
        "look_ahead_ms": 20.0,
        "look_ahead_max_ms": 30.0,
        "drift_mean_db": -0.1,
        "drift_min_db": -0.8,
        "cold_start_db": -0.3,
        "dropped_seconds": 0,
        "speech_seconds": 600,
        "dropout_items": 0,
        "path_switch": False,
        "context_diff_db": -25.0,
        "n_items": 16,
        "reasons": [] if verdict == "yes" else ["drifts over a long input"],
        "rule": {
            "version": "1"
        },
        "items": [{
            "id": "longATp4_001.wav",
            "output_sha256": "out-001"
        }],
    }
    side.update(extra)
    return side


def test_stream_sidecars_load_per_system_else_whole_set_and_ignore_other_builds(
        tmp_path, capsys):
    whole, core = tmp_path, tmp_path / "core"
    core.mkdir()
    (whole / "a.stream.json").write_text(json.dumps(_stream_side("yes")))
    (whole / "b.stream.json").write_text(
        json.dumps(_stream_side("no", set_id="old")))
    (core / "c.stream.json").write_text(json.dumps(_stream_side("no")))
    scores = {
        "a": {
            "_set_id": "abc",
            "_subset": "core"
        },
        "b": {
            "_set_id": "abc",
            "_subset": "core"
        },
        "c": {
            "_set_id": "abc",
            "_subset": "core"
        },
        "d": {
            "_set_id": "abc",
            "_subset": "core"
        },
    }
    stream = streamview.load_streaming(str(core), scores)
    # a: from the whole-set directory beside the view
    assert set(stream) == {"a", "c"}
    assert "b.stream.json probed on set old" in capsys.readouterr().err
    assert (streamview.streaming_cell(stream, "a") == "yes" and
            streamview.streaming_cell(stream, "c") == "no")
    assert (streamview.streaming_cell(stream, "d") == "–" and
            streamview.streaming_cell(None, "a") == "–")
    # a whole-set view never looks beside itself
    scores_all = {k: {"_set_id": "abc", "_subset": "all"} for k in ("a", "c")}
    assert set(streamview.load_streaming(str(core), scores_all)) == {"c"}
    summary = streamview.streaming_summary(stream["a"])
    assert summary["verdict"] == "yes" and summary["look_ahead_max_ms"] == 30.0
    assert summary["dropped_seconds"] == 0 and summary["cold_start_db"] == -0.3
    assert "items" not in summary and "_set_id" not in summary
    (whole / "e.stream.json").write_text(
        json.dumps(_stream_side("yes", _sealed=True)))
    with pytest.raises(ValueError, match="scope"):
        streamview.load_streaming(str(whole),
                                  {"e": {
                                      "_set_id": "abc",
                                      "_subset": "all"
                                  }})


def test_obsolete_streaming_verdicts_are_not_shown(tmp_path, capsys):
    # pylint: disable-next=import-outside-toplevel
    import observations

    scores = {
        k: {
            "_set_id": "abc",
            "_subset": "all"
        } for k in ("rule", "impl", "render", "fresh")
    }
    (tmp_path / "rule.stream.json").write_text(
        json.dumps(_stream_side("yes", _rule_version="0")))
    (tmp_path / "impl.stream.json").write_text(
        json.dumps(_stream_side("yes", _scorer_sha256="x")))
    for name, output in (("render", "out-002"), ("fresh", "out-001")):
        (tmp_path / f"{name}.stream.json").write_text(
            json.dumps(_stream_side("yes")))
        observations.write(
            tmp_path / f"{name}.items.jsonl",
            [{
                "id": "longATp4_001.wav",
                "identity": {
                    "output": output
                }
            }],
            {
                "system": name,
                "set_id": "abc",
                "n_expected": 1
            },
        )
    stream = streamview.load_streaming(str(tmp_path), scores)
    assert set(stream) == {"fresh"}
    err = capsys.readouterr().err
    assert "rule.stream.json follows probe rule '0'" in err
    assert "impl.stream.json scored by another implementation" in err
    assert (
        "render.stream.json read other outputs than the base scores for 1 items"
        in err)


def test_a_streaming_verdict_compared_with_another_render_is_not_shown(
        tmp_path, capsys):
    scores = {
        k: {
            "_set_id": "abc",
            "_subset": "all",
            "_outputs_sha256": "now"
        } for k in ("moved", "same")
    }
    for name, digest in (("moved", "before"), ("same", "now")):
        (tmp_path / f"{name}.stream.json").write_text(
            json.dumps(_stream_side("yes", _outputs_sha256=digest)))
    assert set(streamview.load_streaming(str(tmp_path), scores)) == {"same"}
    assert "compared with another render" in capsys.readouterr().err


def test_streaming_table_lists_probed_systems_in_board_order_and_names_the_rest(
):
    stream = {
        "x": _stream_side("yes"),
        "z": _stream_side("no", look_ahead_max_ms=10000.0),
    }
    tiers = {"x": ("full", "X"), "y": ("full", "Y"), "z": ("nb", "Z")}
    md = streamview.markdown_streaming(stream, {
        "full": ["y", "x"],
        "nb": ["z"]
    }, tiers)
    lines = md.splitlines()
    assert (lines[0] ==
            "| system | future input (ms) | drift over a long input (dB) | cold"
            " start (dB) | speech dropped (s) | streaming probe |")
    assert (lines[2] ==
            "| X | 20 (max 30) | -0.1 (min -0.8) | -0.3 | 0 of 600 | yes |")
    assert (lines[3] ==
            "| Z | 20 (max 10000) | -0.1 (min -0.8) | -0.3 | 0 of 600 | no |")
    assert lines[-1] == "Not probed: Y."
    assert "Not probed" not in streamview.markdown_streaming(
        stream, {"full": ["x"]}, tiers)
    stream["z"].update(
        look_ahead_capped=True,
        path_switch=True,
        dropped_seconds=7,
        cold_start_db=float("nan"),
    )
    md = streamview.markdown_streaming(stream, {"nb": ["z"]}, tiers)
    assert ("| Z | ≥20 (max ≥10000) | -0.1 (min -0.8) † | – | 7 of 600 | no |"
            in md)
    assert md.splitlines()[-1].startswith(
        "† the release takes different inference paths")


def _measured(drifts, look_ahead=40, dropped=0):
    return [{
        "id": f"longATp4_{i:03d}.wav",
        "output_sha256": f"out-{i:03d}",
        "look_ahead_ms": look_ahead,
        "drift_db": d,
        "dropped_seconds": dropped if i == 0 else 0,
        "speech_seconds": 12,
    } for i, d in enumerate(drifts)]


def test_the_board_judges_the_recorded_items_with_its_own_floor():
    # the scorer said no at its -3 dB floor; one item sits at -4.2 dB
    side = _stream_side("no",
                        items=_measured([-0.2, -4.2, -0.1, -0.1, 0.0, 0.3]))
    judged = streamview.judged(side)
    assert judged["verdict"] == "yes" and judged["holds"] and judged["causal"]
    assert not judged["reasons"]
    assert judged["rule"]["board_drift_min_db"] == -5.0
    assert side["verdict"] == "no"  # the sidecar itself is not rewritten
    # below the board's floor, no
    low = streamview.judged(_stream_side("no", items=_measured([-0.2, -5.5])))
    assert low["verdict"] == "no" and "limits -1 / -5" in low["reasons"][0]
    # a mean under -1 dB, no
    mean = streamview.judged(
        _stream_side("no", items=_measured([-1.5, -1.6, -0.9])))
    assert mean["verdict"] == "no" and not mean["holds"]
    # too much look-ahead, no
    late = streamview.judged(
        _stream_side("yes", items=_measured([-0.1, -0.2], look_ahead=120)))
    assert late["verdict"] == "no" and not late["causal"]
    assert "120 ms of future input" in late["reasons"][0]
    # dropped speech, no
    drop = streamview.judged(
        _stream_side("yes", items=_measured([-0.1, -0.2], dropped=3)))
    assert drop["verdict"] == "no" and "drops speech" in drop["reasons"][0]
    # no per-item measurements: the scorer's verdict stands
    assert streamview.judged(_stream_side("yes"))["verdict"] == "yes"
    assert "under 5 dB" in streamview.rule_sentence()
