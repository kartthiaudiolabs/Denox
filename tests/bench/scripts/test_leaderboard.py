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
"""The board's command line: the checks it runs and the files it writes."""

import json
import sys

import pytest

import leaderboard
import ranking
import scorefiles
import tiny_board


def test_json_is_strict_and_incomplete_entrants_listed(tmp_path):
    tiny_board.write(tmp_path, "a", tiny_board.metrics(10.0), deletion_pct=1.0)
    tiny_board.write(
        tmp_path,
        "partial",
        {
            "music@-6": {
                "n": 24,
                "si_sdr": 1.0
            },
            "_level_cal_db": 0.0
        },
    )
    tiers = {"a": ("full", "A"), "partial": ("full", "P")}
    scores = scorefiles.load(str(tmp_path))
    axis_scores, comp, _ = tiny_board.rank(scores, tiers)
    payload = scorefiles.finite({
        "comp": comp,
        "incomplete": ranking.incomplete(scores, axis_scores, comp, tiers)
    })
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["comp"]["partial"] is None
    assert json.loads(text)["incomplete"]["full"]["partial"]


def test_main_refuses_results_from_different_sets(tmp_path, monkeypatch):
    for name, sid in (("a", "set-one"), ("b", "set-two")):
        (tmp_path / f"{name}.metrics.json").write_text(
            json.dumps({
                "_set_id": sid,
                "clean": {
                    "n": 48
                }
            }))
    tiers = tmp_path / "tiers.json"
    tiers.write_text(json.dumps({"a": ["full", "a"], "b": ["full", "b"]}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["leaderboard.py", "--results",
         str(tmp_path), "--tiers",
         str(tiers)],
    )
    with pytest.raises(SystemExit, match="different eval sets"):
        leaderboard.main()


def test_scores_of_another_render_than_the_one_stamped_now_are_refused(
        tmp_path, monkeypatch):
    for name, scored, stamped in (("a", "one", "one"), ("b", "one", "two")):
        (tmp_path / f"{name}.metrics.json").write_text(
            json.dumps({
                "_set_id": "tiny",
                "_outputs_sha256": scored,
                "clean": {
                    "n": 48
                }
            }))
        (tmp_path / f"{name}.meta.json").write_text(
            json.dumps({"outputs_sha256": stamped}))
    # the set's own mixtures carry no digest on either side
    (tmp_path / "INPUT.metrics.json").write_text(json.dumps({"_set_id": "tiny"
                                                            }))
    scores = scorefiles.load(str(tmp_path))
    assert scorefiles.render_problem({k: scores[k] for k in ("a", "INPUT")},
                                     str(tmp_path)) is None
    assert "the scores of b read other audio" in scorefiles.render_problem(
        scores, str(tmp_path))
    tiers = tmp_path / "tiers.json"
    tiers.write_text(json.dumps({"a": ["full", "a"], "b": ["full", "b"]}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["leaderboard.py", "--results",
         str(tmp_path), "--tiers",
         str(tiers)],
    )
    with pytest.raises(SystemExit, match="re-score them"):
        leaderboard.main()
