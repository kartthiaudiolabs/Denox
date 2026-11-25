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
"""The README's figures: the board chart reads the board, both themes render."""

import json
import os
import sys
import xml.etree.ElementTree as ET

import build_figures

SVG = "{http://www.w3.org/2000/svg}"

BOARD = {
    "set_id": "f5a6420d67b7ef45" + "0" * 48,
    "anchors_version": 7,
    "axes_version": 7,
    "tie_bands": {
        "composite": 0.03,
        "timbre": 0.05
    },
    "order": {
        "full": [
            "voice_isolator_v0", "other", "voice_isolator_v0_realtime", "broken"
        ],
        "nb": []
    },
    "systems": {
        "voice_isolator_v0": {
            "label": "Inworld Voice Isolator (64M, offline)",
            "composite": 0.7489
        },
        "other": {
            "label": "Someone's <denoiser> & co",
            "composite": 0.7261
        },
        "voice_isolator_v0_realtime": {
            "label": "Inworld Voice Isolator realtime",
            "composite": 0.6
        },
        "broken": {
            "label": "no composite",
            "composite": None
        },
    },
}


def test_board_rows_follow_the_board_order_and_mark_our_systems():
    rows = build_figures.board_rows(BOARD)
    assert [r[0].split(" (")[0] for r in rows] == [
        "Inworld Voice Isolator", "Someone's <denoiser> & co",
        "Inworld Voice Isolator realtime"
    ]
    assert [r[2] for r in rows] == [True, False, True]


def test_board_chart_is_generated_from_the_board_in_both_themes():
    for mode in ("light", "dark"):
        svg = build_figures.board_svg(BOARD, mode)
        ET.fromstring(svg)  # well-formed
        palette = build_figures.PALETTES[mode]
        # one bar per system, no legend: the row labels carry the identity
        assert svg.count(f'fill="{palette["ours"]}"') == 2
        assert svg.count(f'fill="{palette["others"]}"') == 1
        for value in ("0.749", "0.726", "0.600"):
            assert value in svg
        assert "Someone&#x27;s &lt;denoiser&gt; &amp; co" in svg
        assert "no composite" not in svg
        assert "0.03 are ties" in svg and "f5a6420d67b7ef45" in svg
    changed = json.loads(json.dumps(BOARD))
    changed["systems"]["other"]["composite"] = 0.5
    assert "0.500" in build_figures.board_svg(changed, "light")


def test_architecture_diagram_names_the_stages_and_both_checkpoints():
    for mode in ("light", "dark"):

        root = ET.fromstring(build_figures.architecture_svg(mode))

        # the words are live text, and the description repeats them in full
        words = "".join("".join(t.itertext()) for t in root.iter(f"{SVG}text"))
        for needle in ("STFT", "Band split", "7 dual-path blocks",
                       "Per-band heads", "iSTFT", "Inside each dual-path block",
                       "Realtime checkpoint", "Offline checkpoint"):
            assert needle in words, needle
        desc = root.find(f"{SVG}desc").text
        for needle in ("STFT", "B = 46 bands", "Seven dual-path blocks",
                       "Per-band linear heads", "X_hat = X + M (.) X + R",
                       "realtime checkpoint", "offline checkpoint"):
            assert needle in desc, needle


def test_architecture_diagram_is_drawn_per_theme_within_its_canvas():
    svgs = {
        mode: build_figures.architecture_svg(mode) for mode in ("light", "dark")
    }

    for mode, svg in svgs.items():
        root = ET.fromstring(svg)
        width = float(root.get("width"))
        assert build_figures.ARCH_PALETTES[mode]["bg"] in svg
        assert all(
            0 <= float(t.get("x")) <= width for t in root.iter(f"{SVG}text"))
    assert svgs["light"] != svgs["dark"]


def test_main_without_a_board_draws_the_architecture_only(
        tmp_path, monkeypatch, capsys):
    out = tmp_path / "assets"
    monkeypatch.setattr(sys, "argv", ["build_figures.py", "--out", str(out)])
    build_figures.main()
    assert sorted(
        os.listdir(out)) == ["architecture-dark.svg", "architecture-light.svg"]
    assert capsys.readouterr().out.count("wrote ") == 2


def test_main_writes_four_figures(tmp_path, monkeypatch, capsys):
    board = tmp_path / "leaderboard.json"
    board.write_text(json.dumps(BOARD))
    out = tmp_path / "assets"
    monkeypatch.setattr(
        sys, "argv",
        ["build_figures.py", "--leaderboard",
         str(board), "--out",
         str(out)])
    build_figures.main()
    names = sorted(os.listdir(out))
    assert names == [
        "architecture-dark.svg", "architecture-light.svg", "board-dark.svg",
        "board-light.svg"
    ]
    assert capsys.readouterr().out.count("wrote ") == 4
