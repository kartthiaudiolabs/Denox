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
"""Example notebooks contain portable source without saved execution state."""

import json
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted(path for path in (REPO / "examples").rglob("*.ipynb")
                   if ".ipynb_checkpoints" not in path.parts)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_contains_source_without_execution_state(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    saved = []
    if notebook.get("metadata", {}).get("widgets"):
        saved.append("widget state")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("outputs") or cell.get("execution_count") is not None:
            saved.append(f"cell {index}: outputs or execution count")
        if cell.get("attachments"):
            saved.append(f"cell {index}: embedded attachments")
        if any(
                cell.get("metadata", {}).get(key)
                for key in ("execution", "ExecuteTime")):
            saved.append(f"cell {index}: execution metadata")

    assert not saved, {str(path.relative_to(REPO)): saved}


def test_editor_notebook_checkpoints_are_not_tracked():
    if not (REPO / ".git").exists():
        pytest.skip("needs a Git checkout")
    tracked = subprocess.check_output(["git", "ls-files", "-z"],
                                      cwd=REPO,
                                      text=True).split("\0")

    checkpoints = [
        path for path in tracked
        if ".ipynb_checkpoints" in pathlib.PurePosixPath(path).parts
    ]

    assert not checkpoints
