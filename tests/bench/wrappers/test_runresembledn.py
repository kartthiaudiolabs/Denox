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
"""The resemble-enhance wrapper renders every input.

It needs the pinned weights folder.
"""

import pytest

import runresembledn


def test_every_input_is_rendered_under_a_receipt(tiny_set, run_wrapper):
    in_dir, out_dir = tiny_set
    calls = run_wrapper(runresembledn, in_dir, out_dir)
    assert calls["exit"] == 0 and calls["begin"][0] == "resemble_dn"
    assert len(list(out_dir.glob("*.wav"))) == 3


def test_a_missing_weights_folder_stops_before_the_package_loads(
        tmp_path, monkeypatch):
    monkeypatch.setattr(runresembledn, "RUN_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="setupfield"):
        runresembledn.load("cpu")
