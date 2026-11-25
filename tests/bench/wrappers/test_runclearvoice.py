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
"""The ClearVoice wrapper renders with absolute paths.

It renders only on the first visible GPU.
"""

import os

import pytest

import runclearvoice


def test_paths_are_absolute_and_every_input_is_rendered(tiny_set, run_wrapper,
                                                        monkeypatch):
    in_dir, out_dir = tiny_set
    monkeypatch.chdir(in_dir.parent)
    calls = run_wrapper(runclearvoice, "mix",
                        os.path.relpath(out_dir, in_dir.parent))
    assert calls["exit"] == 0 and calls["load"] == ("cuda:0",)
    assert calls["begin"][0] == "clearvoice_mf2_48k"
    assert os.path.isabs(calls["begin"][2]) and os.path.isabs(calls["begin"][4])
    assert len(list(out_dir.glob("*.wav"))) == 3


def test_another_device_or_a_missing_checkpoint_stops_before_the_package_loads(
        tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        runclearvoice.load("cuda:1")
    monkeypatch.setattr(runclearvoice, "SRC", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="setupfield"):
        runclearvoice.load("cuda:0")
