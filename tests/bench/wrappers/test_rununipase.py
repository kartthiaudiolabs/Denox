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
"""The UniPASE wrapper hands every pinned checkpoint to the release runner.

It hands the device to the runner as well.
"""

# pylint: disable=unused-argument

import os
import sys

import pase
import receipt
import rununipase


def test_the_release_gets_its_four_checkpoints_by_path(tiny_set, monkeypatch):
    in_dir, out_dir = tiny_set
    seen = {}

    def render(repo, files, out, device, checkpoints):
        seen.update(device=device, checkpoints=checkpoints)
        return 0

    monkeypatch.setattr(pase, "render", render)
    monkeypatch.setattr(receipt, "begin",
                        lambda *a, **k: seen.setdefault("begin", a))
    monkeypatch.setattr(
        sys, "argv", ["w", "--in-dir",
                      str(in_dir), "--out-dir",
                      str(out_dir)])
    assert rununipase.main() == 0
    assert seen["begin"][0] == "unipase"
    assert {os.path.basename(p) for p in seen["checkpoints"].values()} == {
        "DeWavLM-Omni.pt",
        "Adapter.pt",
        "Vocoder_DWO-L1.pt",
        "PostNet.pt",
    }
