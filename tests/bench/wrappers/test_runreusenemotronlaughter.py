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
"""The NemotronLabs RE-USE-laughter wrapper receipts its render.

It runs the release over every input.
"""

import soundfile as sf

import runreusenemotronlaughter


def test_every_input_is_rendered_under_a_receipt(tiny_set, run_wrapper):
    in_dir, out_dir = tiny_set
    calls = run_wrapper(runreusenemotronlaughter, in_dir, out_dir, "--device",
                        "cuda:1")
    assert calls["exit"] == 0 and calls["load"] == ("cuda:1",)
    assert calls["begin"][0] == "reuse_nemotron_laughter" and calls["begin"][
        5] == "cuda:1"
    assert sorted(p.name for p in out_dir.glob("*.wav")) == [
        f"item_{i}.wav" for i in range(3)
    ]
    assert sf.info(str(out_dir / "item_0.wav")).frames == 24000
