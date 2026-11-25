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
"""The URGENT FlowSE wrapper writes 48 kHz.

It writes another rate only when the native-rate view asks for one.
"""

import runurgent26bsrnnflow


def test_the_native_rate_view_reaches_the_loader(tiny_set, run_wrapper):
    in_dir, out_dir = tiny_set
    calls = run_wrapper(runurgent26bsrnnflow, in_dir, out_dir)
    assert calls["exit"] == 0 and calls["load"] == ("cuda:0", False)
    assert calls["begin"][0] == "urgent26_bsrnn_flow"
    calls = run_wrapper(runurgent26bsrnnflow, in_dir, out_dir.parent / "native",
                        "--native-rate")
    assert calls["load"] == ("cuda:0", True)
