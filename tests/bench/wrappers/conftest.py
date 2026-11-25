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
"""Wrapper tests import the wrappers as plain modules.

Release code is never imported here.

``tiny_set`` is a set build of three 48 kHz inputs (``mix/`` beside
``checksums.json``); ``run_wrapper`` runs a wrapper's ``main`` with its release
replaced by a fake.
"""

import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "bench", "wrappers"))

SECONDS = (0.5, 0.75, 1.0)


@pytest.fixture
def tiny_set(tmp_path):
    """Returns (input folder, output folder) of a three-item set build."""
    root = tmp_path / "set"
    (root / "mix").mkdir(parents=True)
    (root / "checksums.json").write_text(json.dumps({"set_id": "tiny-set"}))
    rng = np.random.default_rng(0)
    for i, seconds in enumerate(SECONDS):
        x = (rng.standard_normal(int(seconds * 48000)) * 0.1).astype("float32")
        sf.write(root / "mix" / f"item_{i}.wav", x, 48000, subtype="FLOAT")
    return root / "mix", tmp_path / "out"


@pytest.fixture
def run_wrapper(monkeypatch):
    """Runs ``module.main()`` over a folder with ``load`` and the receipt faked.

    Returns:
        {"load": the arguments load was given, "begin": the receipt call's
        arguments, "exit": main's return value}.
    """

    def run(module, in_dir, out_dir, *extra, enhance=None):
        calls = {}

        def load(*args):
            calls["load"] = args
            return enhance or (lambda path, samples, rate:
                               (samples * 0.5, rate))

        monkeypatch.setattr(module, "load", load)
        monkeypatch.setattr(
            module.receipt,
            "begin",
            lambda *a, **k: calls.setdefault("begin", a),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                module.__file__,
                "--in-dir",
                str(in_dir),
                "--out-dir",
                str(out_dir),
                *extra,
            ],
        )
        calls["exit"] = module.main()
        return calls

    return run
