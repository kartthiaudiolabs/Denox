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
"""Golden parity against the reference implementation.

Fixtures live in tests/fixtures/<name>/{meta.json, input.wav, expected.npy}
and are rendered by the training-side reference implementation. ``meta.json``
names the model (registry name or path, overridable with
VOICE_ISOLATION_GOLDEN_MODEL), the mode (``offline`` or ``streaming`` with
``lookahead_frames``), the window, the tolerance and, for a fixture rendered
through the reference's stateful runner, ``runner`` with its ``block_frames``
and the piece sizes pushed. Output parity is skipped when the model is absent;
input reproducibility and fixture hashes are checked without model weights.
"""

import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import isolator

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    "fixtures",
)
CASES = sorted(glob.glob(os.path.join(FIXTURES, "*", "meta.json")))

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def _model_path(meta):
    """Registry name, absolute path, or path relative to the repository root."""
    p = os.environ.get("VOICE_ISOLATION_GOLDEN_MODEL") or meta.get("model")
    if p and "/" in p and not os.path.isabs(p) and not os.path.exists(p):
        p = os.path.join(REPO, p)
    if not p or (not os.path.exists(p) and "/" in p):
        pytest.skip("golden model checkpoint not available")
    want = meta.get("model_sha256")
    if want and os.path.exists(p) and not os.path.isdir(p):
        have = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if have != want:
            pytest.skip(
                f"checkpoint on disk ({have[:12]}) is not the fixtures' model"
                f" ({want[:12]}); regenerate the goldens with the independent"
                " reference before checking parity")
    return p


def _stream(vi, x, runner):
    # a fixture rendered by the plain stream names no window: 0 turns the
    # session's default windows off
    stream = vi.stream(block_frames=int(runner.get("block_frames", 4)),
                       cache_frames=runner.get("cache_frames"),
                       window_frames=runner.get("window_frames", 0),
                       crossfade_frames=runner.get("crossfade_frames"))
    sizes = [int(s) for s in runner.get("push", [1920])]
    pieces, pos, index = [], 0, 0
    while pos < len(x):
        step = sizes[index % len(sizes)]
        pieces.append(stream.push(x[pos:pos + step]))
        pos += step
        index += 1
    pieces.append(stream.flush())
    return torch.cat(pieces)


def test_fixture_inputs_are_reproducible(tmp_path):
    generator = os.path.join(FIXTURES, "generate_inputs.py")

    subprocess.run([sys.executable, generator, str(tmp_path)], check=True)

    for case in CASES:
        directory = pathlib.Path(case).parent
        seconds = directory.name.rsplit("_", 1)[1]
        recorded = (directory / "input.wav").read_bytes()
        regenerated = (tmp_path / f"{seconds}.wav").read_bytes()
        assert recorded == regenerated


@pytest.mark.parametrize(
    "meta_path",
    CASES,
    ids=[os.path.basename(os.path.dirname(c)) for c in CASES])
def test_fixture_files_match_their_recorded_hashes(meta_path):
    directory = pathlib.Path(meta_path).parent
    meta = json.loads(pathlib.Path(meta_path).read_text(encoding="utf-8"))

    actual = {
        "input_sha256":
            hashlib.sha256((directory / "input.wav").read_bytes()).hexdigest(),
        "expected_sha256":
            hashlib.sha256(
                (directory / "expected.npy").read_bytes()).hexdigest(),
    }

    assert actual == {key: meta[key] for key in actual}


@pytest.mark.parametrize(
    "meta_path",
    CASES,
    ids=[os.path.basename(os.path.dirname(c)) for c in CASES],
)
def test_reference_signal_rejects_silence_and_half_level(meta_path):
    directory = pathlib.Path(meta_path).parent
    meta = json.loads(pathlib.Path(meta_path).read_text(encoding="utf-8"))
    expected = np.load(directory / "expected.npy")

    silence_error = np.abs(expected).max()
    half_level_error = np.abs(expected - 0.5 * expected).max()

    assert silence_error > meta["tolerance"]
    assert half_level_error > meta["tolerance"]


@pytest.mark.parametrize(
    "meta_path",
    CASES,
    ids=[os.path.basename(os.path.dirname(c)) for c in CASES],
)
def test_golden_parity(meta_path):
    d = os.path.dirname(meta_path)
    meta = json.load(open(meta_path, encoding="utf-8"))
    vi = isolator.VoiceIsolator.load(
        _model_path(meta),
        device="cpu",
        precision=meta.get("precision", "fp32"),
        mode=meta.get("mode", "offline"),
        lookahead_frames=int(meta.get("lookahead_frames", 0)),
    )
    x, sr = audio_lib.load_audio(os.path.join(d, "input.wav"))
    if meta.get("runner"):
        assert sr == vi.sample_rate
        y = _stream(vi, x, meta["runner"])
    else:
        y = vi.isolate(
            x,
            sr,
            output_sample_rate=48000,
            chunk_seconds=meta.get("chunk_seconds"),
            overlap_seconds=meta.get("overlap_seconds"),
        )
    expected = torch.from_numpy(np.load(os.path.join(d, "expected.npy")))
    # the whole render, sample for sample: a truncated output is a failure
    assert len(y) == len(
        expected), f"{len(y)} samples, expected {len(expected)}"
    err = (y - expected).abs().max().item()
    assert err <= meta.get("tolerance", 1e-4), f"max abs error {err}"
