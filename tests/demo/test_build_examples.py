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
"""Checks offline example generation and source-integrity enforcement."""

import hashlib
import pathlib

import numpy as np
import pytest
import soundfile as sf

import build_examples
import example_audio


def test_builds_the_two_generated_examples_without_external_audio(tmp_path):
    build_examples.build_examples(tmp_path)

    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == ["sample_14.wav", "sample_15.wav"]
    listed = {
        pathlib.Path(row[0]).name for row in example_audio.load_examples(
            example_audio.DEFAULT_DIRECTORY).rows
    }
    assert set(written) <= listed
    for name in written:
        audio, rate = sf.read(tmp_path / name)
        assert rate == 48000
        assert audio.ndim == 1
        assert 8 <= len(audio) / rate <= 12
        assert np.isfinite(audio).all()
        assert 0 < np.max(np.abs(audio)) <= 0.951
    first = [
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in written
    ]

    build_examples.build_examples(tmp_path)

    assert first == [
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in written
    ]


def test_changed_source_is_refused_before_writing(tmp_path, monkeypatch):
    source = tmp_path / "sources"
    source.mkdir()
    (source / "input.flac").write_bytes(b"changed recording")
    recipe = dict(build_examples.RECIPES[0],
                  source="sources/input.flac",
                  source_sha256="0" * 64)
    monkeypatch.setattr(build_examples, "EXAMPLES", tmp_path)
    monkeypatch.setattr(build_examples, "RECIPES", (recipe,))
    destination = tmp_path / "output"

    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        build_examples.build_examples(destination)

    assert not destination.exists()
