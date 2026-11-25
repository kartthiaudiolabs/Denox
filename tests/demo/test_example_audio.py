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
"""Only the decodable audio files of the examples folder are offered."""

import numpy as np
import pytest
import soundfile as sf

import example_audio


def test_load_lists_audio_files_in_name_order_labelled_by_name(
        example_directory):
    (example_directory / "private-note.txt").write_text("Not for the browser.",
                                                        encoding="utf-8")
    (example_directory / "nested").mkdir()
    (example_directory / "nested" / "inner.wav").write_bytes(
        (example_directory / "synthetic_silence.wav").read_bytes())
    for name in ("sample_10.flac", "sample_2.wav"):
        sf.write(example_directory / name, np.zeros(4800), 48000)

    examples = example_audio.load_examples(example_directory)

    assert examples.rows == [[str(example_directory / "sample_2.wav")],
                             [str(example_directory / "sample_10.flac")],
                             [str(example_directory / "synthetic_silence.wav")]]
    assert examples.labels == ["sample_2", "sample_10", "synthetic_silence"]


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_load_requires_a_folder_with_audio(tmp_path, kind):
    directory = tmp_path / "examples"
    if kind == "empty":
        directory.mkdir()
        (directory / "note.txt").write_text("no audio", encoding="utf-8")

    with pytest.raises(ValueError, match="No example"):
        example_audio.load_examples(directory)


def test_load_rejects_linked_audio(example_directory):
    target = example_directory.parent / "private.wav"
    target.write_bytes(
        (example_directory / "synthetic_silence.wav").read_bytes())
    (example_directory / "linked.wav").symlink_to(target)

    with pytest.raises(ValueError, match="regular"):
        example_audio.load_examples(example_directory)


def test_bundled_examples_load_with_distinct_labels():
    examples = example_audio.load_examples(example_audio.DEFAULT_DIRECTORY)

    assert examples.rows and len(examples.rows) == len(examples.labels)
    assert len(set(examples.labels)) == len(examples.labels)
    for row in examples.rows:
        info = sf.info(row[0])
        assert (info.samplerate, info.channels) == (48000, 1)


def _replace_audio(directory, name, content):
    (directory / "synthetic_silence.wav").unlink()
    (directory / name).write_bytes(content)


@pytest.mark.parametrize("suffix", ["wav", "flac", "mp3", "ogg"])
@pytest.mark.parametrize(
    "content",
    [b'{"private": "sidecar content"}', b"<html>Private note</html>"])
def test_load_rejects_disguised_sidecars(example_directory, suffix, content):
    _replace_audio(example_directory, "disguised." + suffix, content)

    with pytest.raises(ValueError, match="Cannot decode example audio"):
        example_audio.load_examples(example_directory)


@pytest.mark.parametrize("suffix,format_name,subtype",
                         [("wav", "WAV", "PCM_16"), ("flac", "FLAC", "PCM_16"),
                          ("mp3", "MP3", "MPEG_LAYER_III"),
                          ("ogg", "OGG", "VORBIS")])
def test_load_accepts_decodable_advertised_formats(example_directory, tmp_path,
                                                   suffix, format_name,
                                                   subtype):
    if not sf.check_format(format_name, subtype):
        pytest.skip(f"libsndfile cannot encode the {format_name} test fixture")
    source = tmp_path / ("generated." + suffix)
    sf.write(source, np.zeros(4800), 48000, format=format_name, subtype=subtype)
    _replace_audio(example_directory, source.name, source.read_bytes())

    examples = example_audio.load_examples(example_directory)

    assert examples.rows == [[str(example_directory / source.name)]]


def test_load_rejects_valid_audio_with_a_mismatched_suffix(example_directory):
    content = (example_directory / "synthetic_silence.wav").read_bytes()
    _replace_audio(example_directory, "wrong.flac", content)

    with pytest.raises(ValueError, match="format does not match"):
        example_audio.load_examples(example_directory)


@pytest.mark.parametrize("kind", ["truncated", "empty", "nonfinite"])
def test_load_rejects_invalid_audio_payloads(example_directory, tmp_path, kind):
    source = tmp_path / "invalid.wav"
    samples = np.array([float("nan")]) if kind == "nonfinite" else np.zeros(0)
    sf.write(source, samples, 48000, subtype="FLOAT")
    content = source.read_bytes()
    if kind == "truncated":
        content = content[:14]
    _replace_audio(example_directory, source.name, content)

    with pytest.raises(ValueError, match="decode|empty|non-finite"):
        example_audio.load_examples(example_directory)
