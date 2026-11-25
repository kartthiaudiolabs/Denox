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
"""Registry resolution rules.

The tests use no network, only parsing and the local-path shortcut.
"""

import hashlib
import sys
import types

import pytest

from inworld.voice_isolation import registry


def test_hf_spec_parses_repo_file_and_revision():
    assert registry.parse_hf(
        "hf://inworld/voice-isolator/denoiser.safetensors") == (
            "inworld/voice-isolator",
            "denoiser.safetensors",
            None,
        )
    assert registry.parse_hf("hf://org/repo/sub/dir/w.safetensors@v0.1") == (
        "org/repo",
        "sub/dir/w.safetensors",
        "v0.1",
    )


@pytest.mark.parametrize("bad",
                         ["hf://org/repo", "hf:///repo/file", "hf://org//file"])
def test_hf_spec_rejects_incomplete_specs(bad):
    with pytest.raises(ValueError):
        registry.parse_hf(bad)


def test_existing_path_is_returned_untouched(tmp_path):
    p = tmp_path / "weights.safetensors"
    p.write_bytes(b"x")
    assert registry.resolve(str(p)) == str(p)


@pytest.mark.parametrize("name",
                         ["not-a-model", "denoiser", "denoiser-realtime"])
def test_unknown_name_lists_known_models(name):
    with pytest.raises(ValueError, match="voice-isolator-v1"):
        registry.resolve(name)


def test_registry_entries_are_consistent():
    assert set(
        registry.MODELS) == {"voice-isolator-v1", "voice-isolator-v1-realtime"}
    for name, entry in registry.MODELS.items():
        assert entry["repo_id"].count("/") == 1
        assert entry["filename"].endswith(".safetensors")
        assert entry["stage"] == 1
        assert len(entry["sha256"]) == 64, name


def test_a_registry_name_is_never_shadowed_and_always_verified(
        tmp_path, monkeypatch):
    weights = tmp_path / "hub" / "w.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"released weights")
    calls = []

    def hf_hub_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        return str(weights)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    entry = dict(
        registry.MODELS["voice-isolator-v1"],
        sha256=hashlib.sha256(b"released weights").hexdigest(),
        revision="abc123",
    )
    monkeypatch.setitem(registry.MODELS, "voice-isolator-v1", entry)
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "voice-isolator-v1"
    local.write_bytes(b"a local file of the same name")
    assert registry.resolve("voice-isolator-v1") == str(weights)
    assert calls == [(entry["repo_id"], entry["filename"], "abc123")]
    assert registry.resolve("./voice-isolator-v1") == "./voice-isolator-v1"
    weights.write_bytes(b"other weights")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        registry.resolve("voice-isolator-v1")


def test_the_repository_id_resolves_to_the_checkpoint_of_the_mode(
        tmp_path, monkeypatch):
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"released weights")
    calls = []

    def hf_hub_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        return str(weights)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    digest = hashlib.sha256(b"released weights").hexdigest()
    for name in registry.MODE_MODELS.values():
        monkeypatch.setitem(registry.MODELS, name,
                            dict(registry.MODELS[name], sha256=digest))

    paths = [
        registry.resolve(registry.REPO_ID),
        registry.resolve(registry.REPO_ID, "streaming"),
        registry.resolve("voice-isolator-v1", "streaming"),
    ]

    assert registry.REPO_ID == "inworld/voice-isolator"
    assert paths == [str(weights)] * 3
    assert [filename for _, filename, _ in calls] == [
        "denoiser.safetensors", "denoiser_realtime.safetensors",
        "denoiser.safetensors"
    ]
    assert {repo_id for repo_id, _, _ in calls} == {registry.REPO_ID}
    with pytest.raises(ValueError, match="mode must be one of"):
        registry.resolve(registry.REPO_ID, "live")


def test_an_unreleased_name_is_refused_rather_than_loaded_unverified(
        monkeypatch):
    entry = dict(registry.MODELS["voice-isolator-v1"], sha256=None)
    monkeypatch.setitem(registry.MODELS, "voice-isolator-v1", entry)
    with pytest.raises(ValueError, match="not released yet"):
        registry.resolve("voice-isolator-v1")


def test_the_file_hash_is_read_in_blocks(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(b"x" * (3 << 20))
    digest = registry.sha256_file(str(p))
    assert digest == hashlib.sha256(b"x" * (3 << 20)).hexdigest()


def test_a_hub_failure_names_the_file_to_load_by_path(monkeypatch):

    def hf_hub_download(repo_id, filename, revision=None):
        raise OSError(f"404 Client Error: {repo_id}/{filename}@{revision}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    with pytest.raises(RuntimeError, match="by its path") as info:
        registry.resolve("voice-isolator-v1")
    assert "inworld/voice-isolator/denoiser.safetensors" in str(info.value)
    assert isinstance(info.value.__cause__, OSError)


def test_pinned_names_fetch_the_published_files(monkeypatch):
    fetched = []

    def hf_hub_download(repo_id, filename, revision=None):
        fetched.append((repo_id, filename, revision))
        raise OSError("offline")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )

    for former in ("voice-isolator-v0", "voice-isolator-v0-realtime"):
        with pytest.raises(RuntimeError, match="by its path"):
            registry.resolve(former)

    offline = ("inworld/voice-isolator", "denoiser.safetensors",
               registry.PINNED["voice-isolator-v0"]["revision"])
    live = ("inworld/voice-isolator", "denoiser_realtime.safetensors",
            registry.PINNED["voice-isolator-v0-realtime"]["revision"])
    assert fetched == [offline, live]
    assert not set(registry.PINNED) & set(registry.MODELS)


def test_the_v0_names_keep_the_shipped_files_when_v1_changes(
        tmp_path, monkeypatch):
    candidate = tmp_path / "denoiser.safetensors"
    candidate.write_bytes(b"a release candidate")
    calls = []

    def hf_hub_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        return str(candidate)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    monkeypatch.setitem(
        registry.MODELS, "voice-isolator-v1",
        dict(registry.MODELS["voice-isolator-v1"],
             sha256=hashlib.sha256(b"a release candidate").hexdigest()))

    v1 = registry.resolve("voice-isolator-v1")

    assert v1 == str(candidate)
    # the published v0 rows' name refuses the new file rather than load it
    with pytest.raises(RuntimeError,
                       match="Checksum mismatch for voice-isolator-v0"):
        registry.resolve("voice-isolator-v0")
    assert registry.PINNED["voice-isolator-v0"]["sha256"].startswith("647335a6")
    assert len(calls) == 2
