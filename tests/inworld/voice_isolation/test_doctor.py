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
"""The `voice-isolate-doctor` readiness check.

No network and no released weights: the Hub cache and the registry's fetch
are replaced, and ``--full`` runs a tiny checkpoint.
"""

# pylint: disable=redefined-outer-name,unused-argument

import hashlib
import json
import sys

import huggingface_hub
from huggingface_hub import errors as hub_errors
import pytest
import torch

from inworld.voice_isolation import doctor
from inworld.voice_isolation import isolator
from inworld.voice_isolation import registry


@pytest.fixture
def empty_cache(monkeypatch):
    """No released weights in the Hugging Face cache."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *args, **kwargs: None)


@pytest.fixture
def on_cpu(monkeypatch):
    """Inference picks the CPU, as on a machine with no GPU."""
    monkeypatch.setattr(isolator, "default_device", lambda: torch.device("cpu"))


def _lines(capsys):
    return capsys.readouterr().out.strip().splitlines()


def _hub_error(cls):
    """A Hub HTTP error, built bare: its constructor needs a live response."""
    return cls.__new__(cls, "401 Client Error")


def test_default_level_without_weights_is_ready_and_succeeds(
        empty_cache, on_cpu, capsys):

    status = doctor.main([])

    lines = _lines(capsys)
    assert status == 0
    assert lines[-1] == ("Ready to install weights: run voice-isolate-doctor"
                         " --full")
    assert ("INFO  voice-isolator-v1: not downloaded yet; run with --full"
            in lines)
    assert not any(line.startswith("FAIL") for line in lines)
    assert any(line.startswith("INFO  torch: torch ") for line in lines)


@pytest.mark.parametrize(
    "platform_name, machine, torch_version, chosen, level",
    [
        ("darwin", "arm64", "2.5.1", False, "WARN"),
        ("darwin", "arm64", "2.6.0", False, "INFO"),
        ("darwin", "arm64", "2.5.1", True, "INFO"),
        ("linux", "x86_64", "2.5.1", False, "INFO"),
    ],
)
def test_an_apple_silicon_mac_left_on_the_cpu_by_an_old_torch_warns(
        empty_cache, on_cpu, monkeypatch, capsys, platform_name, machine,
        torch_version, chosen, level):
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(doctor.platform, "machine", lambda: machine)
    monkeypatch.setattr(torch, "__version__", torch_version)
    argv = ["--device", "cpu"] if chosen else []

    status = doctor.main(argv)

    (device,) = [line for line in _lines(capsys) if " device: " in line]
    assert status == 0
    assert device.startswith(level)
    if level == "WARN":
        assert "upgrade torch to 2.6 or later" in device


def test_the_device_line_names_the_device_inference_picks(
        empty_cache, monkeypatch, capsys):
    monkeypatch.setattr(isolator, "default_device", lambda: torch.device("mps"))

    doctor.main([])

    assert "INFO  device: inference runs on mps" in _lines(capsys)


@pytest.mark.parametrize("matches", [True, False])
def test_cached_weights_are_checked_against_the_registry(
        tmp_path, on_cpu, monkeypatch, capsys, matches):
    cached = tmp_path / "denoiser.safetensors"
    cached.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights" if matches else b"other").hexdigest()
    entry = dict(registry.MODELS["voice-isolator-v1"], sha256=digest)
    monkeypatch.setattr(registry, "MODELS", {"voice-isolator-v1": entry})
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *args, **kwargs: str(cached))

    status = doctor.main([])

    lines = _lines(capsys)
    if matches:
        assert status == 0
        assert f"OK    voice-isolator-v1: cached at {cached}" in lines
        assert lines[-1].startswith("Ready: run voice-isolate-doctor --full")
    else:
        assert status == 1
        assert (f"FAIL  voice-isolator-v1: checksum mismatch: delete {cached}"
                " and run again") in lines
        assert lines[-1] == "1 problem: see the FAIL lines above"


def test_full_runs_both_given_checkpoints_and_is_all_set(
        tiny_ckpt, empty_cache, capsys):

    status = doctor.main([
        "--full", "--offline-model", tiny_ckpt, "--online-model", tiny_ckpt,
        "--device", "cpu"
    ])

    lines = _lines(capsys)
    assert status == 0
    assert lines.count(f"OK    {tiny_ckpt}: loads and runs on cpu") == 2
    assert lines[-1] == (f"All set: {tiny_ckpt} and {tiny_ckpt} on cpu"
                         f" (torch {torch.__version__})")


@pytest.mark.parametrize(
    "cause, message",
    [
        (_hub_error(hub_errors.RepositoryNotFoundError),
         "inworld/voice-isolator is not reachable: pass --offline-model and"
         " --online-model with local .safetensors files"),
        (_hub_error(hub_errors.GatedRepoError),
         "inworld/voice-isolator is not reachable"),
        (hub_errors.LocalEntryNotFoundError("connection error"),
         "cannot reach huggingface.co: check the network, or pass a local"
         " file"),
        (None, "checksum mismatch: delete the cached file and run again"),
    ],
)
def test_full_names_why_the_released_weights_could_not_be_fetched(
        empty_cache, on_cpu, monkeypatch, capsys, cause, message):

    def resolve(name_or_path, mode="offline"):
        if cause is None:
            raise RuntimeError(f"Checksum mismatch for {name_or_path}:"
                               " expected a, got b.")
        raise RuntimeError("could not be fetched from the Hub") from cause

    monkeypatch.setattr(registry, "resolve", resolve)

    status = doctor.main(["--full"])

    lines = _lines(capsys)
    failed = [line for line in lines if line.startswith("FAIL")]
    assert status == 1
    assert [line.split(": ", 1)[0] for line in failed
           ] == ["FAIL  voice-isolator-v1", "FAIL  voice-isolator-v1-realtime"]
    assert all(message in line for line in failed)
    assert lines[-1] == "2 problems: see the FAIL lines above"


@pytest.mark.parametrize(
    "error, message",
    [
        (_hub_error(hub_errors.RepositoryNotFoundError),
         "not reachable: check the location and your access to it, or pass a"
         " local .safetensors file"),
        (hub_errors.LocalEntryNotFoundError("connection error"),
         "cannot reach huggingface.co: check the network, or pass a local"
         " file"),
        (OSError("disk full"),
         "could not fetch hf://org/repo/w.safetensors: disk full"),
    ],
)
def test_full_reports_an_hf_location_that_cannot_be_fetched(
        tiny_ckpt, empty_cache, monkeypatch, capsys, error, message):
    spec = "hf://org/repo/w.safetensors"

    def hf_hub_download(repo_id, filename, revision=None):
        raise error

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)

    status = doctor.main([
        "--full", "--offline-model", spec, "--online-model", tiny_ckpt,
        "--device", "cpu"
    ])

    lines = _lines(capsys)
    assert status == 1
    assert f"FAIL  {spec}: {message}" in lines
    assert f"OK    {tiny_ckpt}: loads and runs on cpu" in lines
    assert lines[-1] == "1 problem: see the FAIL lines above"


def test_json_still_reports_when_an_hf_location_cannot_be_fetched(
        tiny_ckpt, empty_cache, monkeypatch, capsys):

    def hf_hub_download(repo_id, filename, revision=None):
        raise _hub_error(hub_errors.RepositoryNotFoundError)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)

    status = doctor.main([
        "--json", "--full", "--offline-model", "hf://org/repo/w.safetensors",
        "--online-model", tiny_ckpt, "--device", "cpu"
    ])

    report = json.loads(capsys.readouterr().out)
    failed = [c for c in report["checks"] if c["level"] == "FAIL"]
    assert status == 1
    assert report["ok"] is False
    assert [c["name"] for c in failed] == ["hf://org/repo/w.safetensors"]


def test_full_reports_a_checkpoint_that_does_not_load(tmp_path, tiny_ckpt,
                                                      empty_cache, capsys):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a checkpoint")

    status = doctor.main([
        "--full", "--offline-model",
        str(broken), "--online-model", tiny_ckpt, "--device", "cpu"
    ])

    lines = _lines(capsys)
    (failed,) = [line for line in lines if line.startswith("FAIL")]
    assert status == 1
    assert failed.startswith(f"FAIL  {broken}: failed on cpu: ")
    assert "try --device cpu" not in failed
    assert f"OK    {tiny_ckpt}: loads and runs on cpu" in lines


def test_json_carries_the_same_checks_and_summary(empty_cache, on_cpu, capsys):
    doctor.main([])
    text = _lines(capsys)

    status = doctor.main(["--json"])

    report = json.loads(capsys.readouterr().out)
    assert status == 0
    assert report["ok"] is True
    assert report["summary"] == text[-1]
    rows = [(c["level"], c["name"], c["message"]) for c in report["checks"]]
    assert [f"{level:<4}  {name}: {message}" for level, name, message in rows
           ] == text[:-1]
