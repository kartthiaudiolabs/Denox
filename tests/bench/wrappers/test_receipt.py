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
"""A receipt names everything a render depends on.

The code, the release, the weights, the packages, the device, the arguments and
the inputs of a render, and it changes when any of them does.
"""

# pylint: disable=redefined-outer-name

import hashlib
import json
import os
import subprocess
import sys

import pytest

import receipt


def _git(path, *args):
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def release(tmp_path, monkeypatch, tiny_set):
    """A release checkout with code and weights.

    It comes with its description and a renderer.
    """
    meta = tmp_path / "meta"
    meta.mkdir()
    src = tmp_path / "src" / "upstream"
    (src / "ckpts").mkdir(parents=True)
    (src / "ckpts" / "weights.pt").write_bytes(b"w" * 100)
    (src / "model.py").write_text("LAYERS = 4\n")
    (src / "notes.txt").write_text("not code")
    _git(src, "init", "-q")
    _git(src, "add", ".")
    _git(src, "commit", "-q", "-m", "release")
    json.dump(
        {
            "source_dir": "upstream",
            "weight_globs": ["ckpts/*.pt"]
        },
        open(meta / "sys.json", "w", encoding="utf-8"),
    )
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    (wrappers / "helper.py").write_text("import other\n")
    (wrappers / "other.py").write_text("X = 1\n")
    renderer = wrappers / "runsys.py"
    renderer.write_text("import argparse\nimport helper\nimport receipt\n")
    monkeypatch.setattr(receipt, "META_DIR", str(meta))
    monkeypatch.setattr(receipt, "SRC", str(tmp_path / "src"))
    in_dir, _ = tiny_set
    return src, renderer, str(in_dir)


ARGV = [
    "runsys.py",
    "--in-dir",
    "/a/b",
    "--out-dir",
    "/c",
    "--shard",
    "1/4",
    "--device",
    "cpu",
    "--limit",
    "3",
    "--overwrite",
    "--steps",
    "8",
    "--config",
    "/x/y/cfg.yaml",
]


def test_identity_names_code_release_weights_packages_device_args_and_inputs(
    release,):
    src, renderer, in_dir = release
    ident = receipt.identity("sys", str(renderer), ARGV, in_dir, "cpu")
    assert ident["wrapper"] == "runsys.py"
    code = renderer.read_bytes()
    assert ident["wrapper_sha256"] == hashlib.sha256(code).hexdigest()
    # transitively, not receipt
    assert sorted(ident["helpers_sha256"]) == ["helper.py", "other.py"]
    head = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ident["source"]["git_commit"] == head.stdout.strip()
    assert ident["weights_sha256"] == {
        "ckpts/weights.pt": hashlib.sha256(b"w" * 100).hexdigest()
    }
    assert ident["n_packages"] > 0 and len(ident["packages_sha256"]) == 64
    assert ident["device"] == "cpu"
    # the work selection, the device and --overwrite are not the computation;
    # paths lose folders
    assert ident["inference_args"] == ["--steps", "8", "--config", "cfg.yaml"]
    assert ident["inputs"] == {"set_id": "tiny-set"}


def test_every_ingredient_changes_the_digest_and_the_work_selection_does_not(
    release,):
    src, renderer, in_dir = release
    base = receipt.digest(
        receipt.identity("sys", str(renderer), ARGV, in_dir, "cpu"))
    same = [
        "runsys.py",
        "--steps",
        "8",
        "--config",
        "/elsewhere/cfg.yaml",
        "--shard",
        "0/2",
    ]
    assert (receipt.digest(
        receipt.identity("sys", str(renderer), same, in_dir, "cpu")) == base)
    assert (receipt.digest(
        receipt.identity("sys", str(renderer), ["r", "--steps", "16"], in_dir,
                         "cpu")) != base)

    def changed(path, text):
        before = path.read_bytes()
        path.write_bytes(text)
        try:
            return receipt.digest(
                receipt.identity("sys", str(renderer), ARGV, in_dir, "cpu"))
        finally:
            path.write_bytes(before)

    assert changed(src / "ckpts" / "weights.pt", b"v" * 100) != base
    # an edit, uncommitted
    assert changed(src / "model.py", b"LAYERS = 5\n") != base
    # not code or configuration
    assert changed(src / "notes.txt", b"other notes") == base
    # a helper's helper
    assert changed(renderer.parent / "other.py", b"X = 2\n") != base
    assert changed(renderer, b"import argparse\nimport helper\n") != base


def test_the_environment_digest_leaves_the_inputs_out(release, tmp_path):
    _, renderer, in_dir = release
    probe = tmp_path / "probe"
    (probe / "in").mkdir(parents=True)
    (probe / "probe.json").write_text(json.dumps({"identity": "probe-1"}))
    board = receipt.identity("sys", str(renderer), ARGV, in_dir, "cpu")
    cut = receipt.identity("sys", str(renderer), ARGV, str(probe / "in"), "cpu")
    assert cut["inputs"] == {"probe": "probe-1"}
    assert receipt.digest(cut) != receipt.digest(board)
    assert receipt.environment_digest(cut) == receipt.environment_digest(board)


def test_inputs_without_a_manifest_are_identified_by_their_bytes(tmp_path):
    folder = tmp_path / "loose" / "in"
    folder.mkdir(parents=True)
    (folder / "a.wav").write_bytes(b"one")
    first = receipt.inputs(str(folder))
    (folder / "a.wav").write_bytes(b"two")
    assert (first != receipt.inputs(str(folder)) and
            list(first) == ["files_sha256"])


def test_missing_weights_release_or_device_are_errors_never_empty_records(
        release, monkeypatch):
    src, renderer, in_dir = release
    (src / "ckpts" / "weights.pt").unlink()
    with pytest.raises(FileNotFoundError, match="no weights match"):
        receipt.identity("sys", str(renderer), ARGV, in_dir, "cpu")
    monkeypatch.setattr(receipt, "SRC", str(src.parent / "nowhere"))
    with pytest.raises(FileNotFoundError):
        receipt.identity(
            "sys",
            str(renderer),
            ARGV,
            in_dir,
            "cpu",
            desc={"source_dir": "upstream"},
        )
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        receipt.resolve_device("cuda:0")


def test_a_given_description_code_list_and_weights_replace_the_defaults(
        release, tmp_path):
    _, renderer, in_dir = release
    package = tmp_path / "pkgroot" / "fakepackage"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("X = 1\n")
    checkpoint = tmp_path / "ckpt.safetensors"
    checkpoint.write_bytes(b"c")
    sys.path.insert(0, str(package.parent))
    try:
        ident = receipt.identity(
            "anything",
            str(renderer),
            ARGV,
            in_dir,
            "cpu",
            weights=(str(checkpoint),),
            desc={"package_dirs": ["fakepackage"]},
            code=(str(renderer.parent / "other.py"),),
        )
        assert ident["source"] is None and list(
            ident["helpers_sha256"]) == ["other.py"]
        assert ident["weights_sha256"] == {
            "ckpt.safetensors": hashlib.sha256(b"c").hexdigest()
        }
        before = ident["package_code_sha256"]["fakepackage"]
        (package / "__init__.py").write_text("X = 2\n")
        again = receipt.identity(
            "anything",
            str(renderer),
            ARGV,
            in_dir,
            "cpu",
            desc={"package_dirs": ["fakepackage"]},
        )
        assert again["package_code_sha256"]["fakepackage"] != before
        with pytest.raises(ModuleNotFoundError):
            receipt.identity(
                "anything",
                str(renderer),
                ARGV,
                in_dir,
                "cpu",
                desc={"package_dirs": ["no_such_package_here"]},
            )
    finally:
        sys.path.remove(str(package.parent))


def test_begin_writes_one_receipt_per_process(release, tmp_path):
    _, renderer, in_dir = release
    out = tmp_path / "out"
    digest = receipt.begin("sys", str(renderer), str(out), ARGV, in_dir, "cpu")
    docs = receipt.read_all(str(out))
    assert len(docs) == 1 and docs[0]["digest"] == digest == receipt.digest(
        docs[0]["identity"])
    assert docs[0]["environment_digest"] == receipt.environment_digest(
        docs[0]["identity"])
    assert docs[0]["pid"] == os.getpid() and docs[0]["argv"] == [
        "--steps",
        "8",
        "--config",
        "cfg.yaml",
    ]
    assert any(p.startswith("numpy==") for p in docs[0]["packages"])
    receipt.begin("sys", str(renderer), str(out), ARGV, in_dir, "cpu")
    assert len(receipt.read_all(str(out))) == 1  # one process, one file


def test_the_probe_prints_what_the_renderer_would_record(release, capsys):
    _, renderer, in_dir = release
    assert (receipt.main([
        "receipt.py",
        "probe",
        "sys",
        str(renderer),
        "--in-dir",
        in_dir,
        "--device",
        "cpu",
        "--",
        "--steps",
        "8",
        "--config",
        "/x/y/cfg.yaml",
    ]) == 0)
    doc = json.loads(capsys.readouterr().out)
    assert doc["digest"] == receipt.digest(
        receipt.identity("sys", str(renderer), ARGV, in_dir, "cpu"))
    assert doc["environment_digest"] == receipt.environment_digest(
        doc["identity"])
