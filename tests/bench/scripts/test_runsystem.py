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
"""Our renderer is held to a wrapper's rules.

Receipts before rendering, whole outputs, resume only under one identity, a
stamp once the selection is complete; linked systems are links.
"""

# pylint: disable=redefined-outer-name

import contextlib
import dataclasses
import hashlib
import json
import os
import socket
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

import common
import receipt
import runsystem
import stamprender

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def test_public_path_is_repo_relative_inside_and_basename_outside(tmp_path):
    inside = os.path.join(REPO, "models", "x.safetensors")
    # registry names pass through
    assert runsystem.public_path("voice-isolator-v1") == "voice-isolator-v1"
    assert runsystem.public_path(None) is None
    outside = tmp_path / "secret_dir" / "ckpt.safetensors"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    assert runsystem.public_path(str(outside)) == "ckpt.safetensors"
    if os.path.exists(inside):
        assert runsystem.public_path(inside) == os.path.join(
            "models", "x.safetensors")


def test_weights_digest_matches_sha256_of_file_and_of_model_pt_in_dir(tmp_path):
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"weights" * 1000)
    digest = runsystem.weights_digest(str(f))
    assert digest == hashlib.sha256(b"weights" * 1000).hexdigest()
    d = tmp_path / "export"
    d.mkdir()
    (d / "model.pt").write_bytes(b"pt")
    assert runsystem.weights_digest(str(d)) == hashlib.sha256(b"pt").hexdigest()


def test_a_folder_is_resumed_only_under_its_own_identity(tmp_path, capsys):
    out = tmp_path / "out"
    ident = {"system": "x", "weights_sha256": {"w": "1"}, "inputs": {"s": 1}}
    other = dict(ident, weights_sha256={"w": "2"})
    # empty: this process's receipt is written, nothing else
    assert runsystem.prepare(str(out), ident, [],
                             False) == receipt.digest(ident)
    assert [d["digest"] for d in receipt.read_all(str(out))
           ] == [receipt.digest(ident)]
    (out / "0.wav").write_bytes(b"")
    # the same identity resumes
    runsystem.prepare(str(out), ident, [], overwrite=False)
    assert (out / "0.wav").exists()
    # another identity's outputs are discarded, what differed named
    (out / ".partial-x.wav").write_bytes(b"")
    runsystem.prepare(str(out), other, [], overwrite=False)
    assert not (out / "0.wav").exists()
    assert not (out / ".partial-x.wav").exists()
    assert "the weights differ" in capsys.readouterr().out
    assert {d["digest"] for d in receipt.read_all(str(out))
           } == {receipt.digest(other)}
    # outputs without a receipt are of unknown origin: nothing is removed
    for path in out.glob(".receipt-*"):
        path.unlink()
    (out / "1.wav").write_bytes(b"")
    with pytest.raises(SystemExit, match="without a receipt"):
        runsystem.prepare(str(out), other, [], overwrite=False)
    assert (out / "1.wav").exists()
    # --overwrite clears outputs and receipts whatever their identity
    runsystem.prepare(str(out), other, [], overwrite=True)
    assert [p.name[:9] for p in out.iterdir()] == [".receipt-"]
    assert not (out / ".prepare-lock").exists()


def _tiny_set(tmp_path, n=3):
    root = tmp_path / "set"
    (root / "mix").mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    rows = []
    for i in range(n):
        x = (torch.randn(int(0.5 * 48000), generator=g) * 0.1).numpy()
        sf.write(root / "mix" / f"clean_{i:03d}.wav", x, 48000, subtype="FLOAT")
        rows.append({
            "id": f"clean_{i:03d}.wav",
            "condition": "clean",
            "mix_path": f"mix/clean_{i:03d}.wav",
        })
    manifest = "\n".join(json.dumps(r) for r in rows) + "\n"
    (root / "manifest.jsonl").write_text(manifest)
    (root / "checksums.json").write_text(
        json.dumps({
            "set_id": "tiny-set-0001",
            "n_items": n,
            "files": {}
        }))
    return root


def _tiny_checkpoint(tmp_path, model):
    # pylint: disable-next=import-outside-toplevel
    import safetensors.torch

    cfg = dataclasses.asdict(model.config)
    st = {k: v.contiguous() for k, v in model.state_dict().items()}
    path = tmp_path / "tiny.safetensors"
    safetensors.torch.save_file(st,
                                str(path),
                                metadata={"config": json.dumps(cfg)})
    return str(path)


@pytest.fixture
def bench(tmp_path, monkeypatch, identity_model):
    root = _tiny_set(tmp_path)
    out_root, results = tmp_path / "out", tmp_path / "results"
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(out_root))
    monkeypatch.setattr(common, "RESULTS", str(results))
    monkeypatch.setattr(common, "DEV", "cpu")
    ckpt = _tiny_checkpoint(tmp_path, identity_model)
    return root, out_root, results, ckpt


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["runsystem.py"] + argv)
    return runsystem.main()


def test_shards_render_once_each_and_the_last_one_stamps(
        bench, monkeypatch, capsys):
    _, out_root, results, ckpt = bench
    base = ["--name", "tiny", "--model", ckpt]
    assert _run(monkeypatch, base + ["--batch", "2", "--shard", "0/2"]) == 0
    # one item still missing: no stamp yet
    assert not (results / "tiny.meta.json").exists()
    assert ("the run that completes the selection stamps it"
            in capsys.readouterr().out)
    assert _run(monkeypatch, base + ["--batch", "2", "--shard", "1/2"]) == 0
    outs = sorted((out_root / "tiny").glob("*.wav"))
    assert [p.name for p in outs] == [f"clean_{i:03d}.wav" for i in range(3)]
    info = sf.info(str(outs[0]))
    assert (info.subtype == "FLOAT" and info.frames == 24000 and
            info.samplerate == 48000)
    meta = json.load(open(results / "tiny.meta.json", encoding="utf-8"))
    assert (meta["set_id"] == "tiny-set-0001" and meta["subset"] == "all" and
            meta["batch"] == 2)
    assert meta["weights"] == "tiny.safetensors"
    assert meta["mode"] == "offline" and meta["lookahead_frames"] == 0
    assert meta["window_frames"] is None and meta["crossfade_frames"] is None
    with open(ckpt, "rb") as fh:
        weights = fh.read()
    assert meta["weights_sha256"] == hashlib.sha256(weights).hexdigest()
    assert (meta["wrapper"] == "bench/scripts/runsystem.py" and
            meta["n_files"] == 3)
    ident = meta["render_receipt"]["identity"]
    assert ident["weights_sha256"] == {
        "tiny.safetensors": meta["weights_sha256"]
    }
    assert list(ident["package_code_sha256"]) == ["inworld.voice_isolation"]
    # one per process: both shards ran here
    assert meta["render_receipt"]["n_receipts"] == 1


def test_a_rerun_resumes_and_another_identity_or_set_build_renders_again(
        bench, monkeypatch, capsys):
    root, out_root, _, ckpt = bench
    base = ["--name", "tiny", "--model", ckpt]
    assert _run(monkeypatch, base) == 0
    outs = sorted((out_root / "tiny").glob("*.wav"))
    before = {p.name: p.stat().st_mtime_ns for p in outs}
    assert _run(monkeypatch, base) == 0
    assert {p.name: p.stat().st_mtime_ns for p in outs} == before
    assert "discarded" not in capsys.readouterr().out
    # other arguments: the outputs are discarded and rendered again
    assert _run(monkeypatch, base + ["--chunk-seconds", "0.25"]) == 0
    again = {p.name: p.stat().st_mtime_ns for p in outs}
    assert set(again) == set(before)
    assert all(again[k] > before[k] for k in again)
    assert "the arguments differ" in capsys.readouterr().out
    # another build of the set likewise
    checksums = {"set_id": "tiny-set-0002"}
    (root / "checksums.json").write_text(json.dumps(checksums))
    assert _run(monkeypatch, base + ["--chunk-seconds", "0.25"]) == 0
    assert "the inputs" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _run(monkeypatch, base + ["--overwrite", "--shard", "0/2"])
    assert _run(monkeypatch, base + ["--overwrite"]) == 0


def test_batches_match_the_one_clip_path(bench, monkeypatch):
    _, out_root, _, ckpt = bench
    assert (_run(
        monkeypatch,
        [
            "--name",
            "batched",
            "--model",
            ckpt,
            "--batch",
            "3",
        ],
    ) == 0)
    assert _run(monkeypatch, ["--name", "single", "--model", ckpt]) == 0
    a = sf.read(str(out_root / "batched" / "clean_000.wav"), dtype="float32")[0]
    b = sf.read(str(out_root / "single" / "clean_000.wav"), dtype="float32")[0]
    assert abs(a - b).max() < 1e-5


def test_explicit_zero_overlap_is_recorded_as_zero(bench, monkeypatch):
    _, _, results, ckpt = bench
    assert (_run(
        monkeypatch,
        [
            "--name",
            "z",
            "--model",
            ckpt,
            "--overlap-seconds",
            "0",
            "--chunk-seconds",
            "0.25",
        ],
    ) == 0)
    meta = json.load(open(results / "z.meta.json", encoding="utf-8"))
    assert meta["overlap_seconds"] == 0.0 and meta["chunk_seconds"] == 0.25


def test_the_probe_folder_is_rendered_under_receipts_naming_its_inputs(
        bench, monkeypatch, tmp_path):
    _, out_root, results, ckpt = bench
    probe = tmp_path / "probe"
    (probe / "in").mkdir(parents=True)
    (probe / "probe.json").write_text(json.dumps({"identity": "probe-1"}))
    g = torch.Generator().manual_seed(1)
    for name in ("longATp4_000__tail.wav", "longATp4_000__future.wav"):
        sf.write(
            probe / "in" / name,
            (torch.randn(36000, generator=g) * 0.1).numpy(),
            48000,
            subtype="FLOAT",
        )
    base = ["--name", "tiny", "--model", ckpt]
    assert _run(monkeypatch, base + ["--inputs", str(probe / "in")]) == 0
    outs = sorted(p.name for p in (out_root / "tiny").glob("*.wav"))
    assert outs == ["longATp4_000__future.wav", "longATp4_000__tail.wav"]
    # a probe render is never stamped
    assert not (results / "tiny.meta.json").exists()
    docs = receipt.read_all(str(out_root / "tiny"))
    assert docs[0]["identity"]["inputs"] == {"probe": "probe-1"}
    # another probe cut: the outputs are discarded and rendered again
    (probe / "probe.json").write_text(json.dumps({"identity": "probe-2"}))
    assert _run(monkeypatch, base + ["--inputs", str(probe / "in")]) == 0
    docs = receipt.read_all(str(out_root / "tiny"))
    assert [d["identity"]["inputs"] for d in docs] == [{"probe": "probe-2"}]
    with pytest.raises(SystemExit):  # only a model or the hint can be probed
        _run(
            monkeypatch,
            ["--name", "INPUT", "--input", "--inputs",
             str(probe / "in")],
        )
    with pytest.raises(SystemExit):
        _run(monkeypatch, base + ["--inputs", str(tmp_path / "nowhere")])


def test_reference_renderer_uses_only_refs_and_never_real_mixtures(
        tmp_path, monkeypatch):
    root = _tiny_set(tmp_path, n=2)
    (root / "ref").mkdir()
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    rows[0]["clean_path"] = "ref/clean_000.wav"
    (root / rows[0]["clean_path"]).write_bytes(b"reference bytes")
    rows[1]["condition"] = "real_noisy"
    (root / "manifest.jsonl").write_text("".join(
        json.dumps(r) + "\n" for r in rows))
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(outputs))
    monkeypatch.setattr(common, "RESULTS", str(tmp_path / "results"))
    assert _run(monkeypatch, ["--name", "REF", "--reference"]) == 0
    assert (outputs / "REF" / rows[0]["id"]).read_bytes() == b"reference bytes"
    assert not (outputs / "REF" / rows[1]["id"]).exists()
    meta = json.load(
        open(tmp_path / "results" / "REF.meta.json", encoding="utf-8"))
    assert meta["reference"] is True and meta["n_files"] == 1
    (root / rows[0]["clean_path"]).unlink()
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["--name", "REF", "--reference"])


def test_a_folder_of_outputs_is_linked_in_only_when_complete(
        tmp_path, monkeypatch):
    root = _tiny_set(tmp_path, n=2)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    sf.write(elsewhere / "clean_000.wav", np.zeros(4800, dtype="float32"),
             48000)
    outputs, results = tmp_path / "outputs", tmp_path / "results"
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(outputs))
    monkeypatch.setattr(common, "RESULTS", str(results))
    argv = ["--name", "theirs", "--outputs", str(elsewhere)]
    # one of two items missing: nothing is linked
    with pytest.raises(SystemExit):
        _run(monkeypatch, argv)
    assert not (outputs / "theirs").exists() or not list(
        (outputs / "theirs").glob("*.wav"))
    sf.write(elsewhere / "clean_001.wav", np.zeros(4800, dtype="float32"),
             48000)
    assert _run(monkeypatch, argv) == 0
    assert sorted(p.name for p in (outputs / "theirs").glob("*.wav")) == [
        "clean_000.wav",
        "clean_001.wav",
    ]
    assert os.path.realpath(outputs / "theirs" / "clean_000.wav") == str(
        elsewhere / "clean_000.wav")
    meta = json.load(open(results / "theirs.meta.json", encoding="utf-8"))
    assert (meta["mechanism"] == "rendered elsewhere" and
            meta["source"] == "elsewhere")
    assert meta["set_id"] == "tiny-set-0001"
    assert meta["source_sha256"] == runsystem.folder_digest(str(elsewhere))
    # the outputs digest a computed render's stamp records, over the same files
    assert meta["outputs_sha256"] == stamprender.outputs_digest(
        sorted((outputs / "theirs").glob("*.wav")), str(outputs / "theirs"))
    # another folder of the same name cannot resume into the same system
    other = tmp_path / "b" / "elsewhere"
    other.mkdir(parents=True)
    for name in ("clean_000.wav", "clean_001.wav"):
        sf.write(other / name, np.full(4800, 0.5, dtype="float32"), 48000)
    with pytest.raises(SystemExit, match="not links"):
        _run(monkeypatch, ["--name", "theirs", "--outputs", str(other)])
    assert os.path.realpath(outputs / "theirs" / "clean_000.wav") == str(
        elsewhere / "clean_000.wav")
    assert (_run(
        monkeypatch,
        ["--name", "theirs", "--outputs",
         str(other), "--overwrite"],
    ) == 0)
    assert os.path.realpath(outputs / "theirs" / "clean_000.wav") == str(
        other / "clean_000.wav")
    with open(results / "theirs.meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["source_sha256"] == runsystem.folder_digest(str(other))


def test_a_panel_subset_renders_only_its_items(tmp_path, monkeypatch):
    root = _tiny_set(tmp_path, n=3)
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(common, "EVAL", str(root))
    monkeypatch.setattr(common, "OUT_ROOT", str(outputs))
    monkeypatch.setattr(common, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(common, "SUBSET", "gate")
    monkeypatch.setattr(common,
                        "panel_members",
                        lambda name, root=None: {"clean_001.wav"})
    assert _run(monkeypatch, ["--name", "INPUT", "--input"]) == 0
    names = [p.name for p in (outputs / "INPUT").glob("*.wav")]
    assert names == ["clean_001.wav"]
    with open(tmp_path / "results" / "INPUT.meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["subset"] == "gate"


def test_a_folder_another_identity_is_still_rendering_is_refused(
        bench, monkeypatch, capsys):
    """A live foreign render is never discarded under its process."""
    _, out_root, _, ckpt = bench
    base = ["--name", "tiny", "--model", ckpt]
    assert _run(monkeypatch, base) == 0
    out_dir = out_root / "tiny"
    # a receipt of another identity whose process (this one's parent, alive
    # for the test's duration) still runs
    own = json.load(open(receipt.own_path(str(out_dir)), encoding="utf-8"))
    own["digest"] = "0" * 64
    own["identity"] = dict(own["identity"], system="other")
    own["pid"] = os.getppid()
    own["process_start"] = receipt.process_start(os.getppid())
    (out_dir / ".receipt-elsewhere-424242.json").write_text(json.dumps(own))

    with pytest.raises(SystemExit, match="still rendering"):
        _run(monkeypatch, base + ["--chunk-seconds", "0.25"])
    # --overwrite takes the folder regardless
    assert _run(monkeypatch,
                base + ["--chunk-seconds", "0.25", "--overwrite"]) == 0
    assert "still rendering" not in capsys.readouterr().out


def test_a_render_whose_receipt_is_gone_stops_before_writing(bench):
    root, out_root, _, _ = bench
    files = sorted(str(p) for p in (root / "mix").glob("*.wav"))
    out_dir = out_root / "taken"
    ident = {"system": "taken", "digest": "x"}
    runsystem.prepare(str(out_dir), ident, [], False)
    os.remove(receipt.own_path(str(out_dir)))  # another identity took over

    with pytest.raises(SystemExit, match="receipt is gone"):
        runsystem.render(files, str(out_dir), lambda x: x, None, 1)
    assert not list(out_dir.glob("*.wav"))


def test_linking_the_input_over_a_render_leaves_no_receipt_to_resume(
        bench, monkeypatch, capsys):
    """After --input --overwrite, the model's command renders again instead of
    stamping the linked mixtures as its own."""
    _, out_root, results, ckpt = bench
    model = ["--name", "tiny", "--model", ckpt]
    assert _run(monkeypatch, model) == 0
    assert _run(monkeypatch, ["--name", "tiny", "--input", "--overwrite"]) == 0
    assert all(p.is_symlink() for p in (out_root / "tiny").glob("*.wav"))

    assert _run(monkeypatch, model) == 0

    outputs = sorted((out_root / "tiny").glob("*.wav"))
    assert outputs and not any(p.is_symlink() for p in outputs)
    meta = json.load(open(results / "tiny.meta.json", encoding="utf-8"))
    assert meta["weights"].endswith("tiny.safetensors")
    assert "3 to do" in capsys.readouterr().out


def test_an_export_folder_renders_the_checkpoint_it_holds(
        bench, monkeypatch, identity_model, tmp_path):
    _, _, results, _ = bench
    export = tmp_path / "export"
    export.mkdir()
    torch.save(
        {
            "model": identity_model.state_dict(),
            "config": dataclasses.asdict(identity_model.config)
        }, export / "model.pt")

    assert _run(monkeypatch, ["--name", "exp", "--model", str(export)]) == 0

    meta = json.load(open(results / "exp.meta.json", encoding="utf-8"))
    assert meta["weights_sha256"] == runsystem.weights_digest(str(export))


def test_the_stamp_names_the_window_the_render_used(bench, monkeypatch):
    _, _, results, ckpt = bench

    assert _run(monkeypatch, [
        "--name", "tiny", "--model", ckpt, "--chunk-seconds", "0.25",
        "--overlap-seconds", "0.05"
    ]) == 0

    meta = json.load(open(results / "tiny.meta.json", encoding="utf-8"))
    assert "0.25 s windows, 0.05 s crossfades" in meta["mechanism"]
    assert meta["chunk_seconds"] == 0.25


def _watched_lock(monkeypatch):
    """Wraps runsystem.taken so a test sees when the folder's lock is held."""
    held, real = [], runsystem.taken

    @contextlib.contextmanager
    def taken(out_dir, wait=300.0):
        with real(out_dir, wait):
            held.append(out_dir)
            try:
                yield
            finally:
                held.pop()

    monkeypatch.setattr(runsystem, "taken", taken)
    return held


def test_every_output_is_published_under_the_folders_lock(bench, monkeypatch):
    """The receipt re-check and the write happen inside one hold of the lock
    a takeover (prepare with another identity, or --overwrite) must take
    first, so the takeover cannot slip in between them."""
    root, out_root, _, _ = bench
    files = sorted(str(p) for p in (root / "mix").glob("*.wav"))
    out_dir = out_root / "locked"
    runsystem.prepare(str(out_dir), {"system": "locked"}, [], False)
    held = _watched_lock(monkeypatch)
    real_save, seen = runsystem.writer.save, []

    def save(path, samples, sr):
        seen.append(
            (bool(held), os.path.exists(receipt.own_path(str(out_dir)))))
        real_save(path, samples, sr)

    monkeypatch.setattr(runsystem.writer, "save", save)

    assert runsystem.render(files, str(out_dir), lambda x: x, None, 1) == 3

    assert seen == [(True, True)] * 3
    assert not (out_dir / ".prepare-lock").exists()


def test_linking_in_outputs_holds_the_folders_lock(bench, monkeypatch):
    _, out_root, _, ckpt = bench
    assert _run(monkeypatch, ["--name", "tiny", "--model", ckpt]) == 0
    held = _watched_lock(monkeypatch)
    real_symlink, seen = os.symlink, []

    def symlink(src, dst):
        seen.append(bool(held))
        real_symlink(src, dst)

    monkeypatch.setattr(os, "symlink", symlink)

    assert _run(monkeypatch, ["--name", "tiny", "--input", "--overwrite"]) == 0

    assert seen == [True] * 3
    assert not (out_root / "tiny" / ".prepare-lock").exists()


def test_a_live_process_is_known_by_its_pid_and_start_time(monkeypatch):
    """A reused pid is another process; a process the signal may not reach is
    alive; an old receipt without a start time is judged by its pid alone."""
    host, parent = socket.gethostname(), os.getppid()
    start = receipt.process_start(parent)
    if start is None:
        pytest.skip("needs a process's start time, which only Linux's /proc"
                    " gives")
    live = {"host": host, "pid": parent, "process_start": start}
    reused = {"host": host, "pid": parent, "process_start": "1"}
    old = {"host": host, "pid": parent}
    gone = {"host": host, "pid": 2**22 - 1, "process_start": start}

    assert runsystem.process_alive(live)
    assert runsystem.process_alive(old)
    assert not runsystem.process_alive(reused)
    assert not runsystem.process_alive(gone)
    assert not runsystem.process_alive({"host": "elsewhere", "pid": parent})

    def denied(pid, sig):
        raise PermissionError

    monkeypatch.setattr(os, "kill", denied)
    assert runsystem.process_alive(live)
