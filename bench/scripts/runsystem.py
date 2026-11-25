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
"""Renders the Inworld isolator, or a reference system, over the eval set.

Writes ``<BENCH_OUT>/<name>/<id>.wav`` in the file layout of every external
wrapper, so the scorer compares systems, not harnesses, and stamps
``<BENCH_RESULTS>/<name>.meta.json``.

    runsystem.py --name voice_isolator_v0 --model voice-isolator-v0
    runsystem.py --name <tag> --model ckpt.safetensors --chunk-seconds 8
    runsystem.py --name demucs_vocals --demucs   # the separation reference
    runsystem.py --name INPUT --input            # the unprocessed input
    runsystem.py --name REF --reference          # clean references, paired
    runsystem.py --name <tag> --outputs <dir>    # rendered elsewhere

A computed render (``--model``, ``--demucs``) is held to what a wrapper is:
every process writes a receipt (``bench/wrappers/receipt.py``: this script,
the package's code, every checkpoint including the Demucs hint's, the
installed packages, the device, the arguments and the inputs) before it
renders; each output is written whole or not at all; outputs already there
are resumed only when every receipt beside them carries this run's identity,
another identity's outputs (the code, the weights, the packages, the
arguments or the set changed) are discarded and rendered again, with what
differed printed, and outputs without a receipt stop the run; and the process
that completes the selection (the set, or ``BENCH_SUBSET``, with every shard's
outputs) stamps it through ``stamprender.stamp``. ``--overwrite`` first
removes every output and receipt of the system whatever their identity.
``--inputs`` renders another folder (the streaming probe of
``scorestream.py --probe`` or the same-room pair of ``scoreevents.py --pair``)
and stamps nothing: that render is matched to the board's by its receipts.

INPUT, REF and ``--outputs`` link files instead of computing them; their
stamp names the set and, for ``--outputs``, a digest of the linked folder and
of the outputs it answers the set with (``outputs_sha256``, as a computed
render's stamp records it).
"""

import argparse
import contextlib
import glob
import hashlib
import json
import os
import socket
import sys
import time

import torch

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import stamprender

sys.path.insert(0, os.path.join(common.BENCH_DIR, "wrappers"))
import receipt
import writer

SR = common.SR
HTDEMUCS = "955717e8"  # demucs/remote/htdemucs.yaml: the bag's one model
OURS_META = {
    "mechanism":
        ("band-split dual-path network: 46 bands, time attention (RoPE, "
         "exclusive self attention) and dual-path Mamba blocks, per-bin "
         "complex mask + bounded residual; one set of weights runs offline "
         "({chunk:g} s windows, {overlap:g} s crossfades) or streaming (the "
         "stateful session, block by block)"),
    "native_sr":
        48000,
    "fullband":
        True,
    "license_code":
        "Apache-2.0",
    "license_weights":
        "Apache-2.0",
    "commercial_ok":
        True,
    "renormalizes":
        False,
    "notes":
        ("no input level conditioning: the bare model, output gain matches "
         "the input"),
}
DEMUCS_META = {
    "mechanism": "music source separation (HTDemucs v4) - vocals stem only",
    "native_sr": 44100,
    "fullband": True,
    "license_code": "MIT",
    "license_weights": "MIT",
    "commercial_ok": True,
    "renormalizes": False,
    "params_m": 42.0,
    "repo": "https://github.com/adefossez/demucs",
    "notes": "separator, not a denoiser: the music-separation reference.",
    "precision": "fp32",
}


def parse_args() -> argparse.Namespace:
    """Returns the command line, checked."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model",
                   help="registry name or checkpoint (dir, .safetensors, .pt)")
    g.add_argument("--input",
                   action="store_true",
                   help="the unprocessed input as a system")
    g.add_argument(
        "--reference",
        action="store_true",
        help="the clean references as a system",
    )
    g.add_argument(
        "--demucs",
        action="store_true",
        help="the htdemucs vocals stem as a system",
    )
    g.add_argument(
        "--outputs",
        help=("a folder of <id>.wav rendered elsewhere (an "
              "architecture the package cannot load, a hosted system), "
              "linked in as the system"),
    )
    ap.add_argument(
        "--inputs",
        default="",
        help=("render this folder of 48 kHz WAVs instead of the "
              "set's mixtures (the streaming probe or the same-room "
              "pair); with --model or --demucs"),
    )
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--mode",
                    default="offline",
                    choices=("offline", "streaming"),
                    help="offline (every frame sees the whole window) or "
                    "streaming (causal, with --lookahead frames)")
    ap.add_argument("--lookahead",
                    type=int,
                    default=0,
                    help="streaming look-ahead in frames (10 ms each)")
    ap.add_argument("--block-frames",
                    type=int,
                    default=4,
                    help="streaming mode: frames processed together")
    ap.add_argument("--cache-frames",
                    type=int,
                    default=None,
                    help=("streaming mode: frames the time attention "
                          "remembers; default: the training window"))
    ap.add_argument("--window-frames",
                    type=int,
                    default=None,
                    help=("streaming mode: the attention window in frames; "
                          "default: the model's (600); 0: no windows"))
    ap.add_argument("--crossfade-frames",
                    type=int,
                    default=None,
                    help=("streaming mode: frames the two lanes blend over "
                          "before a window boundary; default: the model's "
                          "(200)"))
    ap.add_argument("--chunk-seconds", type=float, default=None)
    ap.add_argument("--overlap-seconds", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--precision",
        default="auto",
        choices=("auto", "fp32", "fp16", "bf16"),
        help=("network precision: auto is bf16 on a GPU (the training "
              "precision) and fp32 on CPU; the stamp records what ran"),
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=1,
        help=("clips per separator pass (equal lengths are "
              "grouped; numerics match --batch 1 up to kernel "
              "rounding)"),
    )
    ap.add_argument(
        "--shard",
        default="",
        help=("i/k: render every k-th item from the i-th (one "
              "process per GPU; the last to finish stamps)"),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=("remove every output and receipt of this system "
              "first, instead of resuming"),
    )
    a = ap.parse_args()
    if (a.name == "REF") != a.reference:
        ap.error("the reserved REF name requires --reference, and vice versa")
    if a.inputs and not (a.model or a.demucs):
        ap.error("--inputs renders the model or the Demucs hint; INPUT, REF "
                 "and linked outputs have no probe")
    if a.overwrite and a.shard:
        ap.error("--overwrite with --shard would remove the other shards' "
                 "work: clear the folder once, then render the shards")
    return a


def selection(a: argparse.Namespace, rows: list) -> tuple[str, list, dict]:
    """Returns (input folder, input files, {name: reference}) a run answers.

    Raises:
        SystemExit: The probe folder is empty, or REF lacks a reference.
    """
    if a.inputs:
        files = sorted(glob.glob(os.path.join(a.inputs, "*.wav")))
        if not files:
            raise SystemExit(f"no WAV files under {a.inputs}")
        return a.inputs, files, {}
    mix = os.path.join(common.EVAL, "mix")
    if a.reference:
        refs = {r["id"]: r["clean_path"] for r in rows if r.get("clean_path")}
        if not refs or any(not os.path.isfile(p) for p in refs.values()):
            raise SystemExit("REF requires every declared clean reference; "
                             "real inputs are not substitutes")
        return mix, [os.path.join(mix, uid) for uid in sorted(refs)], refs
    return mix, sorted(r["mix_path"] for r in rows), {}


def share(files: list, shard: str, limit: int) -> list:
    """Returns this process's part of the selection."""
    if shard:
        index, count = (int(v) for v in shard.split("/"))
        if count < 1 or not 0 <= index < count:
            raise SystemExit(f"--shard {shard}: need 0 <= i < k")
        files = files[index::count]
    return files[:limit] if limit else files


def checkpoint_file(model: str) -> str:
    """Returns the checkpoint file a model argument names.

    A registry name, an ``hf://`` spec or a path; ``model.pt`` inside an
    export folder.
    """
    # pylint: disable-next=import-outside-toplevel
    from inworld.voice_isolation import registry

    path = registry.resolve(model)
    return os.path.join(path, "model.pt") if os.path.isdir(path) else path


def weights_digest(model: str) -> str:
    """Returns the sha256 of the checkpoint behind ``model``."""
    return receipt.sha256_file(checkpoint_file(model))


def public_path(path: str | None) -> str | None:
    """Returns a checkpoint path fit for a published file.

    Relative to the repository inside it, the bare file name outside; registry
    names pass through.
    """
    if not path or not os.path.exists(path):
        return path
    full = os.path.abspath(path)
    repo = os.path.dirname(common.BENCH_DIR)
    if full.startswith(repo + os.sep):
        return os.path.relpath(full, repo)
    return os.path.basename(full)


def demucs_weights() -> list:
    """Returns the htdemucs checkpoint the hint frontend loaded.

    Demucs keeps it, hash-checked on download, in torch hub's cache.
    """
    hub = os.path.join(torch.hub.get_dir(), "checkpoints")
    found = sorted(glob.glob(os.path.join(hub, f"{HTDEMUCS}-*.th")))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one htdemucs checkpoint under {hub}, found {len(found)}")
    return found


def ours(a: argparse.Namespace) -> tuple:
    """Returns the isolator and what identifies it.

    A tuple (run, run_many, description, weights, packages).
    """
    # pylint: disable-next=import-outside-toplevel
    from inworld import voice_isolation

    vi = voice_isolation.VoiceIsolator.load(
        checkpoint_file(a.model),
        stage2=checkpoint_file(a.stage2) if a.stage2 else None,
        device=common.DEV,
        precision=a.precision,
        mode=a.mode,
        lookahead_frames=a.lookahead,
        block_frames=a.block_frames,
        cache_frames=a.cache_frames,
        window_frames=a.window_frames,
        crossfade_frames=a.crossfade_frames,
    )
    chunk, overlap = vi.model.default_window()
    desc = dict(OURS_META)
    desc["mechanism"] = desc["mechanism"].format(
        chunk=chunk if a.chunk_seconds is None else a.chunk_seconds,
        overlap=overlap if a.overlap_seconds is None else a.overlap_seconds)
    desc.update(
        weights=public_path(a.model),
        weights_sha256=weights_digest(a.model),
        stage2=public_path(a.stage2),
        stage2_sha256=weights_digest(a.stage2) if a.stage2 else None,
        alpha=a.alpha,
        mode=vi.mode,
        lookahead_frames=vi.lookahead_frames,
        latency_samples=vi.latency_samples()
        if vi.mode == "streaming" else None,
        block_frames=vi.block_frames if vi.mode == "streaming" else None,
        cache_frames=vi.cache_frames if vi.mode == "streaming" else None,
        window_frames=vi.window_frames if vi.mode == "streaming" else None,
        crossfade_frames=(vi.crossfade_frames
                          if vi.mode == "streaming" else None),
        precision=vi.precision,  # resolved: what the render ran in
        chunk_seconds=chunk if a.chunk_seconds is None else a.chunk_seconds,
        overlap_seconds=(overlap
                         if a.overlap_seconds is None else a.overlap_seconds),
        params_m=round(vi.model.num_params / 1e6, 1),
        batch=a.batch,
    )
    kw = dict(
        output_sample_rate=SR,
        chunk_seconds=a.chunk_seconds,
        overlap_seconds=a.overlap_seconds,
    )

    def run(x):
        if a.alpha is None:
            return vi.isolate(x, SR, **kw)
        return vi.enhance(x, SR, alpha=a.alpha, **kw)

    def run_many(xs):
        return vi.isolate_many(xs, SR, **kw)

    weights = [checkpoint_file(a.model)]
    weights += [checkpoint_file(a.stage2)] if a.stage2 else []
    packages = ["inworld.voice_isolation"]
    return (run, run_many if a.alpha is None else None, desc, weights, packages)


def oracle() -> tuple:
    """Returns the Demucs vocals stem and what identifies it.

    A tuple (run, run_many, description, weights, packages).
    """
    # pylint: disable-next=import-outside-toplevel
    import demucsvocals

    demucs_vocals = demucsvocals.DemucsVocals(common.DEV)

    def run(x):
        return demucs_vocals.vocals(x.to(common.DEV)).cpu()

    return (
        run,
        None,
        dict(DEMUCS_META),
        demucs_weights(),
        ["inworld.voice_isolation", "demucs"],
    )


FIELDS = {
    "wrapper_sha256": "the renderer",
    "helpers_sha256": "the renderer's code",
    "source": "the release",
    "package_code_sha256": "the package code",
    "weights_sha256": "the weights",
    "python": "python",
    "packages_sha256": "the installed packages",
    "n_packages": "the installed packages",
    "device": "the device",
    "inference_args": "the arguments",
    "inputs": "the inputs (set build or probe)",
}


def differences(old: dict, new: dict) -> list[str]:
    """Names the parts of a render's identity that differ between two."""
    fields = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    return sorted({FIELDS.get(k, k) for k in fields})


@contextlib.contextmanager
def taken(out_dir: str, wait: float = 300.0):
    """Holds the folder's lock while a process decides what it resumes.

    The shards of one render arrive together: the lock serialises their look
    at the receipts, so the first discards another identity's outputs and the
    rest resume its. A lock older than ``wait`` seconds was abandoned and is
    taken over; a live one is waited for that long.
    """
    path = os.path.join(out_dir, ".prepare-lock")
    deadline = time.time() + wait
    fd = None
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                abandoned = time.time() - os.path.getmtime(path) > wait
            except FileNotFoundError:
                continue
            if abandoned:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)
            elif time.time() > deadline:
                raise SystemExit(f"{path} has been held for over {wait:.0f} s;"
                                 " remove it if its owner is gone") from None
            else:
                time.sleep(0.1)
    try:
        os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


def prepare(out_dir: str, ident: dict, argv: list, overwrite: bool) -> str:
    """Resumes or empties ``out_dir`` and writes this process's receipt.

    Outputs are resumed only under the identity that rendered them. Another
    identity's outputs (the code, the weights, the installed packages, the
    arguments or the set differ) are discarded with their receipts, what
    differed printed, and rendered again; ``overwrite`` discards everything
    whatever its identity.

    Returns:
        The identity's digest.

    Raises:
        SystemExit: The folder holds outputs without a receipt: nothing says
            what they are, so nothing is removed.
    """
    digest = receipt.digest(ident)
    os.makedirs(out_dir, exist_ok=True)
    with taken(out_dir):
        held = glob.glob(os.path.join(out_dir, "*.wav"))
        receipts = receipt.read_all(out_dir)
        foreign = [r for r in receipts if r.get("digest") != digest]
        live = [r for r in foreign if process_alive(r)]
        if live and not overwrite:
            raise SystemExit(
                f"{out_dir}: another identity is still rendering here (pid "
                f"{live[0].get('pid')} on {live[0].get('host')}); wait for "
                "it, stop it, or use a new --name")
        if overwrite:
            reason = "--overwrite"
        elif foreign:
            differed = sorted({
                part for r in foreign
                for part in differences(r.get("identity", {}), ident)
            })
            reason = "another identity: " + (", ".join(differed) or
                                             "the receipt's fields") + " differ"
        elif held and not receipts and all(os.path.islink(p) for p in held):
            # linked in by --input, --reference or --outputs: nothing is lost
            reason = "linked outputs"
        elif held and not receipts:
            raise SystemExit(
                f"{out_dir} holds {len(held)} outputs without a receipt "
                "(unknown origin); remove them or use --overwrite")
        else:
            reason = None
        if reason is not None:
            stale = (glob.glob(os.path.join(out_dir, ".receipt-*")) +
                     glob.glob(os.path.join(out_dir, ".partial-*")))
            for path in held + stale:
                os.remove(path)
            print(f"discarded {len(held)} outputs ({reason}); rendering again",
                  flush=True)
        receipt.write(out_dir, ident, argv)
    return digest


def process_alive(doc: dict) -> bool:
    """Whether the process that wrote a receipt still runs on this host.

    Another host's process cannot be seen; its receipt reads as finished,
    and so does this process's own earlier receipt. A process the signal
    may not reach (another user's) is alive; a pid that now names another
    process (the receipt's start time differs) is not.
    """
    if doc.get("host") != socket.gethostname() or not doc.get("pid"):
        return False
    try:
        pid = int(doc["pid"])
        if pid == os.getpid():
            return False
        os.kill(pid, 0)
    except PermissionError:
        pass
    except (OSError, ValueError):
        return False
    start = doc.get("process_start")
    now = receipt.process_start(pid)
    if start is not None and now is not None and start != now:
        return False
    return True


def read(path: str) -> torch.Tensor:
    """Returns one input as a mono float32 tensor.

    Raises:
        ValueError: The input is not at the set's rate.
    """
    samples, rate = writer.read(path)
    if rate != SR:
        raise ValueError(f"{rate} Hz input; the set is {SR} Hz")
    return torch.from_numpy(samples)


def render(todo: list, out_dir: str, run, run_many, batch: int) -> int:
    """Renders ``todo`` into ``out_dir``; returns how many were written.

    Every output is published under the folder's lock, after this process's
    receipt is seen to be still there: a takeover (another identity, or
    ``--overwrite``) holds the same lock, so it cannot slip in between the
    check and the write and stamp this render's audio as its own.
    """
    t0, ok, step = time.time(), 0, max(1, batch)
    for start in range(0, len(todo), step):
        xs = {}
        for path in todo[start:start + step]:
            try:
                xs[path] = read(path)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"FAIL {os.path.basename(path)} {e}", flush=True)
        ys = {}
        if run_many is not None and len(xs) > 1:
            try:
                with torch.inference_mode():
                    ys = dict(zip(xs, run_many(list(xs.values()))))
            except Exception as e:  # pylint: disable=broad-exception-caught
                # one clip at a time keeps per-file failures apart
                print(f"  batch of {len(xs)} failed ({e}); one by one",
                      flush=True)
        for path, x in xs.items():
            try:
                if path not in ys:
                    with torch.inference_mode():
                        ys[path] = run(x)
                y = writer.fit(ys[path].float().cpu().numpy(), len(x), SR)
                with taken(out_dir):
                    if not os.path.exists(receipt.own_path(out_dir)):
                        # another identity took the folder: its stamp must
                        # not vouch for this render's outputs
                        raise SystemExit(
                            f"{out_dir}: this render's receipt is gone "
                            "(another identity or --overwrite took the "
                            "folder); stopping")
                    writer.save(os.path.join(out_dir, os.path.basename(path)),
                                y, SR)
                ok += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"FAIL {os.path.basename(path)} {e}", flush=True)
        done = min(start + step, len(todo))
        if done // 25 != start // 25:
            print(
                f"  {done}/{len(todo)}  {(time.time() - t0) / done:.2f}s/clip",
                flush=True,
            )
    print(f"DONE {ok}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    return ok


def compute(
    a: argparse.Namespace,
    in_dir: str,
    selected: list,
    files: list,
    out_dir: str,
) -> int:
    """Renders the model or the Demucs stem; stamps a complete selection.

    Returns:
        0 when every file of this process's share was written.
    """
    run, run_many, desc, weights, packages = oracle() if a.demucs else ours(a)
    code = [writer.__file__]
    if a.demucs:
        # pylint: disable-next=import-outside-toplevel
        import demucsvocals
        code.append(demucsvocals.__file__)
    ident = receipt.identity(
        a.name,
        os.path.abspath(__file__),
        sys.argv,
        in_dir,
        common.DEV,
        weights=tuple(weights),
        desc={"package_dirs": packages},
        code=tuple(code),
    )
    digest = prepare(out_dir, ident, sys.argv, a.overwrite)
    todo = [
        p for p in files
        if not os.path.exists(os.path.join(out_dir, os.path.basename(p)))
    ]
    print(
        f"{len(files)} inputs from {in_dir}; {len(todo)} to do -> {out_dir}",
        flush=True,
    )
    if render(todo, out_dir, run, run_many, a.batch) < len(todo):
        return 1
    if a.inputs:
        return 0
    present = sum(
        os.path.exists(os.path.join(out_dir, os.path.basename(p)))
        for p in selected)
    if present < len(selected):
        print(
            f"{present}/{len(selected)} outputs present: the run that "
            "completes the selection stamps it",
            flush=True,
        )
        return 0
    meta = stamprender.stamp(
        a.name,
        os.path.abspath(__file__),
        selected,
        out_dir,
        common.RESULTS,
        common.set_id(),
        digest,
        desc,
        common.SUBSET,
    )
    print(
        f"stamped {a.name}: {meta['n_files']} files, identity {digest[:12]}",
        flush=True,
    )
    return 0


def folder_digest(path: str) -> str:
    """Returns the content identity of a folder of outputs.

    Every WAV's name and bytes, in name order.
    """
    hasher = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        if name.endswith(".wav"):
            hasher.update(name.encode())
            hasher.update(
                bytes.fromhex(receipt.sha256_file(os.path.join(path, name))))
    return hasher.hexdigest()


def link(a: argparse.Namespace, selected: list, files: list, refs: dict,
         out_dir: str) -> int:
    """Links INPUT, REF or a folder rendered elsewhere in as the system.

    Raises:
        SystemExit: The folder lacks an item, or links already there point
            elsewhere than this run would point them.
    """

    def source(path: str) -> str:
        base = os.path.basename(path)
        if a.reference:
            return refs[base]
        return os.path.join(a.outputs, base) if a.outputs else path

    missing = [p for p in selected if not os.path.isfile(source(p))]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(selected)} items are missing (first: "
            f"{os.path.basename(missing[0])}); every item must be there")
    os.makedirs(out_dir, exist_ok=True)
    # under the folder's lock, like a render's publication: a renderer
    # re-checking its receipt cannot interleave with the takeover
    with taken(out_dir):
        held = glob.glob(os.path.join(out_dir, "*.wav"))
        if a.overwrite:
            # the receipts go too: a later computed run must not resume the
            # links as its own render
            for path in held + glob.glob(os.path.join(
                    out_dir, ".receipt-*")) + glob.glob(
                        os.path.join(out_dir, ".partial-*")):
                os.remove(path)
        else:
            targets = {os.path.basename(p): source(p) for p in selected}
            foreign = [
                p for p in held
                if os.path.basename(p) not in targets or os.path.realpath(p) !=
                os.path.realpath(targets[os.path.basename(p)])
            ]
            if foreign:
                raise SystemExit(
                    f"{out_dir} holds {len(foreign)} outputs that are not "
                    "links to what this run links (e.g. "
                    f"{os.path.basename(foreign[0])}: another folder or "
                    "system); use --overwrite or a new --name")
        for path in files:
            dst = os.path.join(out_dir, os.path.basename(path))
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(source(path)), dst)
    if a.reference:
        meta = {
            "mechanism": "clean reference",
            "scope": "paired",
            "reference": True,
        }
    elif a.input:
        meta = {"mechanism": "unprocessed input"}
    else:
        meta = {
            "mechanism":
                "rendered elsewhere",
            "source":
                os.path.basename(os.path.abspath(a.outputs.rstrip("/"))),
            "source_sha256":
                folder_digest(a.outputs),
            "outputs_sha256":
                stamprender.outputs_digest(selected, a.outputs),
            "notes": ("outputs linked in from a folder rendered outside this "
                      "harness; the tier and the display name are declared in "
                      "leaderboard.py --tiers"),
        }
    if not a.outputs:
        meta.update(native_sr=SR, fullband=True)
    present = sum(
        os.path.lexists(os.path.join(out_dir, os.path.basename(p)))
        for p in selected)
    print(f"linked {len(files)} -> {out_dir}", flush=True)
    if present == len(selected):
        meta.update(
            name=a.name,
            display=a.name,
            wrapper="bench/scripts/runsystem.py",
            set_id=common.set_id(),
            subset=common.SUBSET,
            n_files=len(selected),
        )
        os.makedirs(common.RESULTS, exist_ok=True)
        dst = os.path.join(common.RESULTS, f"{a.name}.meta.json")
        tmp = f"{dst}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        os.replace(tmp, dst)
    return 0


def main() -> int:
    a = parse_args()
    rows = common.manifest()  # BENCH_SUBSET=<panel>: that panel's items only
    common.guard_private_results(rows)
    in_dir, selected, refs = selection(a, rows)
    files = share(selected, a.shard, a.limit)
    out_dir = os.path.join(common.OUT_ROOT, a.name)
    if a.model or a.demucs:
        return compute(a, in_dir, selected, files, out_dir)
    return link(a, selected, files, refs, out_dir)


if __name__ == "__main__":
    sys.exit(main())
