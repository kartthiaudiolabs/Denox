"""Runs a release of the PASE family (GAP-URGENet, UniPASE) as its README says.

Both releases ship ``inference.inference`` for files up to 20 s and
``inference.inference_long`` (8 s segments, 4 s hop) beyond, take every
checkpoint by path, and write with soundfile's default WAV subtype, 16-bit
PCM. ``render`` drives the release's own modules, code untouched, with three
things set around them:

* every write is float32: a 16-bit write would quantise the model's output
  before anything of ours reads it;
* every checkpoint is passed by path from the pinned download under
  <BENCH_SRC>, with the Hugging Face hub offline, so nothing is fetched while
  rendering and the receipt's hashes name the weights that ran;
* each group of files renders in a temporary folder, and its outputs are then
  fitted to the input's length and written by ``writer``.

Packet-loss concealment stays at the release's default (on): it treats runs of
exact digital zeros as lost packets and fills them.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import soundfile as sf

import writer  # the shared render loop, beside this file

LONG_SECONDS = 20.0
MODULES = ("inference.inference", "inference.inference_long")

# Runs ``python -m <module> <args>`` with soundfile.write forced to float32.
RUNNER = ("import runpy, sys, soundfile as sf\n"
          "_write = sf.write\n"
          "def _float_write(file, data, samplerate, *args, **kwargs):\n"
          "    if not args and 'subtype' not in kwargs:\n"
          "        kwargs['subtype'] = 'FLOAT'\n"
          "    return _write(file, data, samplerate, *args, **kwargs)\n"
          "sf.write = _float_write\n"
          "module = sys.argv.pop(1)\n"
          "runpy.run_module(module, run_name='__main__', alter_sys=True)\n")


def run_module(repo: str, module: str, in_dir: str, out_dir: str, device: str,
               checkpoints: dict) -> None:
    """Runs one inference module of the release over ``in_dir``.

    Args:
        repo: The release's checkout.
        module: ``inference.inference`` or ``inference.inference_long``.
        in_dir: Inputs; out_dir: where the release writes.
        device: The device to compute on.
        checkpoints: {argument: path}, every checkpoint the release takes.
    """
    env = dict(os.environ,
               HF_HUB_OFFLINE="1",
               PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [
        sys.executable, "-c", RUNNER, module, "-I", in_dir, "-O", out_dir, "-D",
        device
    ]
    for argument, path in checkpoints.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is missing: run setupfield.sh")
        cmd += [argument, path]
    subprocess.run(cmd, cwd=repo, env=env, check=True)


def render(repo: str, files: list, out_dir: str, device: str,
           checkpoints: dict) -> int:
    """Renders every file of ``files`` that ``out_dir`` does not hold yet.

    Returns:
        0 when every file is present at the end, 1 otherwise.
    """
    os.makedirs(out_dir, exist_ok=True)
    todo = [
        p for p in files
        if not os.path.exists(os.path.join(out_dir, os.path.basename(p)))
    ]
    groups = {m: [] for m in MODULES}
    for path in todo:
        long_file = sf.info(path).duration > LONG_SECONDS
        groups[MODULES[1] if long_file else MODULES[0]].append(path)
    done = len(files) - len(todo)
    work = tempfile.mkdtemp(prefix="pase_")
    try:
        for module, paths in groups.items():
            if paths:
                done += render_group(repo, module, paths, out_dir, device,
                                     checkpoints, os.path.join(work, module))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"DONE {done}/{len(files)}", flush=True)
    return 0 if done == len(files) else 1


def render_group(repo: str, module: str, paths: list, out_dir: str, device: str,
                 checkpoints: dict, work: str) -> int:
    """Renders one group through one module; returns how many succeeded."""
    in_dir, raw_dir = os.path.join(work, "in"), os.path.join(work, "out")
    os.makedirs(in_dir)
    for path in paths:
        os.symlink(os.path.abspath(path),
                   os.path.join(in_dir, os.path.basename(path)))
    try:
        run_module(repo, module, in_dir, raw_dir, device, checkpoints)
    except subprocess.CalledProcessError as e:
        print(f"FAIL {module}: {e}", flush=True)
    done = 0
    for path in paths:
        name = os.path.basename(path)
        try:
            samples, rate = writer.read(path)
            out, out_rate = writer.read(os.path.join(raw_dir, name))
            if out_rate != rate:
                raise ValueError(f"output rate {out_rate}, input rate {rate}")
            writer.save(os.path.join(out_dir, name),
                        writer.fit(out, len(samples), rate), rate)
            done += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"FAIL {name} {e}", flush=True)
    return done
