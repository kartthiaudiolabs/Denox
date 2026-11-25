"""What a render depends on, written by the process that made it.

Every renderer calls ``receipt.begin(...)`` once its arguments are parsed,
and each process (each shard) writes ``<out_dir>/.receipt-<host>-<pid>.json``
before it renders anything. The identity it records:

* the renderer's own sha256 and that of every sibling module it imports;
* the upstream release it runs (``source_dir`` in ``meta/<system>.json``,
  under ``BENCH_SRC``): its git commit when there is one, and a digest of
  every source file in it (code and configuration, never weights), so an
  edited checkout or another snapshot is another identity;
* a digest of the code of every installed package the description names
  (``package_dirs``), for releases that run from an installed package;
* the sha256 of every weight file the description names (``weight_globs``);
  a pattern that matches nothing is an error, never an empty record;
* the interpreter and every installed distribution with its version;
* the device it computes on: ``cpu``, or ``cuda`` with the GPU's name,
  because kernels and precisions differ between devices;
* the inference arguments: the command line minus the work selection
  (paths, shards, limits), with every path reduced to its basename;
* the inputs it is handed: the set build (``checksums.json`` beside the input
  folder), the streaming probe (``probe.json``), or a digest of the files.

``digest(identity)`` is the sha256 of the whole canonical identity and
``environment_digest(identity)`` that of everything but the inputs, so a
render of the streaming probe can be matched to the environment that rendered
the board. ``bench/scripts/stamprender.py`` stamps a render only when every
receipt carries the digest of the environment and inputs it is stamping.

Standard library only (torch is consulted for the GPU's name when the venv
has it): this module runs inside every system's own venv.

    python receipt.py probe <system> <renderer.py> --in-dir DIR \\
        [--device cuda:0] [--weights path ...] [-- renderer arguments]
"""
import argparse
import datetime
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(HERE, "meta")
SRC = os.environ.get("BENCH_SRC",
                     os.path.expanduser("~/.cache/voice-isolation-bench/src"))
# options that select work or hardware, not the computation, with a value;
# the device is recorded on its own, resolved to what it is
VOLATILE = ("--in-dir", "--out-dir", "--inputs", "--shard", "--device",
            "--limit", "--threads")
VOLATILE_FLAGS = ("--overwrite",)  # the same, without a value
# files of a release that hold its code and configuration
SOURCE_SUFFIXES = (".py", ".pyx", ".c", ".cc", ".cpp", ".cu", ".h", ".yaml",
                   ".yml", ".json", ".toml", ".cfg", ".ini", ".sh")
SKIPPED_DIRS = frozenset(
    (".git", "__pycache__", ".cache", ".ipynb_checkpoints", "node_modules"))


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    """Returns the sha256 of a file, read in chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            hasher.update(block)
    return hasher.hexdigest()


def description(system: str) -> dict:
    """Returns ``meta/<system>.json``, or an empty description."""
    path = os.path.join(META_DIR, f"{system}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def helpers(renderer_path: str) -> dict:
    """Returns {file name: sha256} of the sibling modules a renderer imports.

    Directly or through each other.
    """
    here = os.path.dirname(os.path.abspath(renderer_path))
    out, pending = {}, [renderer_path]
    while pending:
        with open(pending.pop(), encoding="utf-8") as fh:
            text = fh.read()
        for name in re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", text,
                               re.M):
            sibling = os.path.join(here, name + ".py")
            if (name != "receipt" and os.path.exists(sibling) and
                    name + ".py" not in out):
                out[name + ".py"] = sha256_file(sibling)
                pending.append(sibling)
    return out


def tree_digest(root: str, excluded: frozenset = frozenset()) -> str:
    """Returns a digest of every source file under ``root``.

    Args:
        root: The directory to walk.
        excluded: Absolute paths of files to leave out (the weights).
    """
    entries = []
    for top, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIPPED_DIRS)
        for name in sorted(files):
            path = os.path.join(top, name)
            if name.endswith(SOURCE_SUFFIXES) and path not in excluded:
                rel = os.path.relpath(path, root)
                entries.append(f"{rel} {sha256_file(path)}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def weight_paths(desc: dict) -> dict:
    """Returns {relative path: absolute path} of a description's weights.

    Raises:
        FileNotFoundError: A declared pattern matches no file.
    """
    rel = desc.get("source_dir")
    root = os.path.join(SRC, rel) if rel else SRC
    out = {}
    for pattern in desc.get("weight_globs", []):
        matched = [
            p for p in sorted(
                glob.glob(os.path.join(root, pattern), recursive=True))
            if os.path.isfile(p)
        ]
        if not matched:
            raise FileNotFoundError(
                f"no weights match {pattern!r} under {root}: install them "
                "with setupfield.sh")
        for path in matched:
            out[os.path.relpath(path, root)] = path
    return out


def source(desc: dict) -> dict | None:
    """Returns the upstream release: directory, git commit and source digest.

    None when the description names none.

    Raises:
        FileNotFoundError: The release's directory is missing.
    """
    rel = desc.get("source_dir")
    if not rel:
        return None
    path = os.path.join(SRC, rel)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} is missing: run setupfield.sh")
    commit = None
    if os.path.isdir(os.path.join(path, ".git")):
        try:
            command = ["git", "-C", path, "rev-parse", "HEAD"]
            commit = subprocess.check_output(command, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
    excluded = frozenset(weight_paths(desc).values())
    return {
        "dir": rel,
        "git_commit": commit,
        "files_sha256": tree_digest(path, excluded)
    }


def package_code(desc: dict) -> dict:
    """Returns {package: digest of its source files} for the release's packages.

    The installed packages whose code the release runs (``package_dirs``).

    Raises:
        ModuleNotFoundError: A named package is not installed.
    """
    out = {}
    for name in desc.get("package_dirs", []):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ModuleNotFoundError(f"package {name!r} is not installed")
        out[name] = tree_digest(list(spec.submodule_search_locations)[0])
    return out


def packages() -> list:
    """Returns every installed distribution, ``name==version``, sorted."""
    names = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            names.add(f"{name}=={dist.version}")
    return sorted(names)


def resolve_device(device: str) -> str:
    """Returns ``cpu`` or ``cuda:<GPU name>`` for a requested device.

    Raises:
        RuntimeError: CUDA is requested and not available: a render never
            falls back to the CPU silently.
    """
    if not device or not str(device).startswith("cuda"):
        return "cpu"
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise RuntimeError(f"{device} requested; this venv has no torch") from e
    if not torch.cuda.is_available():
        raise RuntimeError(f"{device} requested; CUDA is not available")
    index = int(str(device).split(":")[1]) if ":" in str(device) else 0
    return f"cuda:{torch.cuda.get_device_name(index)}"


def inference_args(argv: list) -> list:
    """Returns a command line minus the work selection.

    Paths are reduced to their basenames.
    """
    out, skip = [], False
    for token in (argv or [])[1:]:
        if skip:
            skip = False
        elif token in VOLATILE:
            skip = True
        elif token in VOLATILE_FLAGS or any(
                token.startswith(v + "=") for v in VOLATILE):
            continue
        else:
            out.append(os.path.basename(token) if os.sep in token else token)
    return out


def inputs(in_dir: str) -> dict:
    """Returns what an input folder is.

    A set build, the streaming probe, or a digest of its files.
    """
    parent = os.path.dirname(os.path.abspath(in_dir.rstrip(os.sep)))
    for name, key, field in (("checksums.json", "set_id", "set_id"),
                             ("probe.json", "identity", "probe")):
        path = os.path.join(parent, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                value = json.load(fh).get(key)
            if value:
                return {field: value}
    entries = [
        f"{n} {sha256_file(os.path.join(in_dir, n))}"
        for n in sorted(os.listdir(in_dir))
        if n.endswith(".wav")
    ]
    digest_ = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return {"files_sha256": digest_}


def identity(system: str,
             renderer_path: str,
             argv: list,
             in_dir: str,
             device: str,
             weights: tuple = (),
             desc: dict | None = None,
             code: tuple | None = None) -> dict:
    """Returns the identity of this environment rendering ``in_dir``.

    Args:
        system: The system's name.
        renderer_path: The script that renders.
        argv: Its command line.
        in_dir: The folder of inputs it renders.
        device: The device it was asked to compute on.
        weights: Further weight files by path (a checkpoint given on the
            command line), hashed beside the declared ones.
        desc: The description; ``meta/<system>.json`` when None.
        code: The files besides the renderer whose code computes, when the
            renderer's sibling imports are not that (they select work).
    """
    desc = description(system) if desc is None else desc
    declared = {k: sha256_file(p) for k, p in weight_paths(desc).items()}
    declared.update({os.path.basename(p): sha256_file(p) for p in weights})
    pkgs = packages()
    if code is None:
        code_sha256 = helpers(renderer_path)
    else:
        code_sha256 = {os.path.basename(p): sha256_file(p) for p in code}
    return {
        "system": system,
        "wrapper": os.path.basename(renderer_path),
        "wrapper_sha256": sha256_file(renderer_path),
        "helpers_sha256": code_sha256,
        "source": source(desc),
        "package_code_sha256": package_code(desc),
        "weights_sha256": declared,
        "python": sys.version.split()[0],
        "packages_sha256": hashlib.sha256("\n".join(pkgs).encode()).hexdigest(),
        "n_packages": len(pkgs),
        "device": resolve_device(device),
        "inference_args": inference_args(argv),
        "inputs": inputs(in_dir),
    }


def digest(ident: dict) -> str:
    """Returns the sha256 of an identity in canonical form."""
    text = json.dumps(ident, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def environment_digest(ident: dict) -> str:
    """Returns the digest of an identity without its inputs."""
    return digest({k: v for k, v in ident.items() if k != "inputs"})


def process_start(pid: int) -> str | None:
    """Returns the start time of ``pid`` as the kernel records it (Linux:
    ``/proc/<pid>/stat``, in clock ticks since boot), or None where it cannot
    be read; with the pid it names one process, since a pid alone can be
    reused."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as fh:
            stat = fh.read()
    except (OSError, ValueError):
        return None
    fields = stat[stat.rfind(")") + 2:].split()
    return fields[19] if len(fields) > 19 else None


def own_path(out_dir: str) -> str:
    """Returns the path of this process's receipt in ``out_dir``."""
    return os.path.join(out_dir,
                        f".receipt-{socket.gethostname()}-{os.getpid()}.json")


def write(out_dir: str, ident: dict, argv: list | None = None) -> str:
    """Writes this process's receipt into ``out_dir``; returns its digest."""
    os.makedirs(out_dir, exist_ok=True)
    host = socket.gethostname()
    path = own_path(out_dir)
    doc = {
        "identity":
            ident,
        "digest":
            digest(ident),
        "environment_digest":
            environment_digest(ident),
        "argv":
            inference_args(argv),
        "packages":
            packages(),
        "host":
            host,
        "pid":
            os.getpid(),
        "process_start":
            process_start(os.getpid()),
        "time":
            datetime.datetime.now(datetime.timezone.utc
                                 ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return doc["digest"]


def begin(system: str, renderer_file: str, out_dir: str, argv: list,
          in_dir: str, device: str, **kwargs) -> str:
    """Writes this process's receipt before it renders anything.

    Args:
        system, renderer_file, argv, in_dir, device: As for ``identity``.
        out_dir: Where the render's outputs go.
        **kwargs: ``weights``, ``desc`` and ``code``, as for ``identity``.

    Returns:
        The identity's digest.
    """
    ident = identity(system, renderer_file, argv, in_dir, device, **kwargs)
    return write(out_dir, ident, argv)


def read_all(out_dir: str) -> list:
    """Returns every receipt in ``out_dir``."""
    docs = []
    for path in sorted(glob.glob(os.path.join(out_dir, ".receipt-*.json"))):
        with open(path, encoding="utf-8") as fh:
            docs.append(json.load(fh))
    return docs


def main(argv: list) -> int:
    """Prints the identity this environment would record, and its digests.

    Args:
        argv: ``receipt.py probe <system> <renderer> --in-dir DIR [options]``,
            then ``--`` and the renderer's own arguments.
    """
    own, rest = argv[1:], []
    if "--" in own:
        own, rest = own[:own.index("--")], own[own.index("--") + 1:]
    ap = argparse.ArgumentParser(prog="receipt.py")
    ap.add_argument("command", choices=("probe",))
    ap.add_argument("system")
    ap.add_argument("renderer")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", nargs="*", default=())
    a = ap.parse_args(own)
    ident = identity(a.system, a.renderer, [a.renderer] + rest, a.in_dir,
                     a.device, tuple(a.weights))
    print(
        json.dumps(
            {
                "identity": ident,
                "digest": digest(ident),
                "environment_digest": environment_digest(ident)
            },
            sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
