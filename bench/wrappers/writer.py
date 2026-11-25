"""The loop every wrapper runs: pick its inputs, render each once, write safely.

``inputs`` lists the files one process renders. ``render`` calls an
``Enhance`` function, ``(path, samples, rate) -> (samples, rate)``, on every
file not yet rendered, brings each result to the input's duration with
``fit``, writes it with ``save``, prints progress and ``DONE n/N``, and
returns the process's exit status: zero only when every file it was given is
present. ``save`` writes a 32-bit float WAV under a temporary name and renames
it into place, so an interrupted run leaves nothing a later run would take as
done; it refuses NaN and inf. ``fit`` absorbs a release's frame rounding, up
to ``TOLERANCE_SECONDS``, and refuses a larger difference: a cut-off output is
a failure, not something to pad.

Standard library, NumPy and soundfile only: every system's venv has them.
"""
import collections.abc
import glob
import os
import tempfile

import numpy as np
import soundfile as sf

TOLERANCE_SECONDS = 0.05  # the most a release's framing may change a length
Enhance = collections.abc.Callable[[str, np.ndarray, int], tuple[np.ndarray,
                                                                 int]]


def inputs(in_dir: str, shard: str = "", limit: int | None = None) -> list:
    """Returns the WAV files of ``in_dir`` one process renders.

    Args:
        in_dir: The folder of inputs.
        shard: ``i/k`` to take every k-th file from the i-th, or empty.
        limit: At most this many files, or None.
    """
    files = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    if shard:
        index, count = (int(v) for v in shard.split("/"))
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"--shard {shard}: need 0 <= i < k")
        files = files[index::count]
    return files[:limit] if limit else files


def fit(samples: np.ndarray, n: int, rate: int) -> np.ndarray:
    """Returns ``samples`` padded or trimmed to exactly ``n``.

    Raises:
        ValueError: The length is off by more than ``TOLERANCE_SECONDS``.
    """
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if abs(len(samples) - n) > TOLERANCE_SECONDS * rate:
        raise ValueError(f"the release returned {len(samples)} samples for "
                         f"{n}: a failed or cut-off output")
    if len(samples) >= n:
        return samples[:n]
    return np.pad(samples, (0, n - len(samples)))


def save(path: str, samples: np.ndarray, rate: int) -> None:
    """Writes ``samples`` to ``path`` as a 32-bit float WAV, atomically.

    Raises:
        ValueError: The samples hold NaN or inf.
    """
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(samples).all():
        raise ValueError("the output holds NaN or inf")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix=".partial-",
                               suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, samples, rate, subtype="FLOAT", format="WAV")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read(path: str) -> tuple[np.ndarray, int]:
    """Returns the file's samples as a mono float32 array, and its rate."""
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    return samples.mean(axis=1), rate


def render(files: list,
           out_dir: str,
           enhance: Enhance,
           expect_rate: int | None = None) -> int:
    """Renders every file of ``files`` that ``out_dir`` does not hold yet.

    Args:
        files: The input files.
        out_dir: Where the outputs go, under the inputs' names.
        enhance: ``(path, samples, rate) -> (samples, rate)``, one file.
        expect_rate: The only input rate accepted, or None for any.

    Returns:
        0 when every file is present at the end, 1 otherwise.
    """
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    for count, path in enumerate(files, 1):
        name = os.path.basename(path)
        target = os.path.join(out_dir, name)
        if os.path.exists(target):
            done += 1
            continue
        try:
            samples, rate = read(path)
            if expect_rate and rate != expect_rate:
                raise ValueError(f"input rate {rate}, expected {expect_rate}")
            out, out_rate = enhance(path, samples, rate)
            n = int(round(len(samples) * out_rate / rate))
            save(target, fit(out, n, out_rate), out_rate)
            done += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
            # one file's failure is reported and the rest continue: the
            # exit status and stamprender's inventory check catch it
            print(f"FAIL {name} {e}", flush=True)
        if count % 25 == 0:
            print(f"  {count}/{len(files)}", flush=True)
    print(f"DONE {done}/{len(files)}", flush=True)
    return 0 if done == len(files) else 1
