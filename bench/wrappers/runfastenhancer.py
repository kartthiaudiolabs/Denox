"""FastEnhancer-L at 48 kHz: a streaming enhancer, driven frame by frame.

Paper: arXiv 2509.21867. Code MIT (github.com/aask1357/fastenhancer); the
model is the release's own ONNX export, ``fastenhancer_l.onnx`` of the GitHub
release onnx-48khz-v1, fetched and hash-checked by setupfield.sh into
<BENCH_SRC>/fastenhancer_onnx. The export consumes 200-sample frames (its hop)
with a 1024-sample window and carries its recurrent state as explicit inputs
and outputs: the speed-optimised streaming corner of the design space.

Each file runs the way the release's test script does: padded on the right by
one window, fed frame by frame until the padding has passed (which flushes the
state, so the last window of speech is emitted), and read from N_FFT - HOP
samples in, which removes the algorithmic delay; the state starts fresh for
every file. onnxruntime on the CPU; its default thread pool spins more than it
computes on a large machine, so two intra-op threads run several times faster,
and shards of one folder run side by side for throughput.

Usage: <BENCH_VENVS>/fastenhancer_venv/bin/python runfastenhancer.py
       --in-dir IN --out-dir OUT [--limit N] [--shard i/k] [--threads 2]
"""
import argparse
import os
import sys

import numpy as np

import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

ONNX = os.path.join(receipt.SRC, "fastenhancer_onnx", "fastenhancer_l.onnx")
SR = 48000
HOP = 200  # the export's frame, which its input shape carries
N_FFT = 1024  # the 48 kHz models' window; the delay is N_FFT - HOP


def stream(session, x: np.ndarray) -> np.ndarray:
    """Runs one file through the streaming export, state fresh, delay removed.

    Returns:
        Exactly ``len(x)`` samples.
    """
    inputs = session.get_inputs()
    names = [i.name for i in inputs]
    outputs = [o.name for o in session.get_outputs()]
    caches = [np.zeros(i.shape, dtype=np.float32) for i in inputs[1:]]
    padded = np.pad(x.astype(np.float32), (0, N_FFT))
    chunks = []
    for start in range(0, len(x) + N_FFT - HOP, HOP):
        frame = padded[start:start + HOP]
        frame = np.pad(frame, (0, HOP - len(frame)))
        feeds = {names[0]: frame.reshape(1, HOP)}
        feeds.update(zip(names[1:], caches))
        result = session.run(outputs, feeds)
        chunks.append(result[0].reshape(-1))
        caches = [np.asarray(c, dtype=np.float32) for c in result[1:]]
    y = np.concatenate(chunks)[N_FFT - HOP:N_FFT - HOP + len(x)]
    return np.pad(y, (0, len(x) - len(y)))


def load(threads: int) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the export."""
    import onnxruntime  # pylint: disable=import-outside-toplevel
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(ONNX,
                                           options,
                                           providers=["CPUExecutionProvider"])
    frame = session.get_inputs()[0].shape[-1]
    if frame != HOP:
        raise ValueError(
            f"the export consumes {frame}-sample frames, not {HOP}")

    def enhance(path, samples, rate):  # pylint: disable=unused-argument
        return stream(session, samples), rate

    return enhance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=("cpu",))
    args = ap.parse_args()
    receipt.begin("fastenhancer_l_48k", __file__, args.out_dir, sys.argv,
                  args.in_dir, args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    enhance = load(args.threads)
    return writer.render(files, args.out_dir, enhance, expect_rate=SR)


if __name__ == "__main__":
    sys.exit(main())
