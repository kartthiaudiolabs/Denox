"""ClearerVoice-Studio MossFormer2_SE_48K: 48 kHz speech enhancement.

Release: github.com/modelscope/ClearerVoice-Studio (the ``clearvoice``
package, Apache-2.0) with the checkpoint alibabasglab/MossFormer2_SE_48K
(Hugging Face, the revision setupfield.sh pins), read from
<BENCH_SRC>/clearvoice/checkpoints. MossFormer2 (gated single-head transformer
with convolutional recurrence) estimates a mask on a 48 kHz STFT; inputs over
20 s are decoded in 4 s windows, as the release's configuration sets.

Run through the package's own API, with three things set around it: the
checkpoint must be present before the model is built (the package would
otherwise download it, or run with random weights when that fails, warning
only); the hub is offline; and the filterbank dither the package draws from
torch's global generator is seeded per file (crc32 of its name), so a render
is reproducible without changing the computation's distribution.

Usage: <BENCH_VENVS>/clearvoice_venv/bin/python runclearvoice.py --in-dir IN
       --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
"""
import argparse
import os
import sys
import zlib

import numpy as np
import torch

import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

SRC = os.path.join(receipt.SRC, "clearvoice")
MODEL = "MossFormer2_SE_48K"
SR = 48000


def load(device: str) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the release."""
    if device != "cuda:0":
        raise ValueError("the package computes on the first visible GPU; "
                         "choose it with CUDA_VISIBLE_DEVICES")
    checkpoints = os.path.join(SRC, "checkpoints", MODEL)
    for name in ("last_best_checkpoint", "last_best_checkpoint.pt"):
        if not os.path.isfile(os.path.join(checkpoints, name)):
            raise FileNotFoundError(f"{checkpoints}/{name} is missing: run "
                                    "setupfield.sh clearvoice")
    os.environ["HF_HUB_OFFLINE"] = "1"  # before the package imports the hub
    os.chdir(SRC)  # the package reads ./checkpoints/<model>
    # pylint: disable=import-outside-toplevel
    from clearvoice import ClearVoice, networks
    # the package picks the physical GPU with the most free memory and fails
    # under CUDA_VISIBLE_DEVICES; the harness assigns GPUs through that
    networks.SpeechModel.get_free_gpu = lambda self: 0
    model = ClearVoice(task="speech_enhancement", model_names=[MODEL])

    def enhance(path, samples, rate):  # pylint: disable=unused-argument
        torch.manual_seed(zlib.crc32(os.path.basename(path).encode()))
        out = model(input_path=os.path.abspath(path), online_write=False)
        return np.asarray(out, dtype=np.float32).reshape(-1), SR

    return enhance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    receipt.begin("clearvoice_mf2_48k", __file__, out_dir, sys.argv, in_dir,
                  args.device)
    files = writer.inputs(in_dir, args.shard, args.limit)
    enhance = load(args.device)
    return writer.render(files, out_dir, enhance, expect_rate=SR)


if __name__ == "__main__":
    sys.exit(main())
