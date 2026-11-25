"""DeepFilterNet3: 48 kHz real-time speech enhancement by deep filtering.

Paper: Schröter et al., "DeepFilterNet: Perceptually Motivated Real-Time Speech
Enhancement" (Interspeech 2023). Code MIT or Apache-2.0
(github.com/Rikorose/DeepFilterNet, the ``deepfilternet`` package), model from
the release's ``models/DeepFilterNet3.zip`` at the tag setupfield.sh pins,
unpacked into <BENCH_SRC>/deepfilternet. Per-bin complex filters across
neighbouring frames, two frames of look-ahead.

Run through the package's ``init_df`` and ``enhance`` with the model folder
given explicitly (the package would otherwise download the zip from the
project's main branch into a user cache) and no log file written into it.

Usage: <BENCH_VENVS>/dfn3_venv/bin/python rundfn3.py --in-dir IN --out-dir OUT
       [--limit N] [--shard i/k] [--device cuda:0]
"""
import argparse
import os
import sys

import torch

import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

MODEL_DIR = os.path.join(receipt.SRC, "deepfilternet", "DeepFilterNet3")
SR = 48000


def load(device: str) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the release."""
    if not os.path.isdir(os.path.join(MODEL_DIR, "checkpoints")):
        raise FileNotFoundError(
            f"{MODEL_DIR} is missing: run setupfield.sh dfn3")
    # pylint: disable=import-outside-toplevel
    from df.enhance import enhance as df_enhance
    from df.enhance import init_df
    from df.utils import get_device
    model, state, _ = init_df(model_base_dir=MODEL_DIR, log_file=None)
    # the package computes on the first visible GPU whenever there is one
    if get_device() != torch.device(device):
        raise ValueError(
            f"the package computes on {get_device()}, not {device}: "
            "choose the GPU with CUDA_VISIBLE_DEVICES")

    def enhance(path, samples, rate):  # pylint: disable=unused-argument
        out = df_enhance(model, state, torch.from_numpy(samples).unsqueeze(0))
        return out.squeeze(0).cpu().numpy(), rate

    return enhance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    receipt.begin("dfn3", __file__, args.out_dir, sys.argv, args.in_dir,
                  args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    enhance = load(args.device)
    return writer.render(files, args.out_dir, enhance, expect_rate=SR)


if __name__ == "__main__":
    sys.exit(main())
