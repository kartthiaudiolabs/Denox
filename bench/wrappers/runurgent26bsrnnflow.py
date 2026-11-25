"""URGENT 2026 Track-1 baseline, generative: flow matching on a BSRNN backbone.

Release: github.com/urgent-challenge/urgent2026_challenge_track1 (Apache-2.0)
at the commit setupfield.sh pins, checkpoint ``flow_bsrnn.ckpt`` from the
organisers' Hugging Face repository lichenda/icassp_2026_urgent_baseline at
the pinned revision, in <BENCH_SRC>/urgent26. The FlowSE recipe: conditional
flow matching, an Euler solver with 15 steps, EMA weights. Sampling-frequency
independent, so 48 kHz input is processed over the full band.

Follows the release's baseline_code/inference.py with three differences:
FlowSEModel.enhance() hard-codes ``.cuda()``, so the identical sampler call is
made with the device given; the prior noise is seeded per file (crc32 of its
name), so a render is reproducible, where the script leaves it unseeded (a
cut of an item and the item draw different noise); and the script's final peak
normalisation to 0.9, a convention of the challenge's scale-invariant metrics,
is left out because the model's output keeps the input's scale
(renormalizes=false).

Usage: <BENCH_VENVS>/urgent26_venv/bin/python runurgent26bsrnnflow.py
       --in-dir IN --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
       [--native-rate]
"""
import argparse
import os
import sys
import zlib

import soxr
import torch

import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

REPO = os.path.join(receipt.SRC, "urgent26")
CHECKPOINT = os.path.join(REPO, "ckpts", "flow_bsrnn.ckpt")
SR = 48000
STEPS = 15  # the release's FlowSEModel.enhance() default


def load(device: str, native_rate: bool) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the release."""
    # torch >= 2.6 loads checkpoints with weights_only=True and refuses the
    # config class the organisers' checkpoints pickle; the switch, read when
    # torch.load is called, lets the release load them as it was built to
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, REPO)
    # pylint: disable=import-outside-toplevel
    from baseline_code.flow_model import FlowSEModel
    from baseline_code.sampling import get_white_box_solver
    dev = torch.device(device)
    model = FlowSEModel.load_from_checkpoint(CHECKPOINT, map_location=str(dev))
    # eval() copies the EMA weights over the parameters, as the release does
    model.eval()
    model.to(dev)

    def enhance(path, samples, rate):
        wav = torch.from_numpy(samples).to(dev).view(1, -1)
        length = torch.tensor([wav.shape[-1]], device=dev)
        torch.manual_seed(zlib.crc32(os.path.basename(path).encode()))
        with torch.no_grad():
            features = model.speech_to_feature(wav, rate, length)
            sampler = get_white_box_solver("euler",
                                           model.ode,
                                           model,
                                           features,
                                           T_rev=model.T_rev,
                                           t_eps=model.t_eps,
                                           N=STEPS)
            sample, _ = sampler()
            out = model.feature_to_speech(sample, rate, length)
        out = out.cpu().numpy().reshape(-1)
        out_rate = rate if native_rate else SR
        if out_rate != rate:
            out = soxr.resample(out, rate, out_rate)
        return out, out_rate

    return enhance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--native-rate",
                    action="store_true",
                    help="write each output at its input's rate (the URGENT "
                    "view) instead of 48 kHz")
    args = ap.parse_args()
    receipt.begin("urgent26_bsrnn_flow", __file__, args.out_dir, sys.argv,
                  args.in_dir, args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    enhance = load(args.device, args.native_rate)
    return writer.render(files, args.out_dir, enhance)


if __name__ == "__main__":
    sys.exit(main())
