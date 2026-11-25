"""URGENT 2026 Track-1 baseline, discriminative: BSRNN complex masking.

Release: github.com/urgent-challenge/urgent2026_challenge_track1 (Apache-2.0)
at the commit setupfield.sh pins, checkpoint ``bsrnn.ckpt`` from the
organisers' Hugging Face repository lichenda/icassp_2026_urgent_baseline at
the pinned revision, in <BENCH_SRC>/urgent26. Sampling-frequency independent:
an adaptive STFT (espnet STFTEncoder, default_fs 48000) runs at the input's
own rate, so 48 kHz input is processed over the full band.

Follows the release's baseline_code/inference.py: SEModel.load_from_checkpoint,
then model.se_model(wav, length, sr), then the script's peak normalisation to
0.9, kept on purpose: trained with a scale-invariant loss, the model returns
audio at a scale of its own (about 17 dB above the input on this set, peaks
past full scale), so that step is how the shipped baseline sets its level
(renormalizes=true). The FlowSE baseline, whose model keeps the input's scale,
leaves the step out.

Usage: <BENCH_VENVS>/urgent26_venv/bin/python runurgent26bsrnn.py --in-dir IN
       --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
       [--native-rate]
"""
import argparse
import os
import sys

import soxr
import torch

import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

REPO = os.path.join(receipt.SRC, "urgent26")
CHECKPOINT = os.path.join(REPO, "ckpts", "bsrnn.ckpt")
SR = 48000


def load(device: str, native_rate: bool) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the release."""
    # torch >= 2.6 loads checkpoints with weights_only=True and refuses the
    # config class the organisers' checkpoints pickle; the switch, read when
    # torch.load is called, lets the release load them as it was built to
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, REPO)
    # pylint: disable-next=import-outside-toplevel
    from baseline_code.d_model import SEModel
    dev = torch.device(device)
    model = SEModel.load_from_checkpoint(CHECKPOINT, map_location=str(dev))
    model.eval()
    model.to(dev)

    def enhance(path, samples, rate):  # pylint: disable=unused-argument
        wav = torch.from_numpy(samples).to(dev).view(1, -1)
        length = torch.tensor([wav.shape[-1]], device=dev)
        with torch.no_grad():
            out, _ = model.se_model(wav, length, rate)
            out = out / out.abs().max().clamp(min=1e-8) * 0.9
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
    receipt.begin("urgent26_bsrnn", __file__, args.out_dir, sys.argv,
                  args.in_dir, args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    enhance = load(args.device, args.native_rate)
    return writer.render(files, args.out_dir, enhance)


if __name__ == "__main__":
    sys.exit(main())
