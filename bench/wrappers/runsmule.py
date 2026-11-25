"""Smule Renaissance Small: 48 kHz complex-STFT vocal restoration.

Release: github.com/smulelabs/smule-renaissance (MIT) at the commit
setupfield.sh pins, weights smulelabs/Smule-Renaissance-Small (Hugging Face,
MIT, the pinned revision) in <BENCH_SRC>/smule-renaissance. 10.4M parameters.

Runs the way the release's main.py does: a 60 Hz high-pass, the input scaled
to unit peak, the model on a 4096/2048 STFT (half precision under autocast on
a GPU, full precision on the CPU), the inverse STFT, the peak scaling undone.
The release's inverse STFT takes no target length, so its output ends up to
2047 samples (43 ms) short; the remainder is filled with zeros.

Usage: <BENCH_VENVS>/smule_venv/bin/python runsmule.py --in-dir IN
       --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
"""
import argparse
import os
import sys

import torch
import torchaudio

import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

REPO = os.path.join(receipt.SRC, "smule-renaissance")
CHECKPOINT = os.path.join(REPO, "smule-renaissance-small.pt")
SR = 48000
N_FFT, HOP = 4096, 2048
HIGHPASS_HZ = 60.0


def load(device: str) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the release."""
    sys.path.insert(0, REPO)
    # pylint: disable=import-outside-toplevel
    from model import Renaissance  # the release's own modules
    from spectral_ops import STFT, iSTFT
    dev = torch.device(device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model = Renaissance().to(dev).to(dtype)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=dev, weights_only=True))
    model.eval()
    stft = STFT(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT)
    istft = iSTFT(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT)

    def enhance(path, samples, rate):  # pylint: disable=unused-argument
        wav = torch.from_numpy(samples).unsqueeze(0)
        wav = torchaudio.functional.highpass_biquad(wav,
                                                    SR,
                                                    cutoff_freq=HIGHPASS_HZ)
        wav = wav.to(dev).to(dtype)
        peak = torch.max(torch.abs(wav))
        if peak > 0:
            wav = wav / peak
        with torch.no_grad():
            spec = stft(wav)
            with torch.autocast(device_type="cuda", enabled=dev.type == "cuda"):
                spec = model(spec)
            out = istft(spec)
        if peak > 0:
            out = out * peak
        return out.squeeze(0).float().cpu().numpy(), rate

    return enhance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    receipt.begin("smule", __file__, args.out_dir, sys.argv, args.in_dir,
                  args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    enhance = load(args.device)
    return writer.render(files, args.out_dir, enhance, expect_rate=SR)


if __name__ == "__main__":
    sys.exit(main())
