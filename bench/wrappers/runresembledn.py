"""resemble-enhance, the denoiser stage alone (the release's --denoise_only).

Release: github.com/resemble-ai/resemble-enhance (the ``resemble_enhance``
package, MIT) with the weights of the Hugging Face repository
ResembleAI/resemble-enhance at the revision setupfield.sh pins, cloned into
<BENCH_SRC>/resemble-enhance. The denoiser predicts a mask at 44.1 kHz; the
package resamples the input to that rate, processes inputs over 29 s in 30 s
chunks, and returns 44.1 kHz audio, which is resampled back to the input rate
with the package's own resampler settings. An input whose length leaves a final
chunk under a second is padded with silence to a one-second remainder first
(the package's padding crashes on such a chunk) and trimmed afterwards.

Run through the package's ``denoise`` with the weights folder given explicitly:
called without one, the package clones or pulls the repository's main branch
into its own install on every start.

Usage: <BENCH_VENVS>/resemble_venv/bin/python runresembledn.py --in-dir IN
       --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
"""
import argparse
import os
import pathlib
import sys

import torch
import torchaudio

import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

RUN_DIR = os.path.join(receipt.SRC, "resemble-enhance", "enhancer_stage2")


def load(device: str) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the release."""
    if not os.path.isfile(os.path.join(RUN_DIR, "hparams.yaml")):
        raise FileNotFoundError(
            f"{RUN_DIR} is missing: run setupfield.sh resemble")
    # pylint: disable=import-outside-toplevel
    from resemble_enhance.enhancer.inference import denoise
    run_dir = pathlib.Path(RUN_DIR)

    def enhance(path, samples, rate):  # pylint: disable=unused-argument
        # The package cuts the input into 30 s chunks every 29 s and crashes
        # on a final chunk shorter than its own padding; an input whose length
        # leaves a remainder under a second gets silence appended to make the
        # remainder one second, and the output is trimmed to the input again.
        hop = 29 * rate
        remainder = samples.shape[-1] % hop
        pad = rate - remainder if 0 < remainder < rate else 0
        x = torch.from_numpy(samples)
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        out, out_rate = denoise(x, rate, device, run_dir=run_dir)
        out = out.cpu().float()
        if out_rate != rate:
            out = torchaudio.functional.resample(
                out,
                orig_freq=out_rate,
                new_freq=rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492)
        return out[:samples.shape[-1]].numpy(), rate

    return enhance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    receipt.begin("resemble_dn", __file__, args.out_dir, sys.argv, args.in_dir,
                  args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    enhance = load(args.device)
    return writer.render(files, args.out_dir, enhance)


if __name__ == "__main__":
    sys.exit(main())
