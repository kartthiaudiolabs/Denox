"""NVIDIA NemotronLabs RE-USE-laughter: RE-USE trained to keep laughter.

The 2026-10 release of RE-USE (USEMamba, 9.6M) that better preserves
laughter: new weights, the same code.
Paper: "Rethinking Training Targets, Architectures and Data Quality for
Universal Speech Enhancement" (arXiv 2603.02641, 2026). Licence: OpenMDW-1.1
(code and weights). Release: the Hugging Face repository
nvidia/NVIDIA-NemotronLabs-RE-USE-laughter, fetched as is into
<BENCH_SRC>/reuse_nemotron_laughter at the revision setupfield.sh pins;
code, recipe and weights load from there.

Its STFT is defined at an 8 kHz base (window 320, hop 40) and every window
scales with the input rate, so at 48 kHz it runs a 1920/240 STFT over the full
band. ``semamba.enhancer`` follows the release's inference.py. Whole files,
as the release prescribes; its chunked script is for inputs that do not fit
in GPU memory (a 45 s item needs about 25 GiB).

Usage: <BENCH_VENVS>/reuse_venv/bin/python runreusenemotronlaughter.py
       --in-dir IN --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
"""
import argparse
import importlib
import os
import sys

import receipt  # the render's provenance, beside this file
import semamba  # the SEMamba family's release runner, beside this file
import writer  # the shared render loop, beside this file

SRC = os.path.join(receipt.SRC, "reuse_nemotron_laughter")
RECIPE = ("USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_"
          "pha_0005_com_04_early_001.yaml")
SR = 48000


def load(device: str) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` running the release."""
    util, stfts = semamba.bind(SRC)
    generator = importlib.import_module("models.generator_SEMamba_time_d4")
    cfg = util.load_config(os.path.join(SRC, "recipes", RECIPE))
    model = generator.SEMamba.from_pretrained(SRC, cfg=cfg).to(device).eval()
    return semamba.enhancer(model, util, stfts, cfg, device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    receipt.begin("reuse_nemotron_laughter", __file__, args.out_dir, sys.argv,
                  args.in_dir, args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    enhance = load(args.device)
    return writer.render(files, args.out_dir, enhance, expect_rate=SR)


if __name__ == "__main__":
    sys.exit(main())
