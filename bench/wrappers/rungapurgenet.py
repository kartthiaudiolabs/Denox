"""GAP-URGENet: first in the objective evaluation of URGENT 2026, Track 1.

Paper: Rong, Wang, Wang, Lu, ICASSP 2026. Two parallel 16 kHz branches, a
generative one (DeWavLM-Omni features to a neural vocoder, the UniPASE
paradigm) and a predictive spectral one, fused by a post-net that generates
the band up to 24 kHz before resampling to the input rate; content above
8 kHz in the input does not reach the output. Code MIT
(github.com/Xiaobin-Rong/gap-urgenet), checkpoints Apache-2.0
(Xiaobin-Rong/gap-urgenet on Hugging Face, the revision setupfield.sh pins).

``pase.render`` runs the release's own inference modules as its README
prescribes (whole files up to 20 s, segmented beyond), checkpoints passed by
path, writes made float32. The release scales every output to the input's
peak (renormalizes=true), and its packet-loss concealment, on by default,
fills runs of exact digital zeros.

Usage: <BENCH_VENVS>/gap_urgenet_venv/bin/python rungapurgenet.py --in-dir IN
       --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
"""
import argparse
import os
import sys

import pase  # the PASE family's release runner, beside this file
import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

REPO = os.path.join(receipt.SRC, "gap-urgenet")
CHECKPOINTS = {
    "--dewavlm_ckpt": "DeWavLM-Omni.pt",
    "--adapter_ckpt": "Adapter.pt",
    "--vocoder_ckpt": "Vocoder.pt",
    "--predictor_ckpt": "Predictor.pt",
    "--postnet_ckpt": "PostNet.pt",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    receipt.begin("gap_urgenet", __file__, args.out_dir, sys.argv, args.in_dir,
                  args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    checkpoints = {
        k: os.path.join(REPO, "ckpts", v) for k, v in CHECKPOINTS.items()
    }
    return pase.render(REPO, files, args.out_dir, args.device, checkpoints)


if __name__ == "__main__":
    sys.exit(main())
