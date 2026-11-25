"""UniPASE: the PASE family's universal speech enhancer at 48 kHz output.

Paper: Rong, Wang, Wang, Gao, Lu, IEEE TASLP 2026 (arXiv 2604.14606), the
generative branch of GAP-URGENet: DeWavLM-Omni phonetic representations, an
adapter, a 16 kHz neural vocoder and a post-net that generates the band up to
24 kHz before resampling to the input rate; content above 8 kHz in the input
does not reach the output. Code MIT (github.com/Xiaobin-Rong/unipase),
checkpoints Apache-2.0 (Xiaobin-Rong/unipase on Hugging Face, the revision
setupfield.sh pins).

``pase.render`` runs the release's own inference modules as its README
prescribes (whole files up to 20 s, segmented beyond), checkpoints passed by
path, writes made float32. The release scales every output to the input's
peak (renormalizes=true), and its packet-loss concealment, on by default,
fills runs of exact digital zeros.

Usage: <BENCH_VENVS>/unipase_venv/bin/python rununipase.py --in-dir IN
       --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
"""
import argparse
import os
import sys

import pase  # the PASE family's release runner, beside this file
import receipt  # the render's provenance, beside this file
import writer  # the shared render loop, beside this file

REPO = os.path.join(receipt.SRC, "unipase")
CHECKPOINTS = {
    "--dewavlm_ckpt": "DeWavLM-Omni.pt",
    "--adapter_ckpt": "Adapter.pt",
    "--vocoder_ckpt": "Vocoder_DWO-L1.pt",
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
    receipt.begin("unipase", __file__, args.out_dir, sys.argv, args.in_dir,
                  args.device)
    files = writer.inputs(args.in_dir, args.shard, args.limit)
    checkpoints = {
        k: os.path.join(REPO, "ckpts", v) for k, v in CHECKPOINTS.items()
    }
    return pase.render(REPO, files, args.out_dir, args.device, checkpoints)


if __name__ == "__main__":
    sys.exit(main())
