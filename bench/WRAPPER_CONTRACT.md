# Wrapper contract (Inworld VIBE)

Every external system is integrated the same way so the leaderboard is fair and re-runnable.
Follow this exactly when adding a system.

## 1. Own venv, on-demand download

Never install a competitor into the benchmark's own environment, and never commit third-party code or weights to this repository.
Add an install recipe to `scripts/setupfield.sh` (one venv under `$BENCH_VENVS/<name>_venv`, clones under `$BENCH_SRC/<name>`), so anyone can reproduce the field from upstream sources under their original licenses.

## 2. Wrapper script

Path: `wrappers/run<name>.py`, invoked as

```
<venv>/bin/python run<name>.py --in-dir IN --out-dir OUT [--limit N] [--shard i/k] [--device cuda:0]
```

Build it on `wrappers/writer.py`, the loop every wrapper shares: `writer.inputs(in_dir, shard, limit)` lists the files one process renders, and `writer.render(files, out_dir, enhance)` calls your `enhance(path, samples, rate) -> (samples, rate)` on every file not rendered yet.
Most of the behavior below is that loop's:

- Read every `*.wav` in `IN` (48 kHz mono float32, 3-20 s; the 45 s items; and, for the
  streaming probe, 15 s cuts of them under `PROBE_IN`). Process each file on its own: the streaming
  probe treats any cross-file dependence as a defect.
- Write `OUT/<same basename>.wav`, **48 kHz mono float32** (`soundfile` `subtype="FLOAT"`;
  the default PCM_16 would clip anything above full scale), of exactly the input's length. The
  loop writes under a temporary name and renames, refuses NaN and inf, and fits the length,
  refusing a difference beyond a release's frame rounding (`writer.TOLERANCE_SECONDS`): a
  cut-off output is a failure, not something to pad. Do not clip, normalize or level-match: the
  scorer calibrates gain itself. Resample in and back out yourself if the model is not 48 kHz
  native.
- **No loudness normalization of your own.** If the model itself changes the level (a
  generative model that re-synthesizes at its own scale, a post-net that matches the input's
  peak), leave that and set `"renormalizes": true` in the meta: the level is then part of what is
  scored. Fidelity's level term reads it, and the scorer undoes it everywhere else (the kept
  energy, suppression, the high-band error, the deletion rule), so a gain is charged once.
  When an upstream *inference script* adds a normalization of its own (the URGENT
  baselines peak-normalize every file to 0.9 because the challenge's metrics are
  scale-invariant), render a few items without it and read the level against the input:
  - if the model's own output keeps the input's scale (the FlowSE baseline: within a decibel),
    the script's step is an evaluation convention, not the system; leave it out and say so in
    `notes`, because kept it would read as a 9 dB level failure;
  - if the model's own output is at a scale of its own (the BSRNN baseline: about 17 dB above
    the input, peaks past full scale, a scale-invariant training loss), the script's step is how
    the shipped system sets its level; keep it, set `"renormalizes": true` and say so, because
    without it the scorer's level calibration saturates (`MAX_LEVEL_CAL_DB`, 12 dB) and the
    leaderboard refuses to rank the row.
- Resumable: a file whose output already exists is skipped.
- Robust: a per-file exception prints `FAIL <basename> <err>` and the loop continues; the exit
  status is non-zero unless every file is present, and a run that exits non-zero is never
  stamped.
- Progress: one line per 25 files and a final `DONE <n_ok>/<n_total>`.

## 3. Description and identity

The wrapper writes no result metadata, but it does write a **receipt**: right after parsing its arguments it calls `receipt.begin("<name>", __file__, args.out_dir, sys.argv, args.in_dir, args.device)` (`bench/wrappers/receipt.py`, standard library only, importable from every venv).
Each process, each shard included, leaves `<out_dir>/.receipt-<host>-<pid>.json` naming what the render depends on:

- the wrapper's sha256 and that of the sibling modules it imports;
- the upstream checkout (`source_dir` in the description, relative to `BENCH_SRC`): its git commit and a digest of its files;
- the sha256 of the weight files it loads (`weight_globs`, relative to that directory);
- the code of the installed packages it runs (`package_dirs`);
- the interpreter and every installed package with its version;
- the device it computes on (its kind, such as the GPU model);
- the inference arguments: the command line without the work selection (folders, shard, limit, device and threads), other paths reduced to their file names;
- the inputs: the set build, or the probe.

`bench/scripts/runfield.sh` stamps a run into `$BENCH_RESULTS/<name>.meta.json` only through `bench/scripts/stamprender.py`, which refuses a wrapper that exited non-zero, a directory without receipts, receipts that disagree with each other or with the environment stamping them, and any inventory that is not exactly the set's files as finite 48 kHz mono float32 audio of the input's length (stray files are ignored, never counted).
The stamp carries the agreed identity and its digest (`render_receipt_sha256`), which the scorer copies into every record, and a digest of the outputs' bytes (`outputs_sha256`), which the scorer recomputes from the bytes it reads before it writes a score (an output changed since the stamp ends the run) and every score file carries: `leaderboard.py` refuses scores of other audio than the render stamped now, and the export refuses audio whose description names another render.
Outputs held under receipts of another identity stop a run (`DISCARD=1` removes them first), an empty input set is an error rather than a successful zero-file run, and `STAMP_ONLY=1` (outputs rendered by sharded wrapper processes) verifies exactly the same way.
Upstream clones and Hugging Face fetches in `setupfield.sh` are pinned to commits and revisions, so a receipt names a reproducible checkout.
A wrapper is part of its render's identity byte for byte: edit it, even its docstring, and its render must be made again before it can be resumed, stamped or probed.

`bench/wrappers/meta/<name>.json` (tracked, hand-maintained, no machine paths):

```json
{
  "display": "MossFormer2_SE_48K",
  "mechanism": "masking | spectral-mapping | GAN | diffusion | flow-matching | generative-codec-LM | time-domain-UNet | DSP-spectral-gating | hybrid-2-stage | SSL-feature+vocoder | separator",
  "native_sr": 48000,
  "fullband": true,
  "params_m": 42.1,
  "license_code": "Apache-2.0",
  "license_weights": "Apache-2.0",
  "commercial_ok": true,
  "repo": "https://github.com/...",
  "weights": "hf:org/model",
  "renormalizes": false,
  "install_notes": "pip install ...; weights auto-download on first use",
  "notes": "anything the reader needs to interpret the numbers",
  "source_dir": "<the clone's folder under BENCH_SRC>",
  "weight_globs": ["checkpoints/*.pt"],
  "package_dirs": ["<the installed package the wrapper imports>"]
}
```

`source_dir`, `weight_globs` and `package_dirs` tell the receipt what to hash; the rest is copied into the stamp for the reader.

`fullband` = can this model emit real content above 8 kHz (48 k native or
>= 44.1 k)? A 16 kHz-native model is `false` and is ranked in the separate
16 kHz tier.
Be honest about `license_weights`: several research repos are non-commercial.

## 4. Smoke test before reporting success

```
<venv>/bin/python wrappers/run<name>.py --in-dir <a folder of 48 kHz mono wavs> \
    --out-dir staging/smoke_<name> --limit 3
python -c "
import soundfile as sf, glob, numpy as np
for p in sorted(glob.glob('staging/smoke_<name>/*.wav')):
    x, sr = sf.read(p); print(p.split('/')[-1], sr, x.shape, 'rms_db', round(20*np.log10(np.sqrt((x**2).mean())+1e-12),1))
"
```

Must be `sr == 48000`, non-silent, and the same length as the input.
A wrapper that emits silence or 16 kHz files is a failed install.

## 5. Register it

Add the system to `TIERS` in `scripts/roster.py` (or pass `--tiers`) and a line to `scripts/runfield.sh`.
Scoring runs on the eval set, which is not distributed: Inworld renders and scores the system from the pull request.
