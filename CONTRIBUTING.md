# Contributing

This repository holds the inference package and the VIBE evaluation harness.
Training code and the dataset builder are not included.
Start with [README.md](README.md) for usage, [bench/README.md](bench/README.md) for evaluation and [AGENTS.md](AGENTS.md) for the coding conventions.

## Development setup

From a checkout, with Python 3.10 or later:

```bash
make install            # requires uv; creates .venv on Python 3.11 (or reuses one) with dev, eval and docs
make test               # CPU tests, including a summary of skipped tests
make lint
make format-check
```

`make` lists the available commands. They use `.venv/bin/python` by default;
set `PYTHON` to use another interpreter. The equivalent manual setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,eval,docs]"
python -m pytest -q -rs
```

The tests use small CPU models and need no GPU, weights or benchmark audio.
A test whose optional dependency is missing skips and names the extra; `eval` covers the benchmark's tests.
A smaller install still gives a green run, with more skips.

### What a fresh checkout shows

After `make install`, `make test` passes with nothing failing and these skips; `pytest -rs` names each one:

- The golden parity tests, without the released checkpoints under `models/` (or `VOICE_ISOLATION_GOLDEN_MODEL`).
  The checkpoints' expected checksums are in `src/inworld/voice_isolation/registry.py`.
- The demo tests, without the `demo` extra.
- The Mamba kernel tests, without the `fast` extra, which needs CUDA.
- The autocast parity test, without a GPU.
- On macOS, the tests that need a case-sensitive file system or Linux's `/proc`.

Include the skip count in test reports.

CI runs the same development checks on pushes and pull requests, on Ubuntu: the tests on Python 3.10 and 3.12, and the style checks on Python 3.10 with the pinned tools.
Lint on Python 3.11 or earlier: from 3.12, pylint reads quotes inside f-strings differently and reports `inconsistent-quotes` warnings that CI does not.

Use the `eval`, `asr` or `demo` extras when working on those features.
The `demucs` extra adds the benchmark's music-separation reference row.
Third-party benchmark systems use their own environments; see [the wrapper contract](bench/WRAPPER_CONTRACT.md).

## Reporting a problem

Include the command or a minimal Python example, the expected behavior, the actual behavior and the traceback.
For inference issues, paste the output of `voice-isolate-doctor --json`, and also include:

- Package or git revision, model registry name or checkpoint checksum, and inference options.
- OS, Python, PyTorch and torchaudio versions, and the CPU or GPU model.
- Input sample rate, channel count and duration; use a synthetic or shareable clip if needed.

For benchmark issues, name the system and the board or axis in question.
If you ran a system on your own audio, say how you ran it, and explain what you hear as well as what a metric reports.

## Preparing a change

Keep changes focused and describe the behavior they fix.
Commit notebooks with outputs cleared and execution counts reset; keep editor
checkpoints untracked.
Follow [AGENTS.md](AGENTS.md): format touched Python with `isort` and `yapf -i`, check it with `pylint --rcfile=.pylintrc`, keep the license headers, and put tests beside the mirrored module under `tests/`.
Run the relevant tests from the repository root; run the full suite before proposing a release.

`make format` applies the pinned isort and yapf versions to `src/`, `docs/`,
`demo/` and `tests/`. Benchmark scripts and wrappers are checked by
`make format-check` but left unchanged by `make format`: their bytes identify
the scorer and the rendered outputs, so changing them requires the corresponding
re-score or re-render. Keep touched tests in arrange / act / assert sections
separated by blank lines, and exercise public behavior rather than private helpers.
`make lint` also checks tracked files for machine-specific paths.

Source archives contain the inference package, model documentation and README figures;
wheels contain the package and license notices. Use a repository checkout for
the benchmark, tests and development commands. CI checks both archive inventories.

Rebuild the architecture figures with `python docs/build_figures.py`. A smaller
installation can add just the `docs` extra with `pip install -e ".[docs]"`.
Benchmark charts additionally need `--leaderboard <board.json>`; a checkout
includes the generated charts.
The architecture is written directly as SVG, with live text in the benchmark
charts' font stack and a full accessible description. Both light and dark
variants are generated from the same layout.

Benchmark experiments write to `staging/` and local scratch.
Generate reported benchmark numbers with the leaderboard scripts.
Explain any change to model inference, metric meaning or ranking policy, so reviewers can assess its effect on comparisons.
Model weights, third-party code and private audio do not belong in patches.
