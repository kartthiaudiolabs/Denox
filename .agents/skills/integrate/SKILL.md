---
name: integrate
description: Integrate the Inworld Voice Isolator (the inworld-voice-isolation package) into another Python project or service via the pip package - install with the right extras, load weights by registry name, Hugging Face path or local file, call isolate() or the CLI correctly (rates, shapes, chunking, device, memory), and pin versions. Use when someone wants to use the model rather than develop it.
---

# Integrate the Inworld Voice Isolator

## Install

```bash
pip install inworld-voice-isolation                    # the library (plain PyTorch path)
pip install "inworld-voice-isolation[fast]"            # + the Mamba CUDA kernels (needs a CUDA toolchain to build)
```

- Python ≥ 3.10, PyTorch ≥ 2.1 (install torch for your CUDA first if you need a specific build).
- The `fast` extra provides optional CUDA Mamba kernels. Without it the scan runs in
  plain PyTorch on CUDA and the CPU; on Apple's GPU (`mps`) it runs as a built-in
  Metal kernel. Streaming uses plain PyTorch along time and the optional kernels
  across frequency. Measure throughput on the target hardware and audio.
- Weights come from the Hub on first use: `inworld/voice-isolator` (`denoiser.safetensors` 256 MB, `denoiser_realtime.safetensors` 233 MB, fp32 safetensors)
  lands in the standard HF cache (`HF_HOME` / `HF_HUB_CACHE`). For an air-gapped machine:
  `hf download inworld/voice-isolator` with `HF_HUB_OFFLINE=1` set afterwards.
- Importing the package never imports mamba_ssm, demucs, transformers or gradio and never
  touches the network; the first `VoiceIsolator(...)` does.
- When PyPI cannot install it, the install routes, in order, are in
  [the setup section of AGENTS.md](../../../AGENTS.md#setting-up-the-voice-isolator-for-a-user).
- After installing, run `voice-isolate-doctor --full`: it fetches both models, runs each
  on the device inference would use and ends with `All set: …` or one line per problem.

## Load

```python
from inworld import voice_isolation

isolator = voice_isolation.VoiceIsolator("inworld/voice-isolator")   # the Hub repository: file + sha256 check
isolator = voice_isolation.VoiceIsolator("voice-isolator-v1")                 # a registry name picks a checkpoint directly
isolator = voice_isolation.VoiceIsolator("hf://inworld/voice-isolator/denoiser.safetensors@main")
isolator = voice_isolation.VoiceIsolator("/models/denoiser.safetensors", device="cuda:1")
live = voice_isolation.VoiceIsolator("inworld/voice-isolator", mode="streaming", lookahead_frames=2)
# VoiceIsolator.load(...) is the same call with the same arguments
```

- `inworld/voice-isolator` loads the checkpoint for the mode: `voice-isolator-v1` offline, `voice-isolator-v1-realtime`
  for streaming. It is the default, so `VoiceIsolator()` and `VoiceIsolator(mode="streaming")` work too.
- `voice_isolation.MODELS` lists the registry (repo, filename, sha256, revision, stage, description).
- `hf://<org>/<repo>/<file>[@<revision>]` fetches any Hub file with no checksum; pin a
  revision (tag or commit) in production instead of `main`.
- `device` defaults to CUDA when available, else `mps` on an Apple-silicon Mac with PyTorch ≥ 2.6
  (earlier versions keep the CPU), else the CPU. Keep one `VoiceIsolator`
  per process/GPU and reuse it. It is not thread-safe across concurrent calls; serialise calls or
  run one worker per GPU.
- `mode` is `"offline"` (every frame sees its whole chunk, best quality, the default; for `voice-isolator-v1`) or
  `"streaming"` (causal, for `voice-isolator-v1-realtime`; `lookahead_frames` is one of
  `live.lookahead_choices`, frames of 10 ms). Each released checkpoint runs in one mode; only a
  dual-mode checkpoint switches with `.mode = ...`.

## Call

```python
clean = isolator.isolate(wav, sample_rate)                          # same type and rate back
clean48 = isolator.isolate(wav, sample_rate, output_sample_rate=48000)

session = live.stream(block_frames=4)                         # delay in samples: session.latency_samples
out = session.push(piece)                                     # the samples that became final (same type as the piece)
tail = session.flush()                                        # the rest, once the input ends
```

- `wav`: 1-D mono or 2-D stereo (`[C, T]` or `[T, C]`, downmixed to mono), NumPy array or
  torch Tensor, float in [-1, 1] or integer PCM. Output is mono, same container type, length
  matched to the input, at the input rate unless `output_sample_rate` is given. Resampling in
  and out is kaiser-quality (`voice_isolation.resample`).
- Any input rate works. Content above the input's bandwidth is never invented: a 16 kHz input
  comes back band-limited to 8 kHz even at `output_sample_rate=48000`.
- `offline_passes` (constructor argument, `--offline-passes` on the CLI) runs the offline model again over
  its own output, in the same chunks and precision: None takes the checkpoint's own count (one for the
  released files). Each pass costs one offline render, so two passes double the time; above one it applies
  to offline mode only.
- Offline, long audio is processed in the checkpoint's chunks, 8 s with a 2 s crossfade for `voice-isolator-v1` (`chunk_seconds`,
  `overlap_seconds` override); halve `chunk_seconds` if GPU memory is tight. In streaming mode
  `isolate()` runs the stream itself over the clip (no chunks, no seams, exactly what a live
  session gives); `isolate_many()` streams up to 16 clips in
  lockstep for throughput.
- The input's level is left alone: output level matches input level, with no normalisation.
  Isolated speech can peak above full scale; when writing integer PCM, scale such a file down
  rather than clipping it (`voice_isolation.save_audio` does).
- The streaming session wants 48 kHz mono float samples; resample before pushing. Pieces of any
  size work and never change the output; `block_frames` controls work per call, without
  adding algorithmic latency. See `MODEL_CARD.md` for the algorithmic latency.
  The state is never reset within a session.

## Batch / CLI

```bash
voice-isolate noisy.wav clean.wav                          # clean.wav and clean_background.wav, both float32
voice-isolate in_dir/ out_dir/ --shard 0 --num-shards 4       # resumable, one worker per shard/GPU
voice-isolate in.wav out.wav --model hf://inworld/voice-isolator/denoiser.safetensors
voice-isolate in.wav out.wav --output-sr 0                  # keep the input rate for both float WAVs
voice-isolate in.wav out.wav --background noise.wav          # explicit background path, one input file
voice-isolate in.wav out.wav --no-background --float32        # voice only, float WAV
voice-isolate in.wav out.wav --mode streaming --lookahead 2   # what a live stream with 20 ms look-ahead gives
```

Offline writes the requested voice WAV and `<output-stem>_background.wav`. Both use
float32 at the output rate, without independent normalization; their sum reconstructs
the aligned mono input. `--background` overrides the companion path for one input
file. `--no-background` disables it and uses PCM16 unless `--float32` is supplied.
Streaming writes only the voice.

Directory mode mirrors the tree and resumes only when all requested outputs for a
recording exist. The output directory must be outside the input tree. Collisions
with another output or any input are refused before inference. A failed file prints
`FAIL <file>` and the batch continues with a nonzero exit status.
`CUDA_VISIBLE_DEVICES` selects the GPU per worker.

## Pinning and updates

- Pin the package (`inworld-voice-isolation==<version>`) and refer to the model by the repository id
  or a registry name: the name → file → sha256 mapping is part of the package, so a package version always
  resolves the same weights. Use `hf://...@<revision>` only when you deliberately want a file
  the registry does not know.
- The package version tracks the API, the registry name tracks the model (`voice-isolator-v1` for
  recorded audio, offline; `voice-isolator-v1-realtime` for live audio, streaming only). Read the model card (`README.md` on the Hub
  repo, `MODEL_CARD.md` in this repository) for what the weights were trained and evaluated on.
- Licenses: code and weights Apache-2.0; the Mamba kernels Apache-2.0.

## Throughput

`isolator.isolate_many([clip, ...], sample_rate)` isolates a list of clips, grouping equal
lengths into one separator pass; results equal `isolate()` to about 1e-5 and come back in input
order. On a GPU the separator is compute-bound, so for bulk work spread clips over several
processes or cards rather than expecting batching alone to multiply speed.
