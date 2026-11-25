# Examples

## Audio cleanup notebook

[audio_cleanup.ipynb](audio_cleanup.ipynb) is a guided first run of the offline model, for evaluating it on your own audio:

1. Install the package and check the machine.
2. Load the offline model; its weights (256 MB) download from Hugging Face on first use.
3. Clean a bundled noisy sample: speech over a hum and sharp taps.
4. Play the original, the cleaned voice and the removed background side by side.
5. Swap in your own recording and save the output.

Run it from a clone:

```bash
git clone https://github.com/inworld-ai/voice-isolator && cd voice-isolator
pip install jupyterlab
jupyter lab examples/audio_cleanup.ipynb
```

It also runs in Colab or any other Jupyter environment without a clone: it then installs the package from PyPI and downloads the sample.

A GPU is optional. See [Hardware](../README.md#hardware) for what the model is tested on.

## The smallest reusable code

To clean a file from your own code:

```python
from inworld import voice_isolation

isolator = voice_isolation.VoiceIsolator("inworld/voice-isolator")  # load once, reuse
wav, sample_rate = voice_isolation.load_audio("noisy.wav")
voice = isolator.isolate(wav, sample_rate)
voice_isolation.save_audio("clean.wav", voice, sample_rate)
```

From a terminal, `voice-isolate noisy.wav clean.wav` does the same and also writes the removed background.

For live audio, see [Streaming](../README.md#streaming) and the [interactive demo](../demo/README.md).
