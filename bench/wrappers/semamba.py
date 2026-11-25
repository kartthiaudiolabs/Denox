"""Runs a release of NVIDIA's SEMamba family as its inference script does.

RE-USE and Real-time RE-USE (NSCLv1, research and development only) ship
their code as plain ``models`` and ``utils`` directories beside the weights,
define the STFT at an 8 kHz base that scales with the input rate, and
post-process alike. ``bind`` points the two package names at a release's
directories (the venv's packages of the same names would otherwise shadow
them); ``enhancer`` returns the release's processing of one file: the STFT
sizes scaled to the input rate and made even, magnitude and phase in and out,
frames whose magnitude is more than half zeros blanked (the releases' guard
against a sweep artefact), the inverse STFT, the length matched to the input.
"""
import importlib
import os
import sys
import types

import torch

import writer  # the shared render loop, beside this file


def bind(src: str) -> tuple:
    """Binds ``models`` and ``utils`` to the release under ``src``.

    Returns:
        The release's ``utils.util`` and ``models.stfts`` modules.
    """
    for package in ("models", "utils"):
        namespace = types.ModuleType(package)
        namespace.__path__ = [os.path.join(src, package)]
        sys.modules[package] = namespace
    return (importlib.import_module("utils.util"),
            importlib.import_module("models.stfts"))


def make_even(value: float) -> int:
    """Returns ``value`` rounded, then raised to the next even integer."""
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def enhancer(model_call, util, stfts, cfg: dict, device: str) -> writer.Enhance:
    """Returns ``enhance(path, samples, rate)`` for a loaded release.

    Args:
        model_call: ``(magnitude, phase) -> (amplitude, phase, ...)``.
        util: The release's ``utils.util``.
        stfts: The release's ``models.stfts``.
        cfg: The release's configuration.
        device: The device the model computes on.
    """
    stft = cfg["stft_cfg"]
    compress = cfg["model_cfg"]["compress_factor"]
    relu = torch.nn.ReLU()

    def enhance(path, samples, rate):  # pylint: disable=unused-argument
        scale = rate / stft["sampling_rate"]
        n_fft = make_even(stft["n_fft"] * scale)
        hop = make_even(stft["hop_size"] * scale)
        win = make_even(stft["win_size"] * scale)
        x = torch.from_numpy(samples).unsqueeze(0).to(device)
        with torch.no_grad():
            mag, pha, _ = stfts.mag_phase_stft(x,
                                               n_fft=n_fft,
                                               hop_size=hop,
                                               win_size=win,
                                               compress_factor=compress,
                                               center=True,
                                               addeps=False)
            amp, pha_out, _ = model_call(mag, pha)
            linear = torch.expm1(relu(amp))
            mostly_zero = (linear == 0).sum(1) / linear.shape[1] > 0.5
            amp[:, :, mostly_zero[0]] = 0
            y = stfts.mag_phase_istft(amp, pha_out, n_fft, hop, win, compress)
            y = util.pad_or_trim_to_match(x, y, pad_value=1e-8)
        return y.squeeze(0).float().cpu().numpy(), rate

    return enhance
