import numpy as np
import pytest
import soundfile as sf

from audio_integrity import analyze

SR = 44100


def _write(tmp_path, name, signal, subtype="PCM_16"):
    path = tmp_path / name
    sf.write(path, signal, SR, subtype=subtype)
    return str(path)


def _noise(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * seconds)) * 0.1


def _brickwall(signal, cutoff_hz):
    """Zero all spectral content above cutoff_hz, mimicking a codec low-pass."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / SR)
    spectrum[freqs > cutoff_hz] = 0
    return np.fft.irfft(spectrum, n=len(signal))


def test_fullband_noise_passes(tmp_path):
    report = analyze(_write(tmp_path, "fullband.wav", _noise()))
    assert report.lossless
    assert report.cutoff_hz > SR / 2 - 1500


def test_lowpassed_noise_flagged_as_transcode(tmp_path):
    signal = _brickwall(_noise(), 15000)
    report = analyze(_write(tmp_path, "lowpassed.wav", signal, subtype="PCM_24"))
    assert not report.lossless
    assert "128kbps" in report.verdict


def test_320kbps_style_cutoff_flagged(tmp_path):
    signal = _brickwall(_noise(), 19800)
    report = analyze(_write(tmp_path, "aacish.wav", signal, subtype="PCM_24"))
    assert not report.lossless
    assert "320kbps" in report.verdict


def test_clipping_detected(tmp_path):
    signal = _noise()
    signal[1000:1100] = 1.0
    report = analyze(_write(tmp_path, "clipped.wav", signal))
    assert report.clipped_samples >= 100


def test_report_fields(tmp_path):
    report = analyze(_write(tmp_path, "meta.wav", _noise()))
    assert report.sample_rate == SR
    assert report.channels == 1
    assert report.duration_s == pytest.approx(2.0, abs=0.01)
    assert report.subtype == "PCM_16"
    assert report.crest_db > 0
