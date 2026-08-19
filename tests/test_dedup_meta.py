import numpy as np
import soundfile as sf

from audio_integrity import decode_hash, find_duplicates, fingerprint, similarity, verify

SR = 44100


def _song(seconds=8.0, seed=0):
    """Synthetic 'music': detuned partials over a pink-noise bed, moving."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    signal = np.zeros_like(t)
    for f in rng.uniform(120, 2200, size=6):
        signal += rng.uniform(0.05, 0.2) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / SR), 1.0))  # ~pink
    bed = np.fft.irfft(spectrum, n=n)
    signal += 0.35 * bed / np.max(np.abs(bed))
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)  # amplitude movement
    return (signal / np.max(np.abs(signal)) * 0.7).astype(np.float64)


def _write(tmp_path, name, signal, sr=SR, subtype="PCM_16"):
    path = tmp_path / name
    sf.write(path, signal, sr, subtype=subtype)
    return str(path)


# ---------------------------------------------------------------- dedup

def test_same_audio_different_container_is_exact_dup(tmp_path):
    song = _song(seed=1)
    a = _write(tmp_path, "a.wav", song)
    # the same PCM re-wrapped in another container, as a vendor would ship it
    b = _write(tmp_path, "b.flac", sf.read(a, dtype="int16")[0])
    assert decode_hash(a) == decode_hash(b)
    clusters = find_duplicates([a, b])
    assert len(clusters) == 1 and clusters[0].kind == "exact"
    assert sorted(clusters[0].paths) == sorted([a, b])


def test_perturbed_copy_is_near_dup_and_different_song_is_not(tmp_path):
    song = _song(seed=2)
    noisy = song + np.random.default_rng(0).standard_normal(len(song)) * 0.003
    other = _song(seed=9)
    a = _write(tmp_path, "orig.wav", song)
    b = _write(tmp_path, "noisy.wav", noisy)
    c = _write(tmp_path, "other.wav", other)

    assert decode_hash(a) != decode_hash(b)          # exact tier must miss it
    assert similarity(fingerprint(a), fingerprint(b)) >= 0.85
    assert similarity(fingerprint(a), fingerprint(c)) < 0.85

    clusters = find_duplicates([a, b, c])
    near = [cl for cl in clusters if cl.kind == "near"]
    assert len(near) == 1
    assert sorted(near[0].paths) == sorted([a, b])
    assert c not in near[0].paths


def test_level_change_survives_fingerprint(tmp_path):
    song = _song(seed=3)
    a = _write(tmp_path, "full.wav", song)
    b = _write(tmp_path, "quiet.wav", song * 0.3)
    assert similarity(fingerprint(a), fingerprint(b)) >= 0.85


# ---------------------------------------------------------------- meta

def test_clean_file_passes(tmp_path):
    report = verify(_write(tmp_path, "clean.wav", _song(seed=4)))
    assert report.passed and report.issues == []


def test_container_mismatch_detected(tmp_path):
    song = _song(seed=5)
    flac = tmp_path / "song.flac"
    sf.write(flac, song, SR)
    lying = tmp_path / "song.wav"
    lying.write_bytes(flac.read_bytes())             # FLAC bytes, .wav name
    report = verify(str(lying))
    assert not report.passed
    assert any("container mismatch" in i for i in report.issues)


def test_edge_silence_detected(tmp_path):
    song = _song(seed=6)
    padded = np.concatenate([np.zeros(SR * 3), song])
    report = verify(_write(tmp_path, "padded.wav", padded))
    assert not report.passed
    assert any("leading silence" in i for i in report.issues)
    assert report.leading_silence_s > 2.5


def test_nonstandard_rate_detected(tmp_path):
    report = verify(_write(tmp_path, "odd.wav", _song(seed=7)[: 37123 * 3], sr=37123))
    assert not report.passed
    assert any("nonstandard sample rate" in i for i in report.issues)
