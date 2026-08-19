"""Duplicate detection for audio datasets.

Two tiers, mirroring how dataset ingestion actually needs them:

- **Exact duplicates** — a hash of the *decoded* signal, not the file
  bytes. The same master delivered as WAV by one vendor and FLAC by
  another is one song, not two; hashing the container would miss it.
- **Near duplicates** — a compact spectral fingerprint. Re-encodes,
  loudness-normalized copies, and lossy transcodes of the same recording
  survive none of the exact hashes but all sound alike; the fingerprint
  catches them by comparing how band energy *moves*, which is robust to
  level and codec changes.

The fingerprint is deliberately simple (log-spaced band energies,
delta-coded across time into per-frame bit patterns) — a reference
implementation of the idea behind production systems like Chromaprint.

Clustering has two strategies. **Pairwise** compares every pair — exact,
and fine for a delivery batch. **Indexed** buckets fingerprints with an
inverted index first (each frame word, quantized to its top 12 bits, is a
key; two files become a candidate pair only when they share at least
``MIN_SHARED_KEYS`` distinct keys) and runs the full similarity only on
candidates — the library-scale path, where comparisons grow with the
number of actual near-dups instead of n². The index is a recall filter,
never a judge: every candidate still passes through the same
``similarity()`` threshold, so a bucket collision can cost compute but
cannot create a false duplicate. The default strategy switches to the
index past ``INDEX_AUTO_THRESHOLD`` files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FP_SAMPLE_RATE = 11025
FP_FRAME = 4096            # ~0.37 s per frame at 11025 Hz
FP_HOP = 2048
FP_BANDS = 17              # 17 band energies -> 16 delta bits per frame
FP_MIN_FRAMES = 8          # refuse to fingerprint clips shorter than ~2 s
DEFAULT_THRESHOLD = 0.85   # bit-agreement to call a near-dup (chance level is 0.5)
MAX_OFFSET_FRAMES = 4      # tolerate small leading-silence misalignment
INDEX_KEY_BITS = 12        # frame words quantized to their top 12 bits as index keys
BOTTOM_K_KEYS = 32         # bottom-k sketch: only each file's K smallest keys are indexed
MIN_SHARED_KEYS = 2        # candidate pair needs >= this many shared sketch keys
INDEX_AUTO_THRESHOLD = 64  # "auto" strategy switches to the index past this many files


@dataclass
class DupCluster:
    """One group of files judged to be the same recording."""

    kind: str                    # "exact" | "near"
    paths: list[str]
    similarity: float            # 1.0 for exact clusters

    def to_dict(self) -> dict:
        return {"kind": self.kind, "paths": self.paths, "similarity": self.similarity}


def decode_hash(path: str) -> str:
    """SHA-256 of the decoded PCM — container-independent identity.

    Decoding to int16 hashes the stored samples, so the same master
    wrapped as WAV or FLAC hashes identically, and higher-depth sources
    still collide only when their audible content is the same.
    """
    data = sf.read(path, always_2d=True, dtype="int16")[0]
    return hashlib.sha256(data.mean(axis=1).astype(np.int16).tobytes()).hexdigest()


def fingerprint(path: str) -> np.ndarray:
    """Per-frame 16-bit spectral fingerprint of one file."""
    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    if sr != FP_SAMPLE_RATE:
        g = np.gcd(sr, FP_SAMPLE_RATE)
        mono = resample_poly(mono, FP_SAMPLE_RATE // g, sr // g)

    n_frames = 1 + max(0, (len(mono) - FP_FRAME)) // FP_HOP
    if n_frames < FP_MIN_FRAMES:
        raise ValueError(f"too short to fingerprint: {path}")

    window = np.hanning(FP_FRAME)
    edges = np.geomspace(300, FP_SAMPLE_RATE / 2 * 0.9, FP_BANDS + 1)
    freqs = np.fft.rfftfreq(FP_FRAME, 1 / FP_SAMPLE_RATE)
    band_of = np.searchsorted(edges, freqs) - 1

    energies = np.zeros((n_frames, FP_BANDS))
    for i in range(n_frames):
        frame = mono[i * FP_HOP : i * FP_HOP + FP_FRAME]
        spec = np.abs(np.fft.rfft(frame * window)) ** 2
        for b in range(FP_BANDS):
            sel = band_of == b
            energies[i, b] = spec[sel].sum() if sel.any() else 0.0
    energies = np.log(energies + 1e-12)

    # delta-code: bit b of frame i answers "did band b gain energy relative
    # to band b+1, compared with the previous frame?" — level-invariant
    d = (energies[1:, :-1] - energies[1:, 1:]) - (energies[:-1, :-1] - energies[:-1, 1:])
    bits = (d > 0).astype(np.uint16)
    return (bits << np.arange(FP_BANDS - 1, dtype=np.uint16)).sum(axis=1).astype(np.uint16)


def similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Best bit-agreement over small frame offsets, on the overlapping span."""
    best = 0.0
    for offset in range(-MAX_OFFSET_FRAMES, MAX_OFFSET_FRAMES + 1):
        a = fp_a[max(0, offset):]
        b = fp_b[max(0, -offset):]
        n = min(len(a), len(b))
        if n < FP_MIN_FRAMES:
            continue
        diff = np.bitwise_xor(a[:n], b[:n])
        wrong = np.unpackbits(diff.view(np.uint8)).sum()
        best = max(best, 1.0 - wrong / (n * 16))
    return best


def candidate_pairs(fps: dict[str, np.ndarray],
                    min_shared: int = MIN_SHARED_KEYS) -> list[tuple[str, str]]:
    """Inverted-index candidate generation for the near-dup pass.

    Each fingerprint's frame words, quantized to their top ``INDEX_KEY_BITS``
    bits, become index keys — quantizing tolerates bit noise in the low
    bands, and set-of-keys matching tolerates frame offsets for free. Only
    each file's ``BOTTOM_K_KEYS`` smallest keys enter the index (a bottom-k
    sketch): near-dups have near-identical key sets, so their smallest keys
    coincide, while full-length songs stop flooding the 2^12-key space and
    bucket sizes stay bounded no matter the file duration. A pair is a
    candidate when the sketches share at least ``min_shared`` keys.
    Recall filter only — candidates still face the real ``similarity()``.
    """
    shift = 16 - INDEX_KEY_BITS
    keys = {p: sorted(set(np.unique(fp >> shift).tolist()))[:BOTTOM_K_KEYS]
            for p, fp in fps.items()}
    index: dict[int, list[str]] = {}
    for p, ks in keys.items():
        for k in ks:
            index.setdefault(k, []).append(p)
    shared: dict[tuple[str, str], int] = {}
    for bucket in index.values():
        for i, a in enumerate(bucket):
            for b in bucket[i + 1:]:
                pair = (a, b) if a < b else (b, a)
                shared[pair] = shared.get(pair, 0) + 1
    return [pair for pair, n in shared.items() if n >= min_shared]


def find_duplicates(paths: list[str], threshold: float = DEFAULT_THRESHOLD,
                    strategy: str = "auto") -> list[DupCluster]:
    """Cluster a batch of files into exact- and near-duplicate groups.

    ``strategy``: ``"pairwise"`` compares every pair (exact, O(n²)),
    ``"indexed"`` generates candidates from an inverted index first
    (library scale), ``"auto"`` picks the index past
    ``INDEX_AUTO_THRESHOLD`` files.
    """
    if strategy not in ("auto", "pairwise", "indexed"):
        raise ValueError(f"unknown strategy: {strategy!r}")
    by_hash: dict[str, list[str]] = {}
    fps: dict[str, np.ndarray] = {}
    for p in paths:
        by_hash.setdefault(decode_hash(p), []).append(p)
        try:
            fps[p] = fingerprint(p)
        except ValueError:
            pass  # too short for near-dup analysis; exact tier still applies

    clusters = [
        DupCluster("exact", group, 1.0)
        for group in by_hash.values()
        if len(group) > 1
    ]

    # near-dup pass on one representative per exact group
    reps = [g[0] for g in by_hash.values() if g[0] in fps]
    parent = {p: p for p in reps}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if strategy == "indexed" or (strategy == "auto" and len(reps) > INDEX_AUTO_THRESHOLD):
        pairs = candidate_pairs({p: fps[p] for p in reps})
    else:
        pairs = [(a, b) for i, a in enumerate(reps) for b in reps[i + 1:]]

    sims: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        s = similarity(fps[a], fps[b])
        if s >= threshold:
            sims[(a, b) if a < b else (b, a)] = s
            parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for p in reps:
        groups.setdefault(find(p), []).append(p)
    for members in groups.values():
        if len(members) > 1:
            pair_sims = [s for (a, b), s in sims.items() if a in members and b in members]
            clusters.append(DupCluster("near", sorted(members), round(min(pair_sims), 4)))
    return clusters
