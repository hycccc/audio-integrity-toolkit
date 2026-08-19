"""audio-integrity-toolkit: acceptance-gate QC for audio datasets."""

from .analysis import Report, analyze
from .dedup import DupCluster, candidate_pairs, decode_hash, find_duplicates, fingerprint, similarity
from .metadata import MetaReport, verify

__all__ = [
    "Report", "analyze",
    "DupCluster", "candidate_pairs", "decode_hash", "fingerprint", "similarity", "find_duplicates",
    "MetaReport", "verify",
]
__version__ = "0.3.0"
