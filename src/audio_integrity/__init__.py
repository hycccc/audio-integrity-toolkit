"""audio-integrity-toolkit: acceptance-gate QC for audio datasets."""

from .analysis import Report, analyze
from .dedup import DupCluster, decode_hash, find_duplicates, fingerprint, similarity
from .metadata import MetaReport, verify

__all__ = [
    "Report", "analyze",
    "DupCluster", "decode_hash", "fingerprint", "similarity", "find_duplicates",
    "MetaReport", "verify",
]
__version__ = "0.2.0"
