"""Data fetchers for the Brazil working paper."""

from .bcb_sgs import BCBSGSFetcher
from .bcb_expectations import BCBExpectationsFetcher
from .sidra import SIDRAFetcher
from .fred import FREDFetcher
from .nyfed import NYFedFetcher
from .local_csv import LocalCSVFetcher
from .fiscal import FiscalFetcher

__all__ = [
    "BCBSGSFetcher",
    "BCBExpectationsFetcher",
    "SIDRAFetcher",
    "FREDFetcher",
    "NYFedFetcher",
    "LocalCSVFetcher",
    "FiscalFetcher",
]
