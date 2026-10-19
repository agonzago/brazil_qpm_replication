"""Base fetcher ABC and parallel execution utilities."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..variable_registry import VariableDef


class BaseFetcher(ABC):
    """Abstract base for all data fetchers.

    Each concrete fetcher downloads data from a single source and returns
    a quarterly DataFrame with columns matching the canonical names in
    ``variable_registry.REGISTRY``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable fetcher name (e.g. 'BCB SGS')."""

    @property
    @abstractmethod
    def variables(self) -> list[VariableDef]:
        """Variables this fetcher provides."""

    @abstractmethod
    def fetch(self, start: str, end: str) -> pd.DataFrame:
        """Fetch data and return a quarterly DataFrame.

        Parameters
        ----------
        start, end : str
            Date strings (e.g. "2000-01-01", "2025-12-31").

        Returns
        -------
        pd.DataFrame
            Index = quarterly ``pd.PeriodIndex``, columns = canonical names.
        """


def to_quarterly(
    df: pd.DataFrame,
    method: str = "mean",
) -> pd.DataFrame:
    """Resample a DatetimeIndex DataFrame to quarterly PeriodIndex.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex (daily or monthly).
    method : "mean", "last", or "sum".
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if method == "mean":
        q = df.resample("QS").mean()
    elif method == "last":
        q = df.resample("QS").last()
    elif method == "sum":
        q = df.resample("QS").sum()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    q.index = q.index.to_period("Q")
    return q


def fetch_all_parallel(
    fetchers: list[BaseFetcher],
    start: str,
    end: str,
    max_workers: int = 7,
) -> dict[str, pd.DataFrame]:
    """Run all fetchers in parallel and return {fetcher_name: DataFrame}.

    Failed fetchers are reported but do not stop the pipeline.
    """
    t0 = time.perf_counter()
    results: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    print(f"  Launching {len(fetchers)} fetchers in parallel "
          f"(max_workers={max_workers}) ...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(f.fetch, start, end): f for f in fetchers
        }
        for future in as_completed(future_map):
            fetcher = future_map[future]
            try:
                df = future.result()
                results[fetcher.name] = df
                print(f"    {fetcher.name}: {df.shape[1]} columns, "
                      f"{df.shape[0]} rows")
            except Exception as exc:
                errors[fetcher.name] = str(exc)
                print(f"    {fetcher.name}: FAILED — {exc}")

    elapsed = time.perf_counter() - t0
    print(f"  Parallel fetch completed in {elapsed:.1f}s  "
          f"({len(results)} OK, {len(errors)} failed)")

    if errors:
        print("  Failed fetchers:")
        for name, msg in errors.items():
            print(f"    {name}: {msg}")

    return results
