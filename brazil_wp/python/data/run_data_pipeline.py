#!/usr/bin/env python3
"""CLI entrypoint for the Brazil WP data pipeline.

Usage:
    .venv/bin/python data/run_data_pipeline.py                 # full pipeline
    .venv/bin/python data/run_data_pipeline.py --fetch-only    # download + consolidate
    .venv/bin/python data/run_data_pipeline.py --transform-only
    .venv/bin/python data/run_data_pipeline.py --plots-only
    .venv/bin/python data/run_data_pipeline.py --metadata      # print registry
    .venv/bin/python data/run_data_pipeline.py --output-dir shared_data  # custom output
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from the python/ directory
# Insert parent of data/ (i.e. python/) so that `from data.xxx` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brazil WP data pipeline: fetch, transform, plot."
    )
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch and consolidate data.")
    parser.add_argument("--transform-only", action="store_true",
                        help="Only transform (requires existing consolidated CSV).")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only generate stylized facts plots.")
    parser.add_argument("--metadata", action="store_true",
                        help="Print the variable registry as a markdown table.")
    parser.add_argument("--start", default="1990-01-01",
                        help="Start date (default: 1990-01-01).")
    parser.add_argument("--end", default="2025-12-31",
                        help="End date (default: 2025-12-31).")
    parser.add_argument("--workers", type=int, default=7,
                        help="Max parallel workers (default: 7).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: python/estimation_data/).")

    args = parser.parse_args()

    # --metadata: print and exit
    if args.metadata:
        from data.variable_registry import print_registry_table
        print_registry_table()
        return

    t0 = time.perf_counter()
    consolidated = None
    transformed = None

    # Determine which stages to run
    run_fetch = not args.transform_only and not args.plots_only
    run_transform = not args.fetch_only and not args.plots_only
    run_plots = not args.fetch_only and not args.transform_only

    # Resolve output directory
    output_dir = args.output_dir
    if output_dir is not None:
        # Resolve relative to python/ working directory
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    # --- Stage 1: Fetch + Consolidate ---
    if run_fetch:
        print("\n" + "=" * 72)
        print("  STAGE 1: Fetch + Consolidate")
        print("=" * 72 + "\n")
        from data.consolidate import consolidate
        consolidated = consolidate(
            start=args.start, end=args.end,
            max_workers=args.workers,
            output_dir=output_dir,
        )

    # --- Stage 2: Transform ---
    if run_transform:
        print("\n" + "=" * 72)
        print("  STAGE 2: Transform")
        print("=" * 72 + "\n")
        from data.transform import apply_transforms
        transformed = apply_transforms(
            consolidated=consolidated,
            output_dir=output_dir,
        )

    # --- Stage 3: Stylized Facts ---
    if run_plots:
        print("\n" + "=" * 72)
        print("  STAGE 3: Stylized Facts Plots")
        print("=" * 72 + "\n")
        from data.stylized_facts import generate_stylized_facts
        generate_stylized_facts(transformed=transformed)

    elapsed = time.perf_counter() - t0
    print(f"\n  Pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
