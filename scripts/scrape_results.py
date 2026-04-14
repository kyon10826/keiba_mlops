#!/usr/bin/env python3
"""netkeibaから過去のレース結果をスクレイピングする。

使い方:
    python scripts/scrape_results.py --year 2024
    python scripts/scrape_results.py --year 2025 --month 1
    python scripts/scrape_results.py --year 2025 --month 6 --place 東京
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_config
from src.scraper.results import scrape_race_results, save_results


def main():
    parser = argparse.ArgumentParser(description="Scrape historical race results")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape (e.g. 2024)")
    parser.add_argument("--month", type=int, default=None, help="Specific month (1-12)")
    parser.add_argument("--place", type=str, default=None, help="Specific place (e.g. 東京)")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["data"]["dir"]
    encoding = cfg["data"]["encoding"]

    os.makedirs(data_dir, exist_ok=True)

    print(f"Scraping race results for {args.year}"
          + (f" month={args.month}" if args.month else "")
          + (f" place={args.place}" if args.place else ""))

    df = scrape_race_results(args.year, month=args.month, place=args.place)

    if df.empty:
        print("No data scraped.")
        return

    # 出力ファイル名を組み立て
    suffix_parts = []
    if args.month:
        suffix_parts.append(f"{args.month:02d}")
    if args.place:
        suffix_parts.append(args.place)
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    output_path = os.path.join(data_dir, f"record_data_{args.year}{suffix}.csv")
    save_results(df, output_path, encoding=encoding)

    print(f"\nScraping complete!")
    print(f"  Rows: {len(df)}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
