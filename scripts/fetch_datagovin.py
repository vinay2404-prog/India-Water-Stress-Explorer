"""Fetch raw datasets from data.gov.in into data/raw/.

Usage:
    python scripts/fetch_datagovin.py --list                 # show configured datasets
    python scripts/fetch_datagovin.py --search "rainfall"    # discover resource IDs
    python scripts/fetch_datagovin.py --all                  # fetch every configured dataset
    python scripts/fetch_datagovin.py --name district_rainfall_monthly
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datagovin import DataGovIn  # noqa: E402

CONFIG = ROOT / "config" / "datagovin_resources.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list configured datasets")
    ap.add_argument("--search", metavar="KEYWORD", help="search data.gov.in catalog")
    ap.add_argument("--all", action="store_true", help="fetch all configured datasets")
    ap.add_argument("--name", metavar="NAME", help="fetch one configured dataset by name")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    datasets = cfg["datasets"]

    if args.list:
        for d in datasets:
            rid = d["resource_id"]
            status = "READY" if not rid.startswith("REPLACE") else "NEEDS ID"
            print(f"{d['name']:32s} {status:10s} -> {d['output']}")
        return

    client = DataGovIn()

    if args.search:
        print(f"Searching data.gov.in for '{args.search}'...")
        hits = client.search(args.search, max_pages=5, page_size=10)
        if not hits:
            print("No matches (catalog endpoint may be down; try the portal search at https://data.gov.in/search).")
            return
        for h in hits:
            print(f"{h['index_name']}  |  {h['title']}  |  {h['org']}")
        return

    targets = datasets
    if args.name:
        targets = [d for d in datasets if d["name"] == args.name]
        if not targets:
            print(f"Unknown dataset: {args.name}"); sys.exit(1)

    for d in targets:
        rid = d["resource_id"]
        if rid.startswith("REPLACE"):
            print(f"SKIP {d['name']}: resource_id not set (see config/datagovin_resources.json)")
            continue
        print(f"Fetching {d['name']} ({rid}) ...")
        df = client.fetch(rid, filters=d.get("filters") or {})
        out = ROOT / d["output"]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  wrote {len(df)} rows -> {out}")
        missing = [c for c in d.get("required_columns", []) if c not in df.columns]
        if missing:
            print(f"  WARNING: missing expected columns: {missing}")
            print(f"  actual columns: {list(df.columns)}")


if __name__ == "__main__":
    main()