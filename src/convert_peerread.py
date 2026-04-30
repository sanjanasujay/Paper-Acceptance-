"""
Convert PeerRead dataset to papers.csv

Usage:
    python src/convert_peerread.py --peerread_dir /path/to/PeerRead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils import DATA_RAW, ensure_dirs

def load_papers(peerread_dir: Path) -> list[dict]:
    rows = []
    for reviews_dir in peerread_dir.glob("data/*/*/reviews"):
        venue = reviews_dir.parts[-3]
        for json_file in reviews_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8").split('\n')[0])
                title = data.get("title", "")
                abstract = data.get("abstract", "")
                accepted = data.get("accepted", None)
                if not title or not abstract or accepted is None:
                    continue
                authors = data.get("authors", "")
                authors_count = len(authors.split(",")) if isinstance(authors, str) else len(authors)
                rows.append({
                    "title": title,
                    "abstract": abstract,
                    "accepted": int(bool(accepted)),
                    "venue": venue,
                    "authors_count": authors_count,
                })
            except Exception:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peerread_dir", required=True, help="Path to cloned PeerRead repo")
    args = parser.parse_args()

    ensure_dirs()
    rows = load_papers(Path(args.peerread_dir))
    if not rows:
        print("No papers found. Check that --peerread_dir points to the cloned PeerRead repo.")
        return

    df = pd.DataFrame(rows)
    out = DATA_RAW / "papers.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} papers to {out}")
    print(f"Accepted: {df['accepted'].sum()} | Rejected: {(df['accepted'] == 0).sum()}")


if __name__ == "__main__":
    main()
