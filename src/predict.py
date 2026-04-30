from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from utils import MODELS_DIR, load_joblib


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    reader = PdfReader(str(pdf_path))
    # Extract first 2 pages — usually contains title + abstract
    text = ""
    for page in reader.pages[:2]:
        text += page.extract_text() or ""

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # First non-empty line is usually the title
    title = lines[0] if lines else pdf_path.stem

    # Find abstract section
    abstract = ""
    full = " ".join(lines)
    match = re.search(r"(?i)abstract[.\s:—-]*(.+?)(?=introduction|1\s*introduction|keywords|$)", full, re.DOTALL)
    if match:
        abstract = match.group(1).strip()
    else:
        # fallback: use first 300 words after title
        abstract = " ".join(lines[1:])[:1000]

    return title, abstract


def predict_one(pipeline, title: str, abstract: str) -> None:
    df = pd.DataFrame([{"title": title, "abstract": abstract, "text": f"{title} {abstract}"}])
    prob = pipeline.predict_proba(df)[0][1]
    label = "ACCEPTED" if prob >= 0.5 else "REJECTED"
    print(f"\nTitle:       {title}")
    print(f"Abstract:    {abstract[:150]}...")
    print(f"Result:      {label}")
    print(f"Confidence:  {prob:.2%} chance of acceptance\n")


def predict_csv(pipeline, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if "title" not in df.columns or "abstract" not in df.columns:
        raise ValueError("CSV must have 'title' and 'abstract' columns")
    df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    probs = pipeline.predict_proba(df)[:, 1]
    for i, row in df.iterrows():
        label = "ACCEPTED" if probs[i] >= 0.5 else "REJECTED"
        print(f"\nTitle:       {row['title']}")
        print(f"Result:      {label}")
        print(f"Confidence:  {probs[i]:.2%} chance of acceptance")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path to a PDF file")
    parser.add_argument("--file", help="Path to a CSV file with title and abstract columns")
    parser.add_argument("--title", help="Paper title")
    parser.add_argument("--abstract", help="Paper abstract")
    args = parser.parse_args()

    pipeline = load_joblib(MODELS_DIR / "tfidf_pipeline.joblib")

    if args.pdf:
        title, abstract = extract_text_from_pdf(Path(args.pdf))
        predict_one(pipeline, title, abstract)
    elif args.file:
        predict_csv(pipeline, Path(args.file))
    elif args.title and args.abstract:
        predict_one(pipeline, args.title, args.abstract)
    else:
        print("Provide --pdf, --file, or both --title and --abstract")
