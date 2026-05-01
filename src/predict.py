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


def build_df(title: str, abstract: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "title": title,
        "abstract": abstract,
        "text": f"{title} {abstract}",
        "authors_count": 0,
        "venue": "unknown",
        "abstract_length": len(abstract.split()),
        "title_length": len(title.split()),
    }])


def predict_one(pipeline, title: str, abstract: str) -> None:
    df = build_df(title, abstract)
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

    model_file = MODELS_DIR / "embedding_model.joblib"
    if not model_file.exists():
        model_file = MODELS_DIR / "tfidf_pipeline.joblib"
    model = load_joblib(model_file)
    print(f"Using model: {model_file.stem}\n")

    # Wrap embedding model to match pipeline interface
    from sentence_transformers import SentenceTransformer
    import numpy as np

    def get_proba(title, abstract):
        text = f"{title} {abstract}"
        if "embedder_name" in model:
            embedder = SentenceTransformer(model["embedder_name"])
            X_text = embedder.encode([text])
            abstract_length = len(abstract.split())
            title_length = len(title.split())
            X_num = np.array([[0, abstract_length, title_length]], dtype=float)
            X_num = model["scaler"].transform(model["imputer"].transform(X_num))
            X = np.hstack([X_text, X_num])
            return model["classifier"].predict_proba(X)[0][1]
        else:
            df = build_df(title, abstract)
            return model.predict_proba(df)[0][1]

    if args.pdf:
        title, abstract = extract_text_from_pdf(Path(args.pdf))
    elif args.title and args.abstract:
        title, abstract = args.title, args.abstract
    elif args.file:
        predict_csv(model, Path(args.file))
        exit()
    else:
        print("Provide --pdf, --file, or both --title and --abstract")
        exit()

    prob = get_proba(title, abstract)
    label = "ACCEPTED" if prob >= 0.5 else "REJECTED"
    print(f"Title:       {title}")
    print(f"Abstract:    {abstract[:150]}...")
    print(f"Result:      {label}")
    print(f"Confidence:  {prob:.2%} chance of acceptance")
