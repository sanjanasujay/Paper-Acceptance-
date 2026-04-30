from __future__ import annotations

import argparse

import pandas as pd

from utils import MODELS_DIR, load_joblib


def predict(title: str, abstract: str) -> None:
    pipeline = load_joblib(MODELS_DIR / "tfidf_pipeline.joblib")

    df = pd.DataFrame([{"title": title, "abstract": abstract, "text": f"{title} {abstract}"}])

    prob = pipeline.predict_proba(df)[0][1]
    label = "ACCEPTED" if prob >= 0.5 else "REJECTED"

    print(f"\nResult:      {label}")
    print(f"Confidence:  {prob:.2%} chance of acceptance\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True, help="Paper title")
    parser.add_argument("--abstract", required=True, help="Paper abstract")
    args = parser.parse_args()

    predict(args.title, args.abstract)
