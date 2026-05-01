from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import DATA_PROCESSED, DATA_RAW, ensure_dirs

RAW_FILE = DATA_RAW / "papers.csv"

COLUMN_MAPPING = {
    "title": "title",
    "abstract": "abstract",
    "accepted": "accepted",
    "authors_count": "authors_count",
    "citation_count": "citation_count",
    "year": "year",
    "venue": "venue",
    "topic": "topic",
    "author_experience": "author_experience",
}

REQUIRED_COLUMNS = ["title", "abstract", "accepted"]
OPTIONAL_COLUMNS = [
    "authors_count",
    "citation_count",
    "year",
    "venue",
    "topic",
    "author_experience",
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    existing_lower = {c.lower().strip(): c for c in df.columns}
    for target, source in COLUMN_MAPPING.items():
        if source.lower() in existing_lower:
            rename_map[existing_lower[source.lower()]] = target
    df = df.rename(columns=rename_map)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}. "
            f"Update COLUMN_MAPPING in src/prepare_data.py to match your dataset."
        )

    keep_cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()

    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df = df[(df["title"].str.len() > 0) | (df["abstract"].str.len() > 0)].copy()

    df["accepted"] = pd.to_numeric(df["accepted"], errors="coerce")
    df = df[df["accepted"].isin([0, 1])].copy()
    df["accepted"] = df["accepted"].astype(int)

    for col in ["authors_count", "citation_count", "year", "author_experience"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["venue", "topic"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    df["text"] = (df["title"].fillna("") + " ") + df["abstract"].fillna("")
    df["abstract_length"] = df["abstract"].str.split().str.len().fillna(0)
    df["title_length"] = df["title"].str.split().str.len().fillna(0)

    return df.reset_index(drop=True)


def balance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    accepted = df[df["accepted"] == 1]
    rejected = df[df["accepted"] == 0]
    # Undersample rejected to 2x accepted to reduce bias
    rejected = rejected.sample(n=min(len(rejected), len(accepted) * 2), random_state=42)
    balanced = pd.concat([accepted, rejected]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced dataset: {len(accepted)} accepted, {len(rejected)} rejected")
    return balanced


def split_and_save(df: pd.DataFrame) -> None:
    df = balance_dataframe(df)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["accepted"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["accepted"],
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False)
    val_df.to_csv(DATA_PROCESSED / "val.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)

    print(f"Saved train: {len(train_df)} rows")
    print(f"Saved val:   {len(val_df)} rows")
    print(f"Saved test:  {len(test_df)} rows")


def main() -> None:
    ensure_dirs()
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Expected input file at {RAW_FILE}. Place your dataset there as papers.csv"
        )

    df = pd.read_csv(RAW_FILE)
    df = clean_dataframe(df)
    split_and_save(df)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
