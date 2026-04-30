from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, ensure_dirs, save_joblib, save_json

TRAIN_FILE = DATA_PROCESSED / "train.csv"
VAL_FILE = DATA_PROCESSED / "val.csv"
MODEL_FILE = MODELS_DIR / "tfidf_pipeline.joblib"
REPORT_FILE = REPORTS_DIR / "tfidf_validation_metrics.json"


def build_feature_lists(df: pd.DataFrame) -> tuple[str, List[str], List[str]]:
    text_col = "text"
    numeric_cols = [
        c for c in ["authors_count", "citation_count", "year", "author_experience", "abstract_length", "title_length"]
        if c in df.columns
    ]
    categorical_cols = [c for c in ["venue", "topic"] if c in df.columns]
    return text_col, numeric_cols, categorical_cols


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    text_col, numeric_cols, categorical_cols = build_feature_lists(df)

    transformers = [
        (
            "text",
            TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english"),
            text_col,
        )
    ]

    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    return pipeline


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def main() -> None:
    ensure_dirs()
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    y_train = train_df["accepted"]
    y_val = val_df["accepted"]

    pipeline = build_pipeline(train_df)
    pipeline.fit(train_df, y_train)

    val_pred = pipeline.predict(val_df)
    val_prob = pipeline.predict_proba(val_df)[:, 1]

    metrics = compute_metrics(y_val, val_pred, val_prob)
    save_joblib(pipeline, MODEL_FILE)
    save_json(metrics, REPORT_FILE)

    print("TF-IDF model trained successfully.")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1: {metrics['f1']:.4f}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
