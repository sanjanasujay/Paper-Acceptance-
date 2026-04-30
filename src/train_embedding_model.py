from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, ensure_dirs, save_joblib, save_json

TRAIN_FILE = DATA_PROCESSED / "train.csv"
VAL_FILE = DATA_PROCESSED / "val.csv"
MODEL_FILE = MODELS_DIR / "embedding_model.joblib"
REPORT_FILE = REPORTS_DIR / "embedding_validation_metrics.json"
EMBEDDER_NAME = "all-MiniLM-L6-v2"


def get_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = [
        c for c in ["authors_count", "citation_count", "year", "author_experience", "abstract_length", "title_length"]
        if c in df.columns
    ]
    if not numeric_cols:
        return np.empty((len(df), 0))
    numeric_df = df[numeric_cols].copy()
    return numeric_df.to_numpy(dtype=float)


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

    y_train = train_df["accepted"].to_numpy()
    y_val = val_df["accepted"].to_numpy()

    embedder = SentenceTransformer(EMBEDDER_NAME)
    X_train_text = embedder.encode(train_df["text"].tolist(), show_progress_bar=True)
    X_val_text = embedder.encode(val_df["text"].tolist(), show_progress_bar=True)

    X_train_num = get_numeric_matrix(train_df)
    X_val_num = get_numeric_matrix(val_df)

    if X_train_num.shape[1] > 0:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(imputer.fit_transform(X_train_num))
        X_val_num = scaler.transform(imputer.transform(X_val_num))
        X_train = np.hstack([X_train_text, X_train_num])
        X_val = np.hstack([X_val_text, X_val_num])
    else:
        imputer = None
        scaler = None
        X_train = X_train_text
        X_val = X_val_text

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    classifier.fit(X_train, y_train)

    val_pred = classifier.predict(X_val)
    val_prob = classifier.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, val_pred, val_prob)
    package = {
        "embedder_name": EMBEDDER_NAME,
        "classifier": classifier,
        "imputer": imputer,
        "scaler": scaler,
    }
    save_joblib(package, MODEL_FILE)
    save_json(metrics, REPORT_FILE)

    print("Embedding model trained successfully.")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1: {metrics['f1']:.4f}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
