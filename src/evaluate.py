from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score

from utils import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_dirs, load_joblib, save_json

TEST_FILE = DATA_PROCESSED / "test.csv"


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


def get_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = [
        c for c in ["authors_count", "citation_count", "year", "author_experience", "abstract_length", "title_length"]
        if c in df.columns
    ]
    if not numeric_cols:
        return np.empty((len(df), 0))
    return df[numeric_cols].to_numpy(dtype=float)


def evaluate_tfidf(test_df: pd.DataFrame):
    pipeline = load_joblib(MODELS_DIR / "tfidf_pipeline.joblib")
    y_pred = pipeline.predict(test_df)
    y_prob = pipeline.predict_proba(test_df)[:, 1]
    return y_pred, y_prob


def evaluate_embedding(test_df: pd.DataFrame):
    package = load_joblib(MODELS_DIR / "embedding_model.joblib")
    embedder = SentenceTransformer(package["embedder_name"])
    classifier = package["classifier"]
    imputer = package["imputer"]
    scaler = package["scaler"]

    X_text = embedder.encode(test_df["text"].tolist(), show_progress_bar=True)
    X_num = get_numeric_matrix(test_df)

    if X_num.shape[1] > 0:
        X_num = imputer.transform(X_num)
        X_num = scaler.transform(X_num)
        X = np.hstack([X_text, X_num])
    else:
        X = X_text

    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)[:, 1]
    return y_pred, y_prob


def plot_probability_distribution(y_prob: np.ndarray, model_name: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob, bins=20)
    plt.xlabel("Predicted acceptance probability")
    plt.ylabel("Number of papers")
    plt.title(f"Acceptance Probability Distribution ({model_name})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name}_probability_distribution.png", dpi=200)
    plt.close()


def plot_class_balance(test_df: pd.DataFrame) -> None:
    counts = test_df["accepted"].value_counts().sort_index()
    labels = ["Rejected (0)", "Accepted (1)"]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution in Test Set")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_test.png", dpi=200)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()


def plot_roc(y_true, y_prob, model_name: str) -> None:
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC Curve ({model_name})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name}_roc_curve.png", dpi=200)
    plt.close()


def plot_pr(y_true, y_prob, model_name: str) -> None:
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"Precision-Recall Curve ({model_name})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name}_pr_curve.png", dpi=200)
    plt.close()


def plot_citation_trend(test_df: pd.DataFrame) -> None:
    if not {"citation_count", "accepted"}.issubset(test_df.columns):
        return
    grouped = test_df.groupby("accepted")["citation_count"].mean().sort_index()
    labels = ["Rejected (0)", "Accepted (1)"]
    values = [grouped.get(0, 0), grouped.get(1, 0)]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.xlabel("Class")
    plt.ylabel("Average citation count")
    plt.title("Citation Trend by Acceptance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "citation_trend_by_acceptance.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tfidf", "embedding"], required=True)
    args = parser.parse_args()

    ensure_dirs()
    test_df = pd.read_csv(TEST_FILE)
    y_true = test_df["accepted"].to_numpy()

    if args.model == "tfidf":
        y_pred, y_prob = evaluate_tfidf(test_df)
    else:
        y_pred, y_prob = evaluate_embedding(test_df)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    save_json(metrics, REPORTS_DIR / f"{args.model}_test_metrics.json")

    plot_class_balance(test_df)
    plot_probability_distribution(y_prob, args.model)
    plot_confusion_matrix(y_true, y_pred, args.model)
    plot_roc(y_true, y_prob, args.model)
    plot_pr(y_true, y_prob, args.model)
    plot_citation_trend(test_df)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Saved evaluation plots and metrics.")


if __name__ == "__main__":
    main()
