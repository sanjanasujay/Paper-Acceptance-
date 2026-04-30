# A Data Driven Approach to Predicting Academic Paper Acceptance

Starter GitHub repo for building a machine learning pipeline that predicts whether a paper will be **accepted** or **rejected** using:
- text features from **title** and **abstract**
- metadata features such as **author count**, **citation count**, **year**, and optional **venue/topic**
- a baseline **Logistic Regression** model with TF-IDF
- an optional **Sentence Transformer** embedding model
- visual analysis and evaluation reports

## Suggested project structure

```bash
paper_acceptance_ml/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
│   ├── prepare_data.py
│   ├── train_tfidf_model.py
│   ├── train_embedding_model.py
│   ├── evaluate.py
│   └── utils.py
├── .gitignore
├── requirements.txt
└── README.md
```

## What this repo expects

A CSV file in `data/raw/papers.csv` with at least these columns:

| column | meaning |
|---|---|
| `title` | paper title |
| `abstract` | paper abstract |
| `accepted` | 1 for accepted, 0 for rejected |

Optional columns:
- `authors_count`
- `citation_count`
- `year`
- `venue`
- `topic`
- `author_experience`

If your dataset uses different names, update `COLUMN_MAPPING` in `src/prepare_data.py`.

## Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Put your dataset here

```bash
data/raw/papers.csv
```

### 4) Prepare the dataset

```bash
python src/prepare_data.py
```

### 5) Train the TF-IDF baseline

```bash
python src/train_tfidf_model.py
```

### 6) Evaluate and generate plots

```bash
python src/evaluate.py --model tfidf
```

### 7) Optional: Train the embedding model

```bash
python src/train_embedding_model.py
python src/evaluate.py --model embedding
```

## Expected outputs

After running the pipeline, you should see:
- `data/processed/train.csv`, `val.csv`, `test.csv`
- `models/tfidf_pipeline.joblib`
- `models/embedding_model.joblib`
- `outputs/reports/*.json`
- `outputs/figures/*.png`

## Proposal alignment

This repo directly supports your proposal by:
- using NLP on titles and abstracts
- combining text features with metadata
- predicting paper acceptance as a binary classification task
- reporting accuracy, precision, recall, F1, ROC-AUC
- generating visual analysis like class balance, acceptance probability distribution, and top TF-IDF features

## Notes on datasets

PeerRead contains over 14K paper drafts with accept/reject decisions and train/dev/test splits for top-tier venues such as ACL, NIPS, and ICLR, making it a strong starting point for acceptance prediction. citeturn605334view0

The ORB paper describes an Open Review-Based dataset for automatic assessment of scientific papers and proposals, which makes it a useful secondary source if you later want to expand beyond a single dataset. citeturn605334view2

## Git commands

```bash
git init
git add .
git commit -m "Initial commit: paper acceptance prediction project"
```

If you create a new GitHub repo:

```bash
git remote add origin YOUR_GITHUB_REPO_URL
git branch -M main
git push -u origin main
```
