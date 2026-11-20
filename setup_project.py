# ============================================================
# SENTIMENT INTELLIGENCE - PROJECT SETUP & TRAINING PIPELINE
# ============================================================

import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SRC_DIR = os.path.join(BASE_DIR, "src")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")


# ============================================================
# 2. CREATE PROJECT DIRECTORIES
# ============================================================

directories = [
    RAW_DIR,
    PROCESSED_DIR,
    EXTERNAL_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SRC_DIR,
    DASHBOARD_DIR
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

print("Project directories ready.")


# ============================================================
# 3. DATA SOURCE
# ============================================================

DATA_URL = (
    "hf://datasets/"
    "Madnesss/amazon-beauty-analytics-ready/"
    "reviews"
)

print("\nLoading dataset...")


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_parquet(
    DATA_URL,
    filters=[("split", "=", "train")],
    columns=[
        "text_full",
        "sentiment",
        "rating"
    ]
)

print("Original dataset shape:", df.shape)


# ============================================================
# 5. SAMPLE BALANCED DATASET
# ============================================================

df = (
    df.groupby(
        "sentiment",
        group_keys=False
    )
    .sample(
        n=5000,
        random_state=42
    )
    .reset_index(drop=True)
)

print("Balanced dataset shape:", df.shape)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# 6. SAVE RAW DATA
# ============================================================

raw_path = os.path.join(
    RAW_DIR,
    "amazon_beauty_reviews_sample.csv"
)

df.to_csv(
    raw_path,
    index=False
)

print("\nRaw dataset saved:")
print(raw_path)


# ============================================================
# 7. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_text"] = (
    df["text_full"]
    .apply(preprocess_text)
)


# ============================================================
# 8. REMOVE DUPLICATE REVIEWS
# ============================================================

before_duplicates = len(df)

df = (
    df.drop_duplicates(
        subset=["text_full"],
        keep="first"
    )
    .copy()
)

after_duplicates = len(df)

print(
    f"\nRemoved duplicates: "
    f"{before_duplicates - after_duplicates}"
)

print("Final dataset shape:", df.shape)


# ============================================================
# 9. SAVE PROCESSED DATA
# ============================================================

processed_path = os.path.join(
    PROCESSED_DIR,
    "sentiment_data.csv"
)

df.to_csv(
    processed_path,
    index=False
)

print("\nProcessed dataset saved:")
print(processed_path)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_text = df["clean_text"]
y = df["sentiment"]

X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_text_train))
print("Testing samples:", len(X_text_test))


# ============================================================
# 11. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(
    X_text_train
)

X_test = vectorizer.transform(
    X_text_test
)

print("\nTF-IDF training shape:", X_train.shape)
print("TF-IDF testing shape:", X_test.shape)


# ============================================================
# 12. TRAIN FINAL MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel training completed.")


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy: "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "negative",
            "neutral",
            "positive"
        ]
    )
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODELS_DIR,
    "sentiment_logistic_regression.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved:")
print(model_path)


# ============================================================
# 15. SAVE VECTORIZER
# ============================================================

vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    vectorizer,
    vectorizer_path
)

print("\nVectorizer saved:")
print(vectorizer_path)


# ============================================================
# 16. SAVE MODEL METRICS
# ============================================================

metrics = pd.DataFrame({
    "metric": [
        "accuracy",
        "macro_f1",
        "training_samples",
        "testing_samples",
        "tfidf_features"
    ],
    "value": [
        accuracy,
        macro_f1,
        len(X_text_train),
        len(X_text_test),
        X_train.shape[1]
    ]
})

metrics_path = os.path.join(
    REPORTS_DIR,
    "model_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print("\nMetrics saved:")
print(metrics_path)


# ============================================================
# 17. SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred
)

report_path = os.path.join(
    REPORTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Sentiment Intelligence System\n"
    )

    file.write(
        "Final Model: "
        "TF-IDF + Logistic Regression\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Macro F1: "
        f"{macro_f1:.4f}\n\n"
    )

    file.write(
        report
    )

print("\nClassification report saved:")
print(report_path)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT PIPELINE COMPLETED")
print("=" * 60)

print("\nCreated:")

print("✓ Raw dataset")
print("✓ Processed dataset")
print("✓ TF-IDF vectorizer")
print("✓ Trained Logistic Regression model")
print("✓ Model metrics")
print("✓ Classification report")

print("\nFinal accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nProject directories are ready.")