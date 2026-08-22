"""
Failure Classifier — AI Payment Recovery Agent

Maps noisy, real-world-style gateway failure text (e.g. "DO_NOT_HONOR",
"Txn declined - low balance") to a clean failure category:
  insufficient_funds | card_expired | bank_declined | network_error
  | invalid_cvv | other

Approach: TF-IDF character/word n-grams + Logistic Regression.
Chosen over an LLM call here because:
  - gives clean, reproducible precision/recall/F1 on a held-out test set
  - trains in seconds, no API cost, no rate limits while iterating
  - keeps the "AI reasoning" LLM step reserved for the strategy selector,
    where judgment (not just text classification) is actually needed

Usage:
    python3 classifier.py
"""

import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = "../data/payments.csv"
MODEL_PATH = "failure_classifier.joblib"
METRICS_PATH = "../docs/classifier_metrics.json"


def load_data(path):
    df = pd.read_csv(path)
    # Model input: ONLY the raw failure text. true_failure_category is the
    # label used for training/eval -- never fed in as a feature at inference.
    return df["failure_reason_raw"], df["true_failure_category"]


def train_and_evaluate():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    print(f"\nTest set size: {len(X_test)}")
    print(f"Overall accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model for reuse by the rest of the pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    # Save metrics for the README / dashboard to read
    metrics_out = {
        "test_set_size": len(X_test),
        "overall_accuracy": round(acc, 4),
        "per_class_report": report,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved -> {METRICS_PATH}")

    return pipeline


def classify(pipeline, raw_failure_text: str) -> dict:
    """Classify a single failure message. Used by the rest of the pipeline."""
    pred = pipeline.predict([raw_failure_text])[0]
    proba = pipeline.predict_proba([raw_failure_text])[0]
    classes = pipeline.classes_
    confidence = float(max(proba))
    return {
        "predicted_category": pred,
        "confidence": round(confidence, 3),
        "all_probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }


if __name__ == "__main__":
    model = train_and_evaluate()

    print("\n--- Example predictions ---")
    for example in [
        "DO_NOT_HONOR",
        "Txn declined - low balance",
        "Gateway timeout",
        "This card is no longer valid",
        "some weird unrecognized bank message xyz",
    ]:
        result = classify(model, example)
        print(f"{example!r:45} -> {result['predicted_category']} (confidence {result['confidence']})")
