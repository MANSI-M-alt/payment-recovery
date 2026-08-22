"""
Recovery Scorer — AI Payment Recovery Agent

Predicts the probability that a given failed payment will be recovered,
using:
  - the failure classifier's predicted category + confidence (NOT the
    ground-truth category -- this mirrors how the real pipeline runs,
    classifier output feeds into the scorer)
  - customer history features (past payment success rate, past failure
    count, account age, avg days to recovery)
  - payment amount

Approach: Random Forest classifier, chosen over logistic regression here
because recovery likelihood depends on non-linear interactions between
failure type and customer history (e.g. "insufficient_funds" + "high past
success rate" recovers well, but "insufficient_funds" + "many past
failures" doesn't) -- a tree-based model captures that without manual
feature engineering.

Usage:
    python3 recovery_scorer.py
(requires ai/classifier.py to have been run first, so failure_classifier.joblib exists)
"""

import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix,
)

PAYMENTS_PATH = "../data/payments.csv"
HISTORY_PATH = "../data/customer_history.csv"
CLASSIFIER_PATH = "failure_classifier.joblib"
MODEL_PATH = "recovery_scorer.joblib"
METRICS_PATH = "../docs/scorer_metrics.json"


def build_feature_table():
    payments = pd.read_csv(PAYMENTS_PATH)
    history = pd.read_csv(HISTORY_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    # Run the failure classifier to get predicted_category + confidence --
    # the scorer sees what the real pipeline would see, not the ground truth.
    preds = classifier.predict(payments["failure_reason_raw"])
    probas = classifier.predict_proba(payments["failure_reason_raw"])
    confidence = probas.max(axis=1)

    payments = payments.copy()
    payments["predicted_failure_category"] = preds
    payments["classifier_confidence"] = confidence

    df = payments.merge(history, on="customer_id", how="left")

    # Derived feature: customer's historical success rate
    df["past_success_rate"] = df["successful_payments"] / df["total_past_payments"].clip(lower=1)

    feature_cols = [
        "predicted_failure_category", "classifier_confidence", "amount_paise",
        "past_success_rate", "past_failure_count", "avg_days_to_recovery",
        "preferred_channel", "account_age_days",
    ]
    X = df[feature_cols]
    y = df["ground_truth_recovered"].astype(bool)
    return X, y, df


def train_and_evaluate():
    X, y, _ = build_feature_table()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    categorical = ["predicted_failure_category", "preferred_channel"]
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nTest set size: {len(X_test)}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["not_recovered", "recovered"]))

    # Feature importance for transparency in the pitch / README
    feature_names = (
        list(pipeline.named_steps["prep"].transformers_[0][1].get_feature_names_out(categorical))
        + [c for c in X.columns if c not in categorical]
    )
    importances = pipeline.named_steps["clf"].feature_importances_
    importance_ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    print("Top feature importances:")
    for name, imp in importance_ranked[:6]:
        print(f"  {name:35} {imp:.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    metrics_out = {
        "test_set_size": len(X_test),
        "roc_auc": round(float(auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {"labels": ["not_recovered", "recovered"], "matrix": cm},
        "top_feature_importances": [
            {"feature": n, "importance": round(float(i), 4)} for n, i in importance_ranked[:10]
        ],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved -> {METRICS_PATH}")

    return pipeline


def score_payment(pipeline, feature_row: dict) -> dict:
    """Score a single payment. feature_row must match the training feature columns."""
    df = pd.DataFrame([feature_row])
    proba = pipeline.predict_proba(df)[0]
    return {
        "recovery_probability": round(float(proba[1]), 3),
        "predicted_recovered": bool(proba[1] >= 0.5),
    }


if __name__ == "__main__":
    model = train_and_evaluate()

    print("\n--- Example scoring ---")
    example = {
        "predicted_failure_category": "network_error",
        "classifier_confidence": 0.89,
        "amount_paise": 99900,
        "past_success_rate": 0.9,
        "past_failure_count": 1,
        "avg_days_to_recovery": 1.2,
        "preferred_channel": "sms",
        "account_age_days": 400,
    }
    result = score_payment(model, example)
    print(f"Example (network_error, good history): {result}")

    example2 = dict(example, predicted_failure_category="card_expired", past_success_rate=0.3, past_failure_count=6)
    result2 = score_payment(model, example2)
    print(f"Example (card_expired, poor history): {result2}")
