import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from ml.preprocess import clean_dataset
from ml.feature_spec import MODEL_FEATURES


BEST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}


def main() -> None:
    data_path = Path("data/maternal_health.csv")
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    df = clean_dataset(df)

    missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required model features: {missing_features}")

    X = df[MODEL_FEATURES].copy()
    y = df["risk_level"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        **BEST_PARAMS,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    joblib.dump(model, artifact_dir / "model.pkl")
    joblib.dump(model, artifact_dir / "model.joblib")

    metadata = {
        "model_name": "RandomForestClassifier",
        "model_version": "rf_v2",
        "features": MODEL_FEATURES,
        "label_mapping": {"0": "low", "1": "high"},
        "best_params": BEST_PARAMS,
        "metrics": {
            "test_accuracy": round(acc, 6),
            "test_f1_score": round(f1, 6),
            "test_auc": round(auc, 6),
            "cv_mean_accuracy": round(float(cv_scores.mean()), 6),
            "cv_std_accuracy": round(float(cv_scores.std()), 6),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["low", "high"], output_dict=True
        ),
    }

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training selesai.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()