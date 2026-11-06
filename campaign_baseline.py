"""Evaluate a campaign on a separate labeled ARFF customer dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["age", "income", "children"]
CATEGORICAL = ["sex", "region", "married", "car", "save_act", "current_act", "mortgage"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "pep"


def load_data(path):
    records, _ = arff.loadarff(path)
    frame = pd.DataFrame(records)
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].map(
            lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
        )
    return frame


def validate(frame):
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The dataset must contain at least one row.")
    if frame[TARGET].isna().any() or not set(frame[TARGET]).issubset({"YES", "NO"}):
        raise ValueError("pep must contain only YES or NO.")
    data = frame[FEATURES].copy()
    data[NUMERIC] = data[NUMERIC].apply(pd.to_numeric, errors="raise")
    if np.isinf(data[NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("Numeric features must not be infinite.")
    return data, (frame[TARGET] == "YES").astype(int)


def evaluate(train, test, benefit=100.0, loss=70.0):
    if not np.isfinite(benefit) or not np.isfinite(loss) or benefit <= 0 or loss <= 0:
        raise ValueError("Benefit and loss must be finite positive numbers.")
    X_train, y_train = validate(train)
    X_test, y_test = validate(test)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Training and test data must each contain both classes.")
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                      OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    model = make_pipeline(transform, LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = loss / (benefit + loss)
    selected = probabilities > threshold
    positives = int(y_test[selected].sum())
    targeted = int(selected.sum())
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "targeted": targeted,
        "positive_targets": positives,
        "observed_profit": float(positives * benefit - (targeted - positives) * loss),
        "contact_all_profit": float(y_test.sum() * benefit - (len(y_test) - y_test.sum()) * loss),
        "contact_none_profit": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--benefit", type=float, default=100)
    parser.add_argument("--loss", type=float, default=70)
    args = parser.parse_args()
    try:
        result = evaluate(load_data(args.training), load_data(args.test), args.benefit, args.loss)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
