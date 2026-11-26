import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.core.categories import CATEGORY_NAMES

DATA_PATH = ROOT / "data" / "processed" / "tickets.csv"
MODEL_PATH = ROOT / "models" / "ticket_classifier.joblib"
RANDOM_FOREST_MODEL_PATH = ROOT / "models" / "ticket_random_forest.joblib"
REPORT_PATH = ROOT / "evaluation" / "outputs" / "ml_metrics.json"
CONFUSION_MATRIX_PATH = ROOT / "evaluation" / "outputs" / "confusion_matrix.png"


def build_candidates() -> dict[str, object]:
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
    }
    return {
        name: Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=estimator,
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        )
        for name, estimator in estimators.items()
    }


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(data_path: Path = DATA_PATH) -> dict:
    frame = pd.read_csv(data_path)
    required = {"ticket_text", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Training CSV must contain columns: {', '.join(sorted(required))}")

    frame = frame.dropna(subset=["ticket_text", "category"])
    frame["ticket_text"] = frame["ticket_text"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip()
    frame = frame[frame["ticket_text"].str.len() > 0]
    unknown = sorted(set(frame["category"]) - set(CATEGORY_NAMES))
    if unknown:
        raise ValueError("Unsupported ticket categories: " + ", ".join(unknown))

    counts = frame["category"].value_counts()
    missing = sorted(set(CATEGORY_NAMES) - set(counts.index))
    if missing:
        raise ValueError("Training data is missing categories: " + ", ".join(missing))
    if int(counts.min()) < 4:
        raise ValueError("Each category needs at least four tickets for calibration and validation.")

    x_train, x_valid, y_train, y_valid = train_test_split(
        frame["ticket_text"],
        frame["category"],
        test_size=0.25,
        random_state=42,
        stratify=frame["category"],
    )

    candidates = build_candidates()
    scores = {}
    best_name = ""
    best_score = -1.0
    best_pipeline = None
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_valid)
        macro_f1 = f1_score(y_valid, prediction, average="macro", zero_division=0)
        scores[name] = macro_f1
        if macro_f1 > best_score:
            best_name, best_score, best_pipeline = name, macro_f1, pipeline

    assert best_pipeline is not None
    random_forest_validation = build_random_forest_pipeline()
    random_forest_validation.fit(x_train, y_train)
    random_forest_valid_prediction = random_forest_validation.predict(x_valid)
    scores["Random Forest"] = f1_score(
        y_valid,
        random_forest_valid_prediction,
        average="macro",
        zero_division=0,
    )

    best_pipeline.fit(frame["ticket_text"], frame["category"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "validation_macro_f1": best_score,
            "validation_scores": scores,
            "categories": CATEGORY_NAMES,
        },
        MODEL_PATH,
    )

    random_forest = build_random_forest_pipeline()
    random_forest.fit(frame["ticket_text"], frame["category"])
    joblib.dump(
        {
            "pipeline": random_forest,
            "model_name": "Random Forest",
            "categories": CATEGORY_NAMES,
        },
        RANDOM_FOREST_MODEL_PATH,
    )

    best_valid = candidates[best_name]
    best_valid.fit(x_train, y_train)
    valid_prediction = best_valid.predict(x_valid)
    report = {
        "selected_model": best_name,
        "validation_macro_f1": best_score,
        "validation_scores_macro_f1": scores,
        "validation_metrics": {
            "accuracy": accuracy_score(y_valid, valid_prediction),
            "precision_macro": precision_score(y_valid, valid_prediction, average="macro", zero_division=0),
            "recall_macro": recall_score(y_valid, valid_prediction, average="macro", zero_division=0),
            "f1_macro": f1_score(y_valid, valid_prediction, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_valid, valid_prediction, average="weighted", zero_division=0),
            "classification_report": classification_report(
                y_valid,
                valid_prediction,
                labels=CATEGORY_NAMES,
                output_dict=True,
                zero_division=0,
            ),
        },
        "validation_rows": len(y_valid),
        "training_rows": len(y_train),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_confusion_matrix(y_valid, valid_prediction)
    return report


def _save_confusion_matrix(actual, predicted) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = confusion_matrix(actual, predicted, labels=CATEGORY_NAMES)
    figure, axis = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CATEGORY_NAMES,
        yticklabels=CATEGORY_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("True category")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    results = train()
    print(json.dumps({key: value for key, value in results.items() if key != "validation_metrics"}, indent=2))

