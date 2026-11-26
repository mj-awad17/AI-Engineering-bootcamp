import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.core.categories import CATEGORY_NAMES, TicketCategory
from app.schemas import ModelPrediction
from app.services.gemini_classifier import classify_with_gemini
from app.services.hybrid_classifier import combine_predictions
from scripts.train_models import DATA_PATH, build_candidates

OUTPUT_DIR = ROOT / "evaluation" / "outputs"


def metric_summary(actual, predicted) -> dict:
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision_macro": precision_score(actual, predicted, average="macro", zero_division=0),
        "recall_macro": recall_score(actual, predicted, average="macro", zero_division=0),
        "f1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
        "f1_weighted": f1_score(actual, predicted, average="weighted", zero_division=0),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=CATEGORY_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(actual, predicted, name: str) -> None:
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
    figure.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(figure)


def evaluate(
    data_path: Path = DATA_PATH,
    use_live_gemini: bool = False,
    limit: int | None = None,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    required = {"ticket_text", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Evaluation CSV must contain columns: {', '.join(sorted(required))}")
    if not set(CATEGORY_NAMES).issubset(set(frame["category"].astype(str))):
        raise ValueError("Evaluation data must contain all ten supported categories.")
    if frame["category"].value_counts().min() < 5:
        raise ValueError("Evaluation data needs at least five tickets per category.")

    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        frame["ticket_text"].astype(str),
        frame["category"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=frame["category"],
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=43,
        stratify=y_train_validation,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("The evaluation limit must be at least one.")
        x_test = x_test.iloc[:limit]
        y_test = y_test.loc[x_test.index]

    candidates = build_candidates()
    validation_scores = {}
    for model_name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        validation_prediction = pipeline.predict(x_validation)
        validation_scores[model_name] = f1_score(
            y_validation,
            validation_prediction,
            average="macro",
            zero_division=0,
        )
    selected_ml_name = max(validation_scores, key=validation_scores.get)

    x_train_final = pd.concat([x_train, x_validation])
    y_train_final = pd.concat([y_train, y_validation])
    ml_predictions = {}
    for model_name, pipeline in build_candidates().items():
        pipeline.fit(x_train_final, y_train_final)
        ml_predictions[model_name] = [
            ModelPrediction(
                category=TicketCategory(pipeline.classes_[int(probabilities.argmax())]),
                confidence=float(probabilities.max()),
                model=model_name,
            )
            for probabilities in pipeline.predict_proba(x_test)
        ]

    os.environ["GEMINI_MODE"] = "live" if use_live_gemini else "mock"
    gemini_predictions = [
        classify_with_gemini(text)
        for text in x_test.tolist()
    ]
    results = {}
    for model_name, predictions in ml_predictions.items():
        categories = [prediction.category.value for prediction in predictions]
        results[model_name] = metric_summary(y_test.tolist(), categories)
        save_confusion_matrix(y_test.tolist(), categories, model_name.lower().replace(" ", "_"))

    selected_ml_predictions = ml_predictions[selected_ml_name]
    hybrid_predictions = [
        combine_predictions(ml_prediction, gemini_prediction).final_category.value
        for ml_prediction, gemini_prediction in zip(
            selected_ml_predictions, gemini_predictions
        )
    ]
    gemini_categories = [prediction.category.value for prediction in gemini_predictions]
    results["Gemini"] = metric_summary(y_test.tolist(), gemini_categories)
    results["Hybrid"] = metric_summary(y_test.tolist(), hybrid_predictions)
    save_confusion_matrix(y_test.tolist(), gemini_categories, "gemini")
    save_confusion_matrix(y_test.tolist(), hybrid_predictions, "hybrid")

    report = {
        "rows_evaluated": len(y_test),
        "selected_ml_model_for_hybrid": selected_ml_name,
        "selection_validation_macro_f1": validation_scores[selected_ml_name],
        "gemini_mode": "live" if use_live_gemini else "mock",
        "metrics": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "hybrid_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate each ML candidate, Gemini, and their hybrid decision."
    )
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--live-gemini",
        action="store_true",
        help="Send held-out ticket text to Gemini; this may incur API charges.",
    )
    args = parser.parse_args()
    report = evaluate(args.input, args.live_gemini, args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

