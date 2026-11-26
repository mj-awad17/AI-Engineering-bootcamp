import os

from app.core.config import confidence_threshold
from app.schemas import ClassificationResponse, ModelPrediction


def combine_predictions(
    ml_prediction: ModelPrediction,
    gemini_prediction: ModelPrediction,
    ml_threshold: float | None = None,
    gemini_threshold: float | None = None,
) -> ClassificationResponse:
    ml_threshold = ml_threshold if ml_threshold is not None else confidence_threshold(
        "ML_CONFIDENCE_THRESHOLD", "0.65"
    )
    gemini_threshold = (
        gemini_threshold
        if gemini_threshold is not None
        else confidence_threshold("GEMINI_CONFIDENCE_THRESHOLD", "0.65")
    )
    if not 0 <= ml_threshold <= 1 or not 0 <= gemini_threshold <= 1:
        raise ValueError("Confidence thresholds must be between 0 and 1.")
    agreement = ml_prediction.category == gemini_prediction.category

    if agreement:
        final_category = ml_prediction.category
        decision = "accepted_agreement"
        needs_review = (
            ml_prediction.confidence < ml_threshold
            or gemini_prediction.confidence < gemini_threshold
        )
        reason = "ML and Gemini agree on the category."
        if needs_review:
            reason += " At least one confidence score is below its review threshold."
    elif ml_prediction.confidence >= ml_threshold and ml_prediction.confidence >= gemini_prediction.confidence:
        final_category = ml_prediction.category
        decision = "accepted_ml_confidence"
        needs_review = True
        reason = "Models disagree; ML has the stronger confidence, so human review is recommended."
    elif gemini_prediction.confidence >= gemini_threshold:
        final_category = gemini_prediction.category
        decision = "accepted_gemini_confidence"
        needs_review = True
        reason = "Models disagree; Gemini has sufficient confidence, but human review is recommended."
    else:
        final_category = ml_prediction.category
        decision = "low_confidence"
        needs_review = True
        reason = "Models disagree and neither confidence meets its threshold; ML is shown as a provisional result."

    return ClassificationResponse(
        ml_prediction=ml_prediction,
        gemini_prediction=gemini_prediction,
        final_category=final_category,
        decision=decision,
        reason=reason,
        needs_human_review=needs_review,
        model_agreement=agreement,
    )

