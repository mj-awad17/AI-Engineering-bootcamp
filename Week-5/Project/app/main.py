from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status

from app.core.config import PROJECT_ROOT
from app.schemas import ClassificationResponse, HealthResponse, TicketRequest
from app.services.gemini_classifier import (
    GeminiClassificationError,
    classify_with_gemini,
)
from app.services.hybrid_classifier import combine_predictions
from app.services.ml_classifier import MLClassifier, ModelNotTrainedError

MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"
RANDOM_FOREST_MODEL_PATH = PROJECT_ROOT / "models" / "ticket_random_forest.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = MLClassifier(MODEL_PATH)
    random_forest_classifier = MLClassifier(RANDOM_FOREST_MODEL_PATH)
    try:
        classifier.load()
    except ModelNotTrainedError:
        pass
    try:
        random_forest_classifier.load()
    except ModelNotTrainedError:
        pass
    app.state.ml_classifier = classifier
    app.state.random_forest_classifier = random_forest_classifier
    yield


app = FastAPI(
    title="AI-Powered Ticket Classification Assistant",
    description="Hybrid ML and Google Gemini support-ticket classification API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    classifier: MLClassifier = request.app.state.ml_classifier
    random_forest_classifier: MLClassifier = request.app.state.random_forest_classifier
    return HealthResponse(
        status="ready" if classifier.pipeline is not None else "model_not_trained",
        ml_model_loaded=classifier.pipeline is not None,
        random_forest_model_loaded=random_forest_classifier.pipeline is not None,
        gemini_mode=os.getenv("GEMINI_MODE", "mock"),
    )


@app.post("/classify", response_model=ClassificationResponse)
def classify(ticket: TicketRequest, request: Request) -> ClassificationResponse:
    classifier: MLClassifier = request.app.state.ml_classifier
    try:
        ml_prediction = classifier.predict(ticket.ticket_text())
    except ModelNotTrainedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    try:
        gemini_prediction = classify_with_gemini(ticket.ticket_text())
    except GeminiClassificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    random_forest_classifier: MLClassifier = request.app.state.random_forest_classifier
    try:
        random_forest_prediction = random_forest_classifier.predict(ticket.ticket_text())
    except ModelNotTrainedError:
        random_forest_prediction = None

    response = combine_predictions(ml_prediction, gemini_prediction)
    response.random_forest_prediction = random_forest_prediction
    return response

