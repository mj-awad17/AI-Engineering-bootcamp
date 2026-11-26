from pathlib import Path
from typing import Any

import joblib

from app.core.categories import TicketCategory
from app.schemas import ModelPrediction


class ModelNotTrainedError(RuntimeError):
    pass


class MLClassifier:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.pipeline: Any | None = None
        self.model_name: str | None = None

    def load(self) -> None:
        if not self.artifact_path.is_file():
            raise ModelNotTrainedError(
                f"Trained model not found at {self.artifact_path}. "
                "Run python scripts/train_models.py first."
            )
        artifact = joblib.load(self.artifact_path)
        self.pipeline = artifact["pipeline"]
        self.model_name = artifact["model_name"]

    def predict(self, ticket_text: str) -> ModelPrediction:
        if self.pipeline is None or self.model_name is None:
            raise ModelNotTrainedError("The ML model has not been loaded.")

        probabilities = self.pipeline.predict_proba([ticket_text])[0]
        classes = self.pipeline.classes_
        best_index = int(probabilities.argmax())
        category = TicketCategory(classes[best_index])
        return ModelPrediction(
            category=category,
            confidence=float(probabilities[best_index]),
            model=self.model_name,
            reasoning=(
                f"Selected as the highest-probability category by the "
                f"{self.model_name} classifier."
            ),
        )

