from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.categories import TicketCategory


class TicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(default="", max_length=300)
    description: str = Field(min_length=1, max_length=5000)

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Ticket description must not be blank.")
        return value

    @field_validator("subject")
    @classmethod
    def trim_subject(cls, value: str) -> str:
        return value.strip()

    def ticket_text(self) -> str:
        return "\n".join(part for part in (self.subject, self.description) if part)


class ModelPrediction(BaseModel):
    category: TicketCategory
    confidence: float = Field(ge=0, le=1)
    model: str
    reasoning: str | None = None


class ClassificationResponse(BaseModel):
    ml_prediction: ModelPrediction
    gemini_prediction: ModelPrediction
    random_forest_prediction: ModelPrediction | None = None
    final_category: TicketCategory
    decision: str
    reason: str
    needs_human_review: bool
    model_agreement: bool


class HealthResponse(BaseModel):
    status: str
    ml_model_loaded: bool
    random_forest_model_loaded: bool
    gemini_mode: str

