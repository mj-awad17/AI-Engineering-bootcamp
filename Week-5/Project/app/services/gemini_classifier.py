import json
import logging
import os
import re
from typing import Any

from google import genai
from google.genai.errors import APIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.categories import TicketCategory
from app.core.config import load_environment
from app.prompts.classification import build_prompt
from app.schemas import ModelPrediction

logger = logging.getLogger(__name__)
load_environment()


class GeminiResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: TicketCategory
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1, max_length=500)


class GeminiClassificationError(RuntimeError):
    pass


def _mock_classification(ticket_text: str) -> GeminiResult:
    text = ticket_text.casefold()
    rules = (
        (TicketCategory.SERVICE_OUTAGES, ("outage", "down for everyone", "service unavailable")),
        (TicketCategory.IT_INFRASTRUCTURE, ("vpn", "server", "network", "infrastructure")),
        (TicketCategory.RETURNS_REFUNDS, ("refund", "return", "reimburse")),
        (TicketCategory.BILLING_PAYMENTS, ("invoice", "charged", "payment", "billing")),
        (
            TicketCategory.ACCOUNT_ACCESS,
            ("password", "sign in", "log in", "login", "authentication", "locked account", "access account"),
        ),
        (TicketCategory.SALES_PRE_SALES, ("demo", "pricing", "quote", "purchase plan")),
        (TicketCategory.TECHNICAL_SUPPORT, ("error", "crash", "bug", "not working")),
        (TicketCategory.PRODUCT_SUPPORT, ("product", "feature", "how do i use")),
        (TicketCategory.CUSTOMER_SERVICE, ("complaint", "agent", "representative")),
    )
    for category, terms in rules:
        if any(term in text for term in terms):
            return GeminiResult(
                category=category,
                confidence=0.75,
                reasoning="Local demo rule matched a support-ticket keyword.",
            )
    return GeminiResult(
        category=TicketCategory.GENERAL_INQUIRIES,
        confidence=0.55,
        reasoning="No specific demo rule matched; treated as a general inquiry.",
    )


def _parse_response(response_text: str) -> GeminiResult:
    try:
        return GeminiResult.model_validate_json(response_text)
    except (ValidationError, ValueError):
        # Some providers wrap valid JSON in a Markdown code fence.
        match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", response_text, re.DOTALL)
        if match:
            try:
                return GeminiResult.model_validate(json.loads(match.group(1)))
            except (ValidationError, ValueError):
                pass
        raise GeminiClassificationError(
            "Gemini returned an invalid classification response."
        ) from None


def classify_with_gemini(ticket_text: str, prompt_style: str | None = None) -> ModelPrediction:
    mode = os.getenv("GEMINI_MODE", "mock").strip().lower()
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    if mode == "mock":
        result = _mock_classification(ticket_text)
        return ModelPrediction(
            category=result.category,
            confidence=result.confidence,
            model="local demo rules (not Gemini)",
            reasoning=result.reasoning,
        )
    if mode != "live":
        raise GeminiClassificationError("GEMINI_MODE must be 'mock' or 'live'.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiClassificationError(
            "GEMINI_API_KEY is required when GEMINI_MODE=live."
        )

    style = prompt_style or os.getenv("PROMPT_STYLE", "instruction")
    try:
        prompt = build_prompt(ticket_text, style)
    except ValueError as exc:
        raise GeminiClassificationError(str(exc)) from exc
    try:
        with genai.Client(api_key=api_key) as client:
            response: Any = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GeminiResult,
                },
            )
            response_text = response.text
        if not response_text:
            raise GeminiClassificationError("Gemini returned an empty response.")
        result = _parse_response(response_text)
    except GeminiClassificationError:
        raise
    except APIError as exc:
        logger.warning("Gemini API request failed with HTTP %s.", exc.code)
        if exc.code in (401, 403):
            detail = (
                "Gemini rejected the API key or project access. Check GEMINI_API_KEY "
                "and confirm the key has access to the Gemini API."
            )
        elif exc.code == 404:
            detail = (
                f"Gemini model '{model}' was not found or is unavailable. "
                "Set GEMINI_MODEL to a supported model, such as gemini-3.5-flash-lite."
            )
        elif exc.code == 429:
            detail = (
                "Gemini rate or quota limit reached. Check the API project's quota "
                "and billing, then try again."
            )
        elif exc.code == 400:
            detail = (
                "Gemini rejected the request. Check that GEMINI_MODEL supports "
                "structured JSON responses and that the prompt configuration is valid."
            )
        else:
            detail = (
                f"Gemini returned HTTP {exc.code}. Check the model, API access, "
                "and service status, then try again."
            )
        raise GeminiClassificationError(detail) from None
    except ValueError as exc:
        logger.warning(
            "Gemini request failed while handling the SDK response: %s",
            type(exc).__name__,
        )
        raise GeminiClassificationError(
            "Gemini returned a response the application could not process. "
            "Check the selected model and its structured-response support."
        ) from None

    return ModelPrediction(
        category=result.category,
        confidence=result.confidence,
        model=model,
        reasoning=result.reasoning,
    )

