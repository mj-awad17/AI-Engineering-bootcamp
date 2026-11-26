import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


load_environment()


def confidence_threshold(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value

