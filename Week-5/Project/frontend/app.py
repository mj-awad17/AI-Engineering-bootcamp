import json
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.getenv("TICKET_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Ticket Classification Assistant",
    page_icon=":material/support_agent:",
    layout="wide",
)
st.title("Ticket intelligence", icon=":material/support_agent:")
st.caption("Classify a support request and compare the ML model predictions.")

with st.sidebar:
    st.header("Workspace", icon=":material/tune:")
    st.caption("AI-powered support ticket triage")
    st.markdown("**Models**")
    st.caption("Calibrated ML · Random Forest")

    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "external"
        / "customer_support_tickets.json"
    )
    if dataset_path.exists():
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset_count = len(payload.get("rows", []))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read the downloaded dataset: {exc}")
        else:
            st.caption(f"Reference dataset: {dataset_count} downloaded tickets")

input_column, result_column = st.columns([0.9, 1.1], gap="large")

with input_column:
    with st.container(border=True):
        st.subheader("Ticket details", icon=":material/description:")
        with st.form("ticket_form"):
            subject = st.text_input(
                "Subject",
                max_chars=300,
                placeholder="e.g. Unable to access my account",
            )
            description = st.text_area(
                "Description",
                height=220,
                max_chars=5000,
                placeholder="Describe the customer's issue...",
            )
            submitted = st.form_submit_button(
                "Classify ticket",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

if submitted:
    if not description.strip():
        st.session_state.pop("classification_result", None)
        st.error("Enter a ticket description.")
    else:
        try:
            with st.spinner("Comparing the models..."):
                response = requests.post(
                    f"{API_URL.rstrip('/')}/classify",
                    json={"subject": subject, "description": description},
                    timeout=90,
                )
                response.raise_for_status()
                st.session_state["classification_result"] = response.json()
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.session_state.pop("classification_result", None)
            st.error(f"Could not classify the ticket: {detail}")

with result_column:
    with st.container(border=True):
        st.subheader("Classification result", icon=":material/analytics:")
        result = st.session_state.get("classification_result")
        if result is None:
            st.caption("Your classification and model comparison will appear here.")
        else:
            st.success(result["final_category"], icon=":material/check_circle:")
            st.caption("Combined classification result")
            if result["needs_human_review"]:
                st.warning("Human review is recommended.")
            else:
                st.info("The hybrid decision passed the configured confidence checks.")

            st.markdown("**Model comparison**")
            model_predictions = [
                ("Calibrated ML", result["ml_prediction"]),
                ("Random Forest", result.get("random_forest_prediction")),
            ]
            cards = st.columns(2)
            for card, (model_label, prediction) in zip(cards, model_predictions):
                with card:
                    if prediction is None:
                        st.metric(model_label, "Not trained")
                        st.caption("Run scripts/train_models.py to build this model.")
                        continue
                    st.metric(
                        model_label,
                        prediction["category"],
                        f"{prediction['confidence']:.1%} confidence",
                        border=True,
                    )
                    st.caption(prediction["model"])
                    if prediction.get("reasoning"):
                        st.write(prediction["reasoning"])

