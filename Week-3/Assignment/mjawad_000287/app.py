from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="centered",
)

MODEL_FILES = {
    "Logistic Regression": "Student-Log_R_model.joblib",
    "Random Forest": "Student-rf_model.joblib",
}


@st.cache_resource
def load_model(filename: str):
    model_path = Path(__file__).resolve().parent / filename
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    return joblib.load(model_path)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(80, 120, 255, 0.18), transparent 38%),
            linear-gradient(145deg, #0b1220 0%, #111b2d 52%, #101827 100%);
        color: #f3f6fc;
    }
    [data-testid="stHeader"] { background: transparent; }
    .hero {
        padding: 2rem 0 1rem;
        text-align: center;
    }
    .hero h1 {
        color: #f5f8ff;
        font-size: clamp(2rem, 5vw, 3rem);
        letter-spacing: -0.04em;
        margin-bottom: 0.45rem;
    }
    .hero p {
        color: #aab8cf;
        font-size: 1.05rem;
        margin: 0;
    }
    [data-testid="stForm"] {
        background: rgba(25, 37, 57, 0.82);
        border: 1px solid rgba(164, 185, 218, 0.18);
        border-radius: 20px;
        padding: 1.4rem 1.5rem 1.5rem;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stWidgetLabel"] p {
        color: #dce5f4;
        font-weight: 600;
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #5575f7, #7958e8);
        color: white;
        font-weight: 700;
        min-height: 3rem;
        transition: transform 160ms ease, filter 160ms ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.12);
        transform: translateY(-2px);
        color: white;
    }
    .prediction {
        animation: prediction-enter 550ms cubic-bezier(.2, .8, .2, 1) both;
        background: rgba(25, 37, 57, 0.9);
        border: 1px solid rgba(164, 185, 218, 0.2);
        border-radius: 20px;
        color: #f6f8ff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 1.5rem auto 0;
        max-width: 22rem;
        padding: 1.2rem;
        text-align: center;
    }
    @keyframes prediction-enter {
        from { opacity: 0; transform: translateY(14px) scale(.94); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Student Success Predictor</h1>
        <p>Enter the student details, choose a model, and get a prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("student_prediction_form"):
    st.subheader("Student details")
    first_row_left, first_row_right = st.columns(2)
    with first_row_left:
        hours_studied = st.number_input(
            "Hours studied",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
        )
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )
        practice_tests = st.number_input(
            "Practice tests",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )
    with first_row_right:
        sleep_hours = st.number_input(
            "Sleep (hours)",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
        )
        study_method = st.selectbox(
            "Study method",
            options=["group", "self", "video"],
            format_func=str.title,
        )
        selected_model = st.selectbox("Prediction model", options=list(MODEL_FILES))

    submitted = st.form_submit_button("Predict")

if submitted:
    model = load_model(MODEL_FILES[selected_model])
    student = pd.DataFrame(
        [
            {
                "hours_studied": hours_studied,
                "attendance": attendance,
                "practice_tests": practice_tests,
                "sleep_hours": sleep_hours,
                "study_method": study_method,
            }
        ],
        columns=[
            "hours_studied",
            "attendance",
            "practice_tests",
            "sleep_hours",
            "study_method",
        ],
    )
    prediction = model.predict(student)[0]
    result = "PASS" if prediction == 1 else "FAIL"

    st.markdown(
        f'<div class="prediction">{result}</div>',
        unsafe_allow_html=True,
    )
    if prediction == 1:
        st.balloons()
    else:
        st.snow()
