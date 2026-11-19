"""Streamlit app:  streamlit run app.py
Loads saved CNN + ViT (no training here). Upload PNG/JPEG or draw a digit."""
import os

import streamlit as st
from PIL import Image

from common import ImageRejected, load_checkpoint, predict, prepare_image

st.set_page_config(page_title="mj.digitStudio", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def get_models():
    out = {}
    for name in ("cnn", "vit"):
        path = os.path.join(HERE, "models", f"{name}.pt")
        if not os.path.exists(path):
            return None
        out[name], _ = load_checkpoint(path)
    return out


models = get_models()
if models is None:
    st.error("Saved models not found in `models/`. Run `python train.py` first.")
    st.stop()


def run(src):
    """Prepare + predict + display. Never raises to the user."""
    try:
        arr, original, _ = prepare_image(src, with_steps=True)
    except ImageRejected as e:
        st.warning(f"Rejected: {e}")
        return
    except Exception:                       # last-resort guard: never crash
        st.warning("Rejected: could not process this image.")
        return
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        with st.container(border=True, height=620):
            st.subheader("Your input")
            st.image(original, use_container_width=True)
    with c2:
        with st.container(border=True, height=310):
            st.subheader("Predictions")
            for name, label in (("cnn", "CNN"), ("vit", "ViT")):
                digit, probs = predict(models[name], arr)
                confidence = probs[digit] * 100
                st.metric(label, f"{digit}", f"{confidence:.1f}% confidence")


with st.sidebar:
    st.markdown("**Digit studio**")
    st.caption("Recognize a handwritten digit with the\ntrained CNN and ViT models.")
    st.markdown("**Choose an input**")
    mode = st.segmented_control(
        "Input mode",
        ["Upload image", "Draw digit"],
        selection_mode="single",
        default="Upload image",
        label_visibility="collapsed",
    )
    st.caption("Upload one digit")

st.title("Handwritten digit recognition")
st.caption("Upload a digit or draw one, then compare the CNN and ViT predictions.")

if mode == "Upload image":
    f = st.file_uploader(
        "Upload digit",
        type=["png", "jpg", "jpeg"],
        help="PNG or JPEG, 200 MB per file.",
        label_visibility="collapsed",
    )
    if f is None:
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            with st.container(border=True, height=310):
                st.subheader("Your input")
                st.caption("Choose an image from the sidebar to get started.")
        with c2:
            with st.container(border=True, height=310):
                st.subheader("Predictions")
                st.caption("Your CNN and ViT predictions will appear here.")
    else:
        run(f.getvalue())
else:
    try:
        from streamlit_drawable_canvas import st_canvas
        have_canvas = True
    except Exception:
        have_canvas = False
        st.error("Drawing board unavailable. Run `pip install streamlit-drawable-canvas`.")
    if have_canvas:
        width = st.slider("Brush width", 8, 40, 22)
        kw = dict(
            stroke_width=width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True,
        )
        try:
            res = st_canvas(return_image_data=True, **kw)
        except TypeError:
            res = st_canvas(**kw)
        if res.image_data is not None:
            img = Image.fromarray(res.image_data.astype("uint8"))
            run(img)
