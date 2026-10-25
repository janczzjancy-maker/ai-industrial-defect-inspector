
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
import io

st.set_page_config(
    page_title="AI Industrial Defect Inspector",
    page_icon="🏭",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "industrial_defect_model.keras"
IMG_SIZE = (128, 128)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🏭 AI Industrial Defect Inspector")
st.caption("Industrial Casting Quality Inspection")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Page",
    ["Inspect Product", "Dashboard & Reports", "About"]
)

def predict_image(image):
    if model is None:
        return None, None, "Model file not found"

    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # Model includes Rescaling(1./255), so do not divide here.
    raw = float(model.predict(arr, verbose=0).reshape(-1)[0])

    # Training class order: def_front = 0, ok_front = 1.
    # Sigmoid output corresponds to class index 1.
    if raw >= 0.5:
        label = "GOOD"
        confidence = raw * 100
    else:
        label = "DEFECTIVE"
        confidence = (1 - raw) * 100

    return label, confidence, None

if page == "Inspect Product":
    st.subheader("📷 Upload Product Images")

    files = st.file_uploader(
        "Choose JPG, JPEG or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files and st.button("🔍 Inspect Products", type="primary"):
        for file in files:
            image = Image.open(file).convert("RGB")
            label, confidence, error = predict_image(image)

            st.divider()
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption=file.name, use_container_width=True)

            with col2:
                if error:
                    st.error(error)
                    continue

                if label == "DEFECTIVE":
                    st.error("⚠️ DEFECTIVE")
                else:
                    st.success("✅ GOOD")

                st.metric("Model Confidence", f"{confidence:.2f}%")
                st.caption(
                    "Prediction is based on the trained model. "
                    "Please verify results before making quality decisions."
                )

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Image": file.name,
                    "Result": label,
                    "Confidence (%)": round(confidence, 2)
                })

elif page == "Dashboard & Reports":
    st.subheader("📊 Inspection Dashboard")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Inspections", len(df))
        c2.metric("Defective", int((df["Result"] == "DEFECTIVE").sum()))
        c3.metric("Good", int((df["Result"] == "GOOD").sum()))

        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Inspection Report",
            data=csv_data,
            file_name="inspection_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No inspections yet. Inspect a product first.")

elif page == "About":
    st.subheader("About This Project")
    st.write(
        "AI Industrial Defect Inspector classifies industrial casting "
        "images as GOOD or DEFECTIVE using a trained TensorFlow model."
    )
    st.write("Team Members: Sri Vishwa and Mary Johnsilin")

if model is None:
    st.warning(
        "AI model not found. Add industrial_defect_model.keras "
        "to the GitHub repository before using predictions."
    )
