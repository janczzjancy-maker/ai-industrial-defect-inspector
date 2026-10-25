
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime

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
        return tf.keras.models.load_model(str(MODEL_PATH))
    return None

try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

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
        return None, None, (
            model_error or
            "Model file not found. Add industrial_defect_model.keras."
        )

    try:
        image = image.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        # Model contains Rescaling(1./255).
        raw = float(model.predict(arr, verbose=0).reshape(-1)[0])

        if not 0 <= raw <= 1:
            return None, None, f"Unexpected model output: {raw}"

        # This assumes sigmoid output 0 = DEFECTIVE, 1 = GOOD.
        if raw >= 0.5:
            label = "GOOD"
            confidence = raw * 100
        else:
            label = "DEFECTIVE"
            confidence = (1 - raw) * 100

        return label, confidence, None

    except Exception as e:
        return None, None, str(e)


if page == "Inspect Product":
    st.subheader("📷 Upload Product Images")

    files = st.file_uploader(
        "Choose JPG, JPEG or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files and st.button("🔍 Inspect Products", type="primary"):
        for file in files:
            try:
                image = Image.open(file).convert("RGB")
                label, confidence, error = predict_image(image)

                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    # Compatible with older Streamlit versions
                    st.image(
                        image,
                        caption=file.name,
                        use_column_width=True
                    )

                with col2:
                    if error:
                        st.error(f"Prediction failed: {error}")
                        continue

                    if label == "DEFECTIVE":
                        st.error("⚠️ DEFECTIVE")
                    else:
                        st.success("✅ GOOD")

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "AI prediction only. Verify results before "
                        "making industrial quality decisions."
                    )

                    st.session_state.history.append({
                        "Time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Image": file.name,
                        "Result": label,
                        "Confidence (%)": round(confidence, 2)
                    })

            except Exception as e:
                st.error(f"Could not process {file.name}: {e}")


elif page == "Dashboard & Reports":
    st.subheader("📊 Inspection Dashboard")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Inspections", len(df))
        c2.metric(
            "Defective",
            int((df["Result"] == "DEFECTIVE").sum())
        )
        c3.metric(
            "Good",
            int((df["Result"] == "GOOD").sum())
        )

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
        "images as GOOD or DEFECTIVE using a TensorFlow model."
    )
    st.write("Team Members: Sri Vishwa and Mary Johnsilin")


if model is None:
    st.warning(
        "AI model is unavailable. Check that "
        "industrial_defect_model.keras exists in the repository "
        "and that its dependencies are installed."
    )
