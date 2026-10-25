

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="AI Industrial Defect Inspector",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background: #081426; color: #eef4ff; }
[data-testid="stSidebar"] { background: #101f35; }
h1, h2, h3 { color: #eef4ff; }
div[data-testid="stMetric"] {
    background: #12243d;
    border: 1px solid #29415f;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏭 AI Industrial Defect Inspector")
st.markdown("**Project By: Mary Johnsilin**")
st.caption("Computer Vision • Industrial Casting Quality Inspection")

MODEL_PATH = Path(__file__).parent / "industrial_defect_model.keras"

if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

@st.cache_resource
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

def predict_image(image):
    if not MODEL_PATH.exists():
        return None, None, None, "Model file not found"

    try:
        model = load_model(str(MODEL_PATH))

        image_rgb = image.convert("RGB").resize((128, 128))
        arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        output = np.asarray(model.predict(arr, verbose=0))

        if output.size != 1:
            return None, None, None, (
                f"Expected one output, but model returned shape {output.shape}"
            )

        raw = float(output.reshape(-1)[0])

        if not np.isfinite(raw):
            return None, None, None, "Invalid model output"

        # This mapping must match the original training labels.
        if raw >= 0.5:
            label = "GOOD"
            confidence = raw * 100
        else:
            label = "DEFECTIVE"
            confidence = (1 - raw) * 100

        return label, confidence, raw, None

    except Exception as e:
        return None, None, None, str(e)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Inspect Product", "Inspection Dashboard", "Model Evaluation"]
)

history = st.session_state.inspection_history

if page == "Home":
    st.header("🏠 Industrial Quality Inspection")

    total = len(history)
    defective = sum(x["Prediction"] == "DEFECTIVE" for x in history)
    good = sum(x["Prediction"] == "GOOD" for x in history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Inspections", total)
    c2.metric("Defective Products", defective)
    c3.metric("OK Products", good)

    st.subheader("Project Features")
    st.markdown("""
    - Industrial casting image upload
    - GOOD / DEFECTIVE classification
    - Model output and prediction score
    - Inspection history
    - CSV report download
    """)

elif page == "Inspect Product":
    st.header("📷 Inspect Industrial Products")

    uploaded_files = st.file_uploader(
        "Upload JPG or PNG images (maximum 3)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) > 3:
        st.warning("Only the first 3 images will be inspected.")
        uploaded_files = uploaded_files[:3]

    if uploaded_files:
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            st.divider()
            st.subheader(f"Product {index}")

            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_column_width=True
                )

                if st.button(
                    f"🔍 Inspect Product {index}",
                    key=f"inspect_{index}",
                    type="primary"
                ):
                    label, confidence, raw, error = predict_image(image)

                    if error:
                        st.error("Prediction failed")
                        st.code(error)
                    else:
                        st.write(f"Model output: {raw:.6f}")
                        st.metric("Prediction Score", f"{confidence:.2f}%")

                        if label == "DEFECTIVE":
                            st.error(f"⚠️ DEFECTIVE — {confidence:.2f}%")
                        else:
                            st.success(f"✅ GOOD — {confidence:.2f}%")

                        st.warning(
                            "Confirm the training label mapping and "
                            "preprocessing before relying on this prediction."
                        )

                        record_key = f"saved_{uploaded_file.name}_{index}"

                        if record_key not in st.session_state:
                            st.session_state.inspection_history.append({
                                "Image Name": uploaded_file.name,
                                "Prediction": label,
                                "Confidence (%)": round(confidence, 2),
                                "Raw Output": round(raw, 6),
                                "Inspection Time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            })
                            st.session_state[record_key] = True

            except Exception as e:
                st.error(f"Could not open image: {e}")
    else:
        st.info("Upload an image to start inspection.")

elif page == "Inspection Dashboard":
    st.header("📊 Inspection Dashboard")

    history = st.session_state.inspection_history
    total = len(history)
    defective = sum(x["Prediction"] == "DEFECTIVE" for x in history)
    good = sum(x["Prediction"] == "GOOD" for x in history)

    c1, c2, c3 = st.columns(3)
    c1.metric("📦 Total Inspections", total)
    c2.metric("🚨 Defective Products", defective)
    c3.metric("✅ OK Products", good)

    if history:
        history_df = pd.DataFrame(history)

        st.subheader("📈 Product Status")
        status_df = pd.DataFrame({
            "Product Status": ["Defective", "OK"],
            "Count": [defective, good]
        }).set_index("Product Status")
        st.bar_chart(status_df)

        st.subheader("📋 Recent Inspection History")
        st.dataframe(
            history_df.iloc[::-1].head(10).reset_index(drop=True),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Inspection Report (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="inspection_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No inspection history yet.")

elif page == "Model Evaluation":
    st.header("📈 Model Evaluation")
    st.warning(
        "The following are previously reported values, not recalculated "
        "by this app. Verify them against the current model."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Reported Accuracy", "99.72%")
    c2.metric("Reported Precision", "99.24%")
    c3.metric("Reported Recall", "100.00%")

    st.subheader("Previously Reported Confusion Matrix")
    st.markdown("""
    | Actual class | Predicted Defective | Predicted Good |
    |---|---:|---:|
    | Defective | 451 | 2 |
    | Good | 0 | 262 |
    """)

    st.caption("Historical metrics are not recalculated here.")

st.divider()
st.caption(
    "AI Industrial Defect Inspector • Industrial Casting Quality Inspection"
)
st.caption(
    "Automatic visual defect highlighting is not implemented in this version."
)
