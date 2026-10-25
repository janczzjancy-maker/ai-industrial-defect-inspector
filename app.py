
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

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: #081426;
    color: #eef4ff;
}
[data-testid="stSidebar"] {
    background: #101f35;
}
h1, h2, h3 {
    color: #eef4ff;
}
div[data-testid="stMetric"] {
    background: #12243d;
    border: 1px solid #29415f;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🏭 AI Industrial Defect Inspector")
st.markdown("**Project By: Mary Johnsilin**")
st.caption("Computer Vision • Industrial Casting Quality Inspection")

MODEL_PATH = Path(__file__).parent / "industrial_defect_model.keras"

# ---------------- HISTORY ----------------
if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

# ---------------- MODEL ----------------
@st.cache_resource
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def predict_image(image):
    if not MODEL_PATH.exists():
        return None, None, None, "Model file not found"

    try:
        model = load_model(str(MODEL_PATH))

        image_rgb = image.convert("RGB").resize((128, 128))
        arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        raw = float(
            np.asarray(model.predict(arr, verbose=0)).reshape(-1)[0]
        )

        # Confirm label mapping using the training class names.
        if raw >= 0.5:
            label = "GOOD"
            confidence = raw * 100
        else:
            label = "DEFECTIVE"
            confidence = (1 - raw) * 100

        return label, confidence, raw, None

    except Exception as e:
        return None, None, None, str(e)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Inspect Product", "Inspection Dashboard", "Model Evaluation"]
)

history = st.session_state.inspection_history

# ---------------- HOME ----------------
if page == "Home":
    st.header("Welcome to Industrial Quality Inspection")

    total = len(history)
    defective = sum(x["Prediction"] == "DEFECTIVE" for x in history)
    good = sum(x["Prediction"] == "GOOD" for x in history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Inspections", total)
    c2.metric("Defective Products", defective)
    c3.metric("OK Products", good)

    st.markdown("""
    ### Project Features
    - Upload industrial casting images
    - AI prediction: GOOD or DEFECTIVE
    - Prediction score
    - Inspection history
    - Downloadable CSV report
    """)

# ---------------- INSPECT PRODUCT ----------------
elif page == "Inspect Product":
    st.header("📷 Inspect Industrial Products")

    uploaded_files = st.file_uploader(
        "Choose up to 3 JPG or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) > 3:
        st.warning("Maximum 3 images allowed.")
        uploaded_files = uploaded_files[:3]

    if uploaded_files:
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            st.divider()
            st.subheader(f"Product {index}")

            try:
                image = Image.open(uploaded_file).convert("RGB")

                # Compatible with older Streamlit versions
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
                        st.error("AI prediction could not be completed.")
                        with st.expander("Technical details"):
                            st.code(error)

                    else:
                        st.write(f"Raw model output: {raw:.6f}")

                        if label == "GOOD":
                            st.success(
                                f"✅ GOOD — {confidence:.2f}% model score"
                            )
                        else:
                            st.error(
                                f"⚠️ DEFECTIVE — {confidence:.2f}% model score"
                            )

                        st.metric("Prediction Score", f"{confidence:.2f}%")

                        # Save inspection result to session history
                        st.session_state.inspection_history.append({
                            "Image Name": uploaded_file.name,
                            "Prediction": label,
                            "Confidence (%)": round(confidence, 2),
                            "Raw Output": round(raw, 6),
                            "Inspection Time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        })

                        st.caption(
                            "The label mapping must match the model's "
                            "training labels. The score may not be a "
                            "calibrated probability."
                        )

            except Exception as e:
                st.error(f"Could not open this image: {e}")
    else:
        st.info("Upload images to start inspection.")

# ---------------- DASHBOARD ----------------
elif page == "Inspection Dashboard":
    st.header("📊 Inspection Dashboard")

    history = st.session_state.inspection_history

    total = len(history)
    defective = sum(x["Prediction"] == "DEFECTIVE" for x in history)
    good = sum(x["Prediction"] == "GOOD" for x in history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Inspections", total)
    c2.metric("Defective Products", defective)
    c3.metric("OK Products", good)

    if history:
        st.subheader("📈 Product Status")

        chart_data = pd.DataFrame({
            "Product Status": ["Defective", "OK"],
            "Count": [defective, good]
        }).set_index("Product Status")

        st.bar_chart(chart_data)

        st.subheader("📋 Inspection History")
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Inspection Report (CSV)",
            data=csv_data,
            file_name="inspection_report.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "No inspection history yet. Go to Inspect Product "
            "and inspect an image first."
        )

# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation")
    st.write("Previously evaluated test dataset: 715 images")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "99.72%")
    c2.metric("Precision", "99.24%")
    c3.metric("Recall", "100.00%")

    st.subheader("Confusion Matrix")

    st.markdown("""
    | Actual class | Predicted Defective | Predicted Good |
    |---|---:|---:|
    | Defective | 451 | 2 |
    | Good | 0 | 262 |
    """)

    st.caption(
        "These metrics are from previously reported test results. "
        "They are not recalculated by this dashboard."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "AI Industrial Defect Inspector • Industrial Casting Quality Inspection"
)
st.caption(
    "Visual defect highlighting is not implemented in this version."
)
