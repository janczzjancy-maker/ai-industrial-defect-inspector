

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="AI Industrial Defect Inspector",
    page_icon="🏭",
    layout="wide"
)

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

st.title("🏭 AI Industrial Defect Inspector")
st.caption("Computer Vision • Industrial Casting Quality Inspection")

MODEL_PATH = Path(__file__).parent / "industrial_defect_model.keras"

st.info(
    "Upload up to 3 product images. "
    "AI predictions require the trained model file."
)


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

        # Show the raw model output to help verify label mapping.
        if raw >= 0.5:
            label = "GOOD"
            confidence = raw * 100
        else:
            label = "DEFECTIVE"
            confidence = (1 - raw) * 100

        return label, confidence, raw, None

    except Exception as e:
        return None, None, None, str(e)


left, right = st.columns(2)

with left:
    st.subheader("1. Upload Product Images")

    uploaded_files = st.file_uploader(
        "Choose up to 3 JPG or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    st.caption("Select up to 3 industrial casting images.")

    if uploaded_files and len(uploaded_files) > 3:
        st.warning("Maximum 3 images allowed.")
        uploaded_files = uploaded_files[:3]

with right:
    st.subheader("2. Inspection Results")

    if uploaded_files:
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            st.divider()
            st.markdown(f"### Product {index}")

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

                        st.metric(
                            "Prediction Score",
                            f"{confidence:.2f}%"
                        )

                        st.warning(
                            "Confirm the model's training class order "
                            "before relying on the GOOD/DEFECTIVE label. "
                            "The displayed score is not a calibrated "
                            "probability unless the model supports that."
                        )

            except Exception as e:
                st.error(f"Could not open this image: {e}")

    else:
        st.write("Upload images to see inspection results here.")

st.divider()
st.subheader("📊 Model Evaluation")

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

st.divider()
st.subheader("📄 Inspection Report")

st.write(
    "Use your browser's Print option to save this page as PDF. "
    "Downloadable inspection history is not included yet."
)

st.caption(
    "Defect heatmaps and visual defect highlighting "
    "are not implemented in this version."
)
