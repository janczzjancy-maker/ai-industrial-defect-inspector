import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AI Industrial Defect Inspector", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.stApp { background: #081426; color: #eef4ff; }
[data-testid="stSidebar"] { background: #101f35; }
h1, h2, h3 { color: #eef4ff; }
div[data-testid="stMetric"] { background: #12243d; border: 1px solid #29415f; padding: 16px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 AI Industrial Defect Inspector")
st.caption("Computer Vision • Industrial Casting Quality Inspection")
st.info("Upload a product image to inspect it. For real AI predictions, add your trained model file named `industrial_defect_model.keras` to this project.")

MODEL_PATH = Path("industrial_defect_model.keras")

@st.cache_resource
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def predict_image(image):
    if not MODEL_PATH.exists():
        return None, None, "Model file not found"
    try:
        model = load_model(str(MODEL_PATH))
        image_rgb = image.convert("RGB").resize((128, 128))
        arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        raw = float(np.asarray(model.predict(arr, verbose=0)).reshape(-1)[0])
        # Assumes a sigmoid model trained with class 0 = defective, class 1 = good.
        # Change this mapping if your training class order was different.
        if raw >= 0.5:
            return "GOOD", raw * 100, None
        return "DEFECTIVE", (1 - raw) * 100, None
    except Exception as e:
        return None, None, str(e)

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Upload product image")
    uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
    st.caption("Tip: use an image similar to the industrial casting images used to train your model.")
with right:
    st.subheader("2. Inspection result")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded product", use_container_width=True)
        if st.button("🔍 Run AI Inspection", type="primary", use_container_width=True):
            label, confidence, error = predict_image(image)
            if error:
                st.warning("AI model is not connected yet. Add `industrial_defect_model.keras` to this project and install the requirements.")
                with st.expander("Technical details"):
                    st.code(error)
            else:
                if label == "GOOD":
                    st.success(f"✅ {label} — {confidence:.2f}% confidence")
                else:
                    st.error(f"⚠️ {label} — {confidence:.2f}% confidence")
                st.metric("Prediction confidence", f"{confidence:.2f}%")
                st.caption("This result uses your trained model. Verify the model's class mapping before presenting results.")
                st.warning("Defect highlighting/heatmap is not implemented in this starter version; it requires a localization method such as Grad-CAM.")
    else:
        st.write("Your prediction will appear here after you upload an image.")

st.divider()
st.subheader("Model evaluation")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", "Add evaluation result")
c2.metric("Precision", "Add evaluation result")
c3.metric("Recall", "Add evaluation result")
st.caption("Replace these placeholders with the final values measured on your held-out test dataset. Do not report demo values as real model performance.")

st.subheader("Inspection report")
st.write("After running an inspection, use your browser's print option to save this page as PDF. Downloadable CSV/PDF history can be added as a next step.")
