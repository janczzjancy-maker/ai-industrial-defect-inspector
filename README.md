# AI Industrial Defect Inspector

A Streamlit starter app for industrial product image inspection.

## Run locally
1. Put `app.py` and `requirements.txt` in one folder.
2. Add your trained model file named `industrial_defect_model.keras` to the same folder.
3. Create/activate a Python environment compatible with TensorFlow 2.15.1 (Python 3.10 or 3.11 recommended).
4. Install dependencies:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`

## Important model note
This starter assumes a single-output sigmoid model where output >= 0.5 means GOOD and output < 0.5 means DEFECTIVE. Verify your training class order and edit `predict_image()` if needed.

The evaluation metrics are intentionally placeholders until you enter values from your own held-out test set. Defect heatmap/localization and downloadable inspection history are not included yet.
