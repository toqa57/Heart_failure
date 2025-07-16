import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors
import streamlit as st

# === Configuration ===
USE_STREAMLIT = True  # Set to True for Streamlit, False for Gradio/Hugging Face


# === Red Theme (Gradio only) ===
class RedButtonTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.red,
            neutral_hue=colors.gray,
        )


# === Paths ===
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"
MLP_PATH = MODEL_DIR / "mlp.pkl"


# === Load Model ===
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")


# === Converter for radio buttons ===
def to_numeric(val): return 1 if val in ["Yes", "Male"] else 0


# === Risk Prediction ===
def predict_risk(model_choice, age, anaemia, cpk, diabetes, ef, hypertension,
                 platelets, creatinine, sodium, sex, smoking, time):
    try:
        model_path = RF_PATH if model_choice == "Random Forest" else MLP_PATH
        model = load_model(model_path)
        scaler = joblib.load(SCALER_PATH)

        # Build DataFrame with correct feature names
        columns = ["age", "anaemia", "creatinine_phosphokinase", "diabetes",
                   "ejection_fraction", "high_blood_pressure", "platelets",
                   "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]
        data = [[age, anaemia, cpk, diabetes, ef, hypertension,
                 platelets, creatinine, sodium, sex, smoking, time]]
        input_df = pd.DataFrame(data, columns=columns)

        input_scaled = scaler.transform(input_df)
        proba = model.predict_proba(input_scaled)[0][1]
        proba = np.clip(proba, 0.05, 0.95)
        will_die = proba >= 0.5

        return will_die, proba, model_choice

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")


# === Streamlit App ===
def run_streamlit_app():
    st.set_page_config(page_title="Heart Failure Risk Predictor", layout="wide")

    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .high-risk {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .low-risk {
        background-color: #ccffcc;
        border-left: 5px solid #00aa00;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🫀 Heart Failure Risk Predictor v2.3")
    st.markdown("_AI-powered decision support tool for clinicians_")

    with st.sidebar:
        st.header("Model Selection")
        model_choice = st.radio("Choose Prediction Model",
                                ["Random Forest", "MLP"],
                                index=0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Patient Info")
        age = st.slider("Age (years)", 40, 95, 65)
        sex = st.radio("Sex", ["Female", "Male"], index=1)
        smoking = st.radio("Smoking", ["No", "Yes"], index=0)

        st.subheader("🩸 Conditions")
        anaemia = st.radio("Anaemia", ["No", "Yes"], index=0)
        diabetes = st.radio("Diabetes", ["No", "Yes"], index=0)
        hypertension = st.radio("Hypertension", ["No", "Yes"], index=0)

    with col2:
        st.subheader("🧪 Clinical Measures")
        ef = st.slider("Ejection Fraction (%)", 15, 80, 38)
        creatinine = st.number_input("Serum Creatinine (mg/dL)", value=1.2, min_value=0.5, max_value=10.0, step=0.1)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", value=250000, min_value=25000, max_value=850000,
                                    step=1000)
        cpk = st.number_input("Creatinine Phosphokinase", value=250, min_value=50, max_value=8000, step=10)
        sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 135)
        time = st.number_input("Follow-up Time (days)", value=100, min_value=1, max_value=300, step=1)

    if st.button("🔍 Predict Risk", type="primary"):
        try:
            will_die, proba, model_used = predict_risk(
                model_choice, age, to_numeric(anaemia), cpk, to_numeric(diabetes),
                ef, to_numeric(hypertension), platelets, creatinine,
                sodium, to_numeric(sex), to_numeric(smoking), time
            )

            risk_class = "high-risk" if will_die else "low-risk"
            emoji = "💀" if will_die else "🫁"
            risk_text = "High Risk" if will_die else "Low Risk"

            st.markdown(f"""
            <div class="result-box {risk_class}">
                <h3>🧾 Prediction Result</h3>
                <p class="big-font">{emoji} Patient will {'likely die' if will_die else 'likely survive'}</p>
                <p><strong>Model Used:</strong> {model_used}</p>
                <p><strong>Risk Level:</strong> {"🔴 " + risk_text if will_die else "🟢 " + risk_text}</p>
                <p><strong>Death Probability:</strong> {proba * 100:.1f}%</p>
                <p><strong>Recommended Action:</strong> {
            "🚨 Immediate intervention" if proba > 0.75 else
            "⚠️ Close monitoring" if proba > 0.45 else
            "✅ Routine checkup"
            }</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"**Error:** {str(e)}")


# === Gradio App ===
def run_gradio_app():
    with gr.Blocks(theme=RedButtonTheme(), title="Heart Failure Risk Predictor") as app:
        gr.Markdown("""
        # 🫀 Heart Failure Risk Predictor v2.3  
        _AI-powered decision support tool for clinicians_
        """)

        with gr.Group():
            model_choice = gr.Radio(
                ["Random Forest", "MLP"],
                value="Random Forest",
                label="Choose Prediction Model"
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👤 Patient Info")
                age = gr.Slider(40, 95, value=65, label="Age (years)")
                sex = gr.Radio(["Female", "Male"], value="Male", label="Sex")
                smoking = gr.Radio(["No", "Yes"], value="No", label="Smoking")

            with gr.Column():
                gr.Markdown("### 🧪 Clinical Measures")
                ef = gr.Slider(15, 80, value=38, label="Ejection Fraction (%)")
                creatinine = gr.Number(value=1.2, label="Serum Creatinine (mg/dL)")
                platelets = gr.Number(value=250000, label="Platelets (kiloplatelets/mL)")

        with gr.Row():
            with gr.Column():
                anaemia = gr.Radio(["No", "Yes"], value="No", label="Anaemia")
                diabetes = gr.Radio(["No", "Yes"], value="No", label="Diabetes")
                hypertension = gr.Radio(["No", "Yes"], value="No", label="Hypertension")

            with gr.Column():
                cpk = gr.Number(value=250, label="Creatinine Phosphokinase")
                sodium = gr.Slider(110, 150, value=135, label="Serum Sodium (mEq/L)")
                time = gr.Number(value=100, label="Follow-up Time (days)")

        submit_btn = gr.Button("🔍 Predict Risk", variant="primary")
        result_output = gr.Markdown(label="🧾 Prediction Result")

        def format_result(will_die, proba, model_choice):
            return f"""
            ### 🧾 Prediction Result
            - **Prediction:** {'💀 Patient will likely die' if will_die else '🫁 Patient will likely survive'}
            - **Model Used:** {model_choice}
            - **Risk Level:** {'🔴 High Risk' if will_die else '🟢 Low Risk'}
            - **Death Probability:** {proba * 100:.1f}%
            - **Recommended Action:** {(
                "🚨 Immediate intervention" if proba > 0.75 else
                "⚠️ Close monitoring" if proba > 0.45 else
                "✅ Routine checkup"
            )}
            """

        submit_btn.click(
            fn=lambda model_choice, age, anaemia, cpk, diabetes, ef, hypertension,
                      platelets, creatinine, sodium, sex, smoking, time:
            predict_risk(
                model_choice, age, to_numeric(anaemia), cpk, to_numeric(diabetes),
                ef, to_numeric(hypertension), platelets, creatinine,
                sodium, to_numeric(sex), to_numeric(smoking), time
            ),
            inputs=[model_choice, age, anaemia, cpk, diabetes, ef, hypertension,
                    platelets, creatinine, sodium, sex, smoking, time],
            outputs=result_output,
            api_name="predict"
        )

    app.launch(server_port=7861, share=True, show_error=True)


# === Main Execution ===
if __name__ == "__main__":
    if USE_STREAMLIT:
        run_streamlit_app()
    else:
        run_gradio_app()