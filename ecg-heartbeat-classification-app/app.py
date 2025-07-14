import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("CNN_model.h5")

# Class labels
class_labels = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
}

# Page configuration
st.set_page_config(page_title="ECG Heartbeat Diagnosis", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Full-page centering */
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            padding-top: 20px;
        }

        /* Title Design (Single Line) */
        .title {
            font-size: 60px;
            font-weight: bold;
            text-align: center;
            color: #D72638;
            margin-bottom: 10px;
            margin-left: -40px;
            white-space: nowrap; /* Ensures the title stays on one line */
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Heartbeat Animation (Now inside title for single-line effect) */
        .title::before {
            content: "💓";
            animation: heartbeat 1.5s infinite;
            display: inline-block;
            margin: 0 10px;
        }

        /* Subtitle */
        .subtext {
            text-align: center;
            font-size: 25px;
            margin-bottom: 30px;
            color: #666;
        }

        /* Tabs Styling */
        .stTabs [role="tab"] {
            font-size: 24px !important;
            font-weight: bold;
            text-transform: uppercase;
        }

        /* Upload Box */
        .stFileUploader {
            font-size: 24px !important;
        }

        /* Input Fields */
        .stTextArea textarea,
        .stTextInput > div > div > input {
            font-size: 24px !important;
        }
        .st-emotion-cache-165fv6u p, .st-emotion-cache-165fv6u ol, .st-emotion-cache-165fv6u ul, .st-emotion-cache-165fv6u dl, .st-emotion-cache-165fv6u li {
            font-size: 17px;
        }

        /* Buttons */
        .stButton button {
            font-size: 24px !important;
            font-weight: bold;
            background-color: #D72638;
            color: white;
            border-radius: 10px;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #B71C1C;
            transform: scale(1.05);
        }

        /* Prediction Result */
        .stSuccess, .stInfo {
            font-size: 26px !important;
            font-weight: bold;
            text-align: center;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 22px;
            color: #888;
            margin-top: 80px;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes heartbeat {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'> Heartbeat Diagnosis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload a file or paste values to predict Heartbeat Categorization</div>", unsafe_allow_html=True)
st.markdown("")

# Tabs for input method
tab1, tab2 = st.tabs(["📁 Upload CSV", "✍️ Manual Input"])

with tab1:
    st.subheader("📄 Upload a CSV file with ECG values")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            values = df.values.flatten()

            if len(values) != 187:
                st.error("⚠️ CSV must contain exactly 187 values.")
            else:
                reshaped = np.array(values, dtype=np.float32).reshape(1, 187, 1)
                prediction = model.predict(reshaped)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100

                st.success(f"✅ Diagnosis: {class_labels[class_index]}")
                st.info(f"📊 Confidence: {confidence:.2f}%")
        except Exception as e:
            st.error(f"❌ Error: {e}")

with tab2:
    st.subheader("📝 Paste 187 comma/tab-separated ECG values")
    user_input = st.text_area("Paste values here:", height=150, placeholder="e.g. 0.1, 0.2, 0.3, ...")
    
    if st.button("🔍 Predict from Text"):
        try:
            cleaned = user_input.strip().replace("\t", ",").replace("\n", ",")
            values = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
            
            if len(values) != 187:
                st.error("⚠️ Please enter exactly 187 numbers.")
            else:
                reshaped = np.array(values, dtype=np.float32).reshape(1, 187, 1)
                prediction = model.predict(reshaped)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100

                st.success(f"✅ Diagnosis: {class_labels[class_index]}")
                st.info(f"📊 Confidence: {confidence:.2f}%")
        except Exception as e:
            st.error(f"❌ Invalid input: {e}")

# Footer
st.markdown("<div class='footer'>Made with ❤️ by CardioAI | Powered by CNN & Streamlit</div>", unsafe_allow_html=True)