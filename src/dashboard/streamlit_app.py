import streamlit as st
import subprocess
import os
import base64
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Fashion Trend Predictor",
    page_icon="👗",
    layout="wide"
)


# --------------------------------
# Background setup
# --------------------------------

def set_background():

    image_path = Path("assets/dashboard_bg.jpeg")

    if not image_path.exists():
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .overlay-box {{
            background-color: rgba(255,255,255,0.92);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #d1d5db;
        }}

        .header-box {{
            background-color: #111827;
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 20px;
        }}

        .chatbot-icon {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background-color: #2563eb;
            color: white;
            padding: 15px;
            border-radius: 50%;
            font-size: 20px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }}

        .scroll-hint {{
            animation: fadeInOut 4s infinite;
            text-align: center;
            font-weight: bold;
            color: #374151;
        }}

        @keyframes fadeInOut {{
            0% {{opacity:0}}
            50% {{opacity:1}}
            100% {{opacity:0}}
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background()


# --------------------------------
# Navigation bar
# --------------------------------

nav1, nav2, nav3, nav4 = st.columns([4,1,1,1])

with nav1:
    st.markdown('<div class="header-box">POKAK TECHNOLOGIES PVT LTD</div>', unsafe_allow_html=True)

with nav2:
    st.button("About")

with nav3:
    st.button("Contact")

with nav4:
    st.text_input("🔍", placeholder="Search")

st.divider()


# --------------------------------
# Hero section
# --------------------------------

st.markdown("## 👗 Fashion Trend Prediction")

st.markdown("""
AI system that analyzes **visual fashion trends, social media discussions, and machine learning predictions**  
to forecast **next-week clothing trends**.
""")

st.markdown('<div class="scroll-hint">⬇ Scroll down for FAQ and Feedback</div>', unsafe_allow_html=True)


# --------------------------------
# Cached helper function
# --------------------------------

@st.cache_data
def load_report(path):

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "Report not found. Please run the pipeline."


# --------------------------------
# Pipeline execution
# --------------------------------

def run_pipeline():

    python_exec = sys.executable

    subprocess.run([python_exec, "-m", "src.run_pipeline"])


# --------------------------------
# Sidebar controls
# --------------------------------

st.sidebar.title("Controls")

if st.sidebar.button("Run Full Pipeline"):

    progress = st.progress(0)

    # CV stage start
    st.toast("📷 Computer Vision Analysis running...")
    progress.progress(25)

    # Run pipeline
    run_pipeline()

    # Stage completion messages
    st.toast("✅ Computer Vision Analysis completed")

    progress.progress(50)

    st.toast("🧠 NLP Sentiment Analysis completed")

    progress.progress(75)

    st.toast("📈 ML Trend Forecast completed")

    progress.progress(100)

    st.toast("🎉 Pipeline completed successfully!")

    # clear cache to reload new results
    load_report.clear()


st.sidebar.markdown("---")

st.sidebar.info("""
Demo Instructions

1. Click **Run Full Pipeline**
2. Wait for analysis to complete
3. Dashboard automatically shows results
""")


# --------------------------------
# Load reports
# --------------------------------

ml_report = load_report("reports/ml_trend_prediction.txt")
cv_report = load_report("reports/visual_trend_summary.txt")
nlp_report = load_report("reports/nlp_trend_summary.txt")


# --------------------------------
# ML Prediction
# --------------------------------

st.header("📈 ML Trend Forecast")

with st.container():
    st.markdown('<div class="overlay-box">', unsafe_allow_html=True)
    st.code(ml_report)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()


# --------------------------------
# CV + NLP + FAQ + Feedback layout
# --------------------------------

col1, col2 = st.columns([1,1])

with col1:

    st.subheader("📷 Visual Trend Analysis")

    st.markdown('<div class="overlay-box">', unsafe_allow_html=True)
    st.code(cv_report)
    st.markdown('</div>', unsafe_allow_html=True)


with col2:

    st.subheader("🧠 NLP Trend Analysis")

    st.markdown('<div class="overlay-box">', unsafe_allow_html=True)
    st.code(nlp_report)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("FAQ")

    with st.expander("How does the prediction work?"):
        st.write("The system combines visual signals, text discussions, and machine learning forecasting.")

    with st.expander("What data sources are used?"):
        st.write("Fashion images, product data, and social media trend signals.")

    with st.expander("What does 'Stable' trend mean?"):
        st.write("The trend is expected to continue at the current level next week.")

    st.subheader("User Feedback")

    name = st.text_input("Your name")
    comment = st.text_area("Your feedback")

    if st.button("Submit Feedback"):

        if name and comment:

            os.makedirs("feedback", exist_ok=True)

            file = "feedback/user_feedback.csv"

            data = {
                "name":[name],
                "comment":[comment],
                "time":[datetime.now()]
            }

            df = pd.DataFrame(data)

            if os.path.exists(file):
                df.to_csv(file, mode="a", header=False, index=False)
            else:
                df.to_csv(file, index=False)

            st.success("Thank you! Your feedback has been recorded.")

        else:
            st.warning("Please enter both name and feedback.")


# --------------------------------
# Chatbot dummy icon
# --------------------------------

st.markdown('<div class="chatbot-icon">💬</div>', unsafe_allow_html=True)


# --------------------------------
# Footer
# --------------------------------

st.divider()

st.caption("© 2026 POKAK TECHNOLOGIES PVT LTD — Fashion Trend Predictor")