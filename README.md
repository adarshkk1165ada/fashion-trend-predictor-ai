# 👗 Fashion Trend Predictor AI

An end-to-end **AI system that predicts upcoming fashion trends** by combining:

- Computer Vision (image analysis)
- Natural Language Processing (social media trend analysis)
- Machine Learning forecasting
- Interactive Streamlit dashboard

The system analyzes fashion images, social media discussions, and historical trend data to forecast **next-week fashion trends**.

---

# 🚀 Features

### 📷 Computer Vision Trend Analysis
- Detects clothing categories from fashion images
- Extracts color style patterns
- Identifies trends such as:
  - Pastel
  - Dark
  - Bright color styles

### 🧠 NLP Fashion Trend Mining
Analyzes fashion discussions from social media:

- Sentiment analysis
- Keyword extraction
- Topic modeling (LDA)
- Engagement scoring

This helps identify **what people are talking about in fashion right now.**

### 📈 Machine Learning Trend Forecasting
A trained ML model predicts **next week’s trend movement**:

- Rising
- Stable
- Declining

Based on engineered fashion trend features.

### 📊 Interactive Dashboard
Built using **Streamlit** to visualize:

- ML predictions
- Visual fashion trends
- NLP trend insights
- FAQ and feedback system

---

# 🏗 System Architecture
# 👗 Fashion Trend Predictor AI

An end-to-end **AI system that predicts upcoming fashion trends** by combining:

- Computer Vision (image analysis)
- Natural Language Processing (social media trend analysis)
- Machine Learning forecasting
- Interactive Streamlit dashboard

The system analyzes fashion images, social media discussions, and historical trend data to forecast **next-week fashion trends**.

---

# 🚀 Features

### 📷 Computer Vision Trend Analysis
- Detects clothing categories from fashion images
- Extracts color style patterns
- Identifies trends such as:
  - Pastel
  - Dark
  - Bright color styles

### 🧠 NLP Fashion Trend Mining
Analyzes fashion discussions from social media:

- Sentiment analysis
- Keyword extraction
- Topic modeling (LDA)
- Engagement scoring

This helps identify **what people are talking about in fashion right now.**

### 📈 Machine Learning Trend Forecasting
A trained ML model predicts **next week’s trend movement**:

- Rising
- Stable
- Declining

Based on engineered fashion trend features.

### 📊 Interactive Dashboard
Built using **Streamlit** to visualize:

- ML predictions
- Visual fashion trends
- NLP trend insights
- FAQ and feedback system

---

# 🏗 System Architecture
Fashion Trend Predictor
│
├── Computer Vision Layer
│ ├── Image classification
│ ├── Color feature extraction
│ └── Visual trend analysis
│
├── NLP Layer
│ ├── Text preprocessing
│ ├── Sentiment analysis
│ ├── Keyword extraction
│ └── Topic modeling
│
├── Machine Learning Layer
│ └── Trend forecasting model
│
└── Streamlit Dashboard
├── ML prediction view
├── CV + NLP insights
└── User feedback system

# 📂 Project Structure
Fashion_Trend_Prediction
│
├── assets
│ └── dashboard background images
│
├── config
│ └── project configuration
│
├── data
│ ├── raw_data
│ └── processed_data
│
├── models
│ └── trained ML models
│
├── reports
│ ├── visual_trend_summary.txt
│ ├── nlp_trend_summary.txt
│ └── ml_trend_prediction.txt
│
├── src
│ ├── vision
│ ├── nlp
│ ├── models
│ └── dashboard
│
├── requirements.txt
└── README.md

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/adarshkk1165ada/fashion-trend-predictor-ai.git
cd fashion-trend-predictor-ai

 # Create virtual environment: 
 python -m venv venv 

 #Activate environment:

Windows:  venv\Scripts\activate

Mac / Linux:  source venv/bin/activate

Install dependencies:  pip install -r requirements.txt

Run the Application

Start the Streamlit dashboard:

python -m streamlit run src/dashboard/streamlit_app.py

Open in browser:

http://localhost:8501

🧪 Running the Full Pipeline

The dashboard includes a Run Full Pipeline button which executes:

Computer Vision analysis

NLP trend analysis

Machine Learning prediction

Generated reports are stored in:

reports/

📸 Example Dashboard

The dashboard displays:

ML Trend Forecast

Visual Fashion Analysis

NLP Trend Insights

FAQ and Feedback section

📌 Future Improvements

Possible extensions:

Real-time social media scraping

Fashion brand trend tracking

Deep learning image classification

Deployment on cloud platforms