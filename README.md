# 🧠 Prediction of Disease Outbreaks System

This project implements a **machine learning-based system** to predict the likelihood of **diabetes, heart disease, and Parkinson's disease** using **Support Vector Classifier (SVC)** models.

## 🚀 **Project Overview**
The application uses **Streamlit** for deployment and is accessible at the following link:  
🔗 [Deployed Website](https://disease-outbreak-predictor-esh.streamlit.app/)

This project was developed as part of the **AICTE Internship on AI: Transformative Learning with TechSaksham**, a joint initiative by **Microsoft & SAP**.

## ⚙️ **Technology Stack**
- **Python** (ML model development)  
- **Streamlit** (web deployment)  
- **scikit-learn** (model training)  
- **pandas, numpy** (data processing)  

## 🛠️ **Setup Instructions**

1️⃣ **Clone the Repository:**
```bash
git clone https://github.com/eshitakundu/disease-outbreak-predictor.git
cd disease-outbreak-predictor
```

2️⃣ **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3️⃣ **Run the Application Locally:**
```bash
streamlit run app.py
```

## 📊 **How It Works**
- Users input health data via the **Streamlit interface**.  
- The model applies **StandardScaler** for preprocessing.  
- Predictions are generated using the **SVC model**.  
- The result is displayed, indicating whether the individual is at risk.
