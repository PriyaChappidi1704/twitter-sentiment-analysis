Customer Churn Prediction & Web App

Overview

This project predicts whether a telecommunications customer will churn (cancel their subscription) based on their account information. It demonstrates the end-to-end machine learning lifecycle, from data processing and handling imbalanced datasets to deploying an interactive web interface.

Technical Highlights

Algorithm: Random Forest Classifier (Optimized for low-latency web deployment).

Imbalanced Data Handling: Utilized SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class, preventing the model from becoming artificially biased toward the majority class (customers who do not churn).

Evaluation: Prioritized the Recall metric over standard accuracy. In a business context, the cost of a False Negative (failing to identify a churning customer) is significantly higher than a False Positive.

Deployment: Built an interactive frontend using Streamlit, allowing non-technical stakeholders to input customer data and receive real-time churn risk predictions.

How to Run Locally

1.Clone this repository to your local machine.

2.Install the required libraries:
pip install -r requirements.txt

3.Run the model training script to generate the serialized .pkl model files:
python train_model.py

4.Launch the web application:
streamlit run app.py
