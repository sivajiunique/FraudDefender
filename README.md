# Credit Card Fraud Detection - README

## Project Overview
This project aims to detect fraudulent credit card transactions using machine learning. A Random Forest Classifier is trained on a dataset containing transactions from European cardholders in September 2013.

## Folder Structure
```
├── creditcard.csv            # https://kh3-ls-storage.s3.us-east-1.amazonaws.com/DS Project Guide Data Set/creditcard.csv

├── credit_card_fraud_detection.py      # Python script for model training
├── fraud_detection_model.pkl  # Trained machine learning model
├── report.pdf              # Project report
├── README.md               # Documentation
```

## Installation & Setup
### 1. Clone the Repository
```
git clone https://github.com/radhaprofile/Credit_Card_fraud.git
cd Credit_Card_fraud
```

### 2. Install Dependencies
Make sure you have Python 3.x installed. Then, install required libraries:
```
pip install -r requirements.txt
```

### 3. Run the Model Training Script
```
python credit_card_fraud_detection.py
```

## Model Details
- **Algorithms Used:** Logistic Regression , Decision Tree, Random Forest , XGBoost
- **Handling Imbalanced Data:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Performance Metrics:** Accuracy, Precision, Recall, Confusion Matrix
- **Final Model Accuracy:** >75%

## Deployment Plan
The model can be deployed using Flask or FastAPI for real-time fraud detection.

## Future Enhancements
- Implement deep learning models for improved performance.
- Integrate with a real-time fraud detection API.
- Optimize hyperparameters further.

## Contact
For any questions, please reach out at [radhamam456@gmail.com].

