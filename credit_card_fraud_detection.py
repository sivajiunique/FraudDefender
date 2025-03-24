#  Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#  Step 2: Load the Data
df = pd.read_csv("creditcard.csv")  # Ensure the CSV file is in the same directory

#  Step 3: Exploratory Data Analysis (EDA)
print(df.info())
print(df.describe())
print(df.head())
print("Missing Values:\n", df.isnull().sum())  # Check for missing values

#  Step 4: Data Preprocessing
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df = df.drop(columns=["Time"])  # Drop 'Time' as it's not useful

#  Step 5: Handling Class Imbalance using SMOTE (Balanced Sampling)
X = df.drop(columns=["Class"])
y = df["Class"]

smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Oversample fraud cases to 50% of non-fraud cases
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())

#  Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

#  Step 7: Train & Evaluate Multiple Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=6, learning_rate=0.1, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC-ROC": roc_auc
    }

    print(f" {name} Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-score: {f1:.4f}")
    print(f"  - AUC-ROC: {roc_auc:.4f}")

#  Step 8: Confusion Matrix Visualization
for name, model in models.items():
    plt.figure(figsize=(5, 5))
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm).plot(cmap='coolwarm', values_format='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

#  Step 9: Save the Best Model (Based on Accuracy)
best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
best_model = models[best_model_name]


with open("fraud_detection_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n Best model saved: {best_model_name}")
