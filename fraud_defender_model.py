# Step 1: Import Required Libraries
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

# Step 2: Load the Data
df = pd.read_csv("creditcard.csv")

# Step 3: Exploratory Data Analysis (EDA)
print(df.info())
print(df.describe())
print(df.head())
print("Missing Values:\n", df.isnull().sum())

# Step 4: Data Preprocessing - Scaling 'Amount' and Dropping 'Time'
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df.drop(columns=["Time"], inplace=True)

# Step 5: Handling Class Imbalance with SMOTE
X = df.drop(columns=["Class"])
y = df["Class"]

# Applying SMOTE to oversample minority class
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verifying class distribution after resampling
print("Class distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 7: Define Models and Evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=6, learning_rate=0.1, n_jobs=-1)
}

# Store performance metrics
results = {}

for model_name, model in models.items():
    print(f"\n🔹 Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred)
    }

    results[model_name] = metrics

    # Print metrics for current model
    print(f"{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.4f}")

# Step 8: Visualize Confusion Matrices
for model_name, model in models.items():
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm).plot(cmap='coolwarm', values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Step 9: Save the Best Model
# Find the model with the highest accuracy and save it
best_model_name = max(results, key=lambda model: results[model]["Accuracy"])
best_model = models[best_model_name]

with open("fraud_defender_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print(f"\nBest model saved: {best_model_name}")
