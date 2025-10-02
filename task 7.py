

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------
# Step 1: Load Dataset
# ------------------------
file_path = r"C:\Users\Dimple.S\Downloads\breast-cancer.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print(df.head())

# ------------------------
# Step 2: Preprocessing
# ------------------------

# If target column is categorical (e.g. 'benign/malignant')
# Replace 'diagnosis' with actual target column name
target_col = "diagnosis"   # 👈 update if your dataset has different name

# Encode target labels (e.g. M=1, B=0)
if df[target_col].dtype == 'object':
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------
# Step 3: Train SVM Model
# ------------------------
svm_model = SVC(kernel='linear', random_state=42)  # try 'rbf' too
svm_model.fit(X_train, y_train)

# ------------------------
# Step 4: Evaluation
# ------------------------
y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()
