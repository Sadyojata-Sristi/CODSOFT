
#CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING

# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

#LOAD DATASET

print("Loading Dataset...")

data = pd.read_csv("creditcard.csv")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

#CHECK MISSING VALUES

print("\nMissing Values:")
print(data.isnull().sum())

#CLASS DISTRIBUTION

print("\nClass Distribution:")
print(data['Class'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data)
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


#FEATURE SCALING

scaler = StandardScaler()

data['Amount'] = scaler.fit_transform(
    data[['Amount']]
)

data['Time'] = scaler.fit_transform(
    data[['Time']]
)


#FEATURE SELECTION

X = data.drop('Class', axis=1)
y = data['Class']


#TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

#HANDLE CLASS IMBALANCE USING SMOTE

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

#FUNCTION TO EVALUATE MODELS

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return y_pred


#LOGISTIC REGRESSION

print("\n")
print("="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_smote,
    y_train_smote
)

evaluate_model(
    lr_model,
    X_test,
    y_test
)

#RANDOM FOREST

print("\n")
print("="*50)
print("RANDOM FOREST")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

evaluate_model(
    rf_model,
    X_test,
    y_test
)

#ROC CURVE

rf_probs = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(
    y_test,
    rf_probs
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    label=f'ROC Curve (AUC = {roc_auc:.4f})'
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#FEATURE IMPORTANCE

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")
plt.show()

#SAVE MODEL

import joblib

joblib.dump(
    rf_model,
    "fraud_detection_model.pkl"
)

print("\nModel Saved Successfully!")
print("File Name: fraud_detection_model.pkl")

#END OF PROJECT
