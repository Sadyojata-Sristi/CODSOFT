#Iris Flower Classification

#IMPORT REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import joblib

#LOAD THE DATASET

print("Loading Dataset...\n")

df = pd.read_csv("IRIS.csv")

print("Dataset Loaded Successfully!\n")


#DISPLAY DATASET INFORMATION

print("First 5 Rows of Dataset:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

#CHECK FOR MISSING VALUES

print("\nMissing Values in Dataset:\n")
print(df.isnull().sum())


#EXPLORATORY DATA ANALYSIS (EDA)

# Display basic statistics

print("\nStatistical Summary:\n")
print(df.describe())

# Pairplot Visualization

sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()


#ENCODE TARGET VARIABLE

# Convert species names into numerical values

encoder = LabelEncoder()

df["species"] = encoder.fit_transform(df["species"])

print("\nSpecies Encoding:")
print(dict(zip(encoder.classes_,
               encoder.transform(encoder.classes_))))


#DEFINE FEATURES AND TARGET VARIABLE

# Input Features

X = df.drop("species", axis=1)

# Target Variable

y = df["species"]


#SPLIT DATA INTO TRAINING AND TESTING SETS

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


#TRAIN THE MACHINE LEARNING MODEL

print("\nTraining Random Forest Model...\n")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

#MAKE PREDICTIONS

y_pred = model.predict(X_test)


#EVALUATE MODEL PERFORMANCE

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


#CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()



#FEATURE IMPORTANCE ANALYSIS

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(feature_importance)

#Plot Feature Importance

plt.figure(figsize=(8, 4))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Feature Importance")
plt.show()


#CROSS VALIDATION

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nAverage Cross Validation Accuracy:")
print(round(cv_scores.mean() * 100, 2), "%")


#SAVE TRAINED MODEL

joblib.dump(model, "iris_model.pkl")

print("\nModel Saved Successfully!")
print("File Name: iris_model.pkl")


#TEST MODEL WITH NEW FLOWER DATA

# Example Flower:
# Sepal Length = 5.1
# Sepal Width  = 3.5
# Petal Length = 1.4
# Petal Width  = 0.2

sample_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample_flower)

species_name = encoder.inverse_transform(prediction)

print("\nPrediction for Sample Flower:")
print("Predicted Species =", species_name[0])

print("\nProject Execution Completed Successfully!")

# END OF PROJECT
