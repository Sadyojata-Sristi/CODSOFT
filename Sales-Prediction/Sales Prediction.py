#SALES PREDICTION USING MACHINE LEARNING

#Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load Dataset

#Read the file
df = pd.read_csv("advertising.csv")

#Display first 5 rows
print("\nFirst 5 Records:")
print(df.head())

#Dataset Information
print("\nDataset Information:")
print(df.info())

#Check Missing Values


print("\nMissing Values:")
print(df.isnull().sum())

#Exploratory Data Analysis

#Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

#Correlation Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Pair Plot

sns.pairplot(fn)
plt.show()

#Define Features and Target

X = df[['TV', 'Radio', 'Newspaper']]  # Independent Variables
y = df['Sales']                       # Target Variable

#Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

#Train Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

#Make Predictions

y_pred = model.predict(X_test)

#Evaluate Model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics")
print("-" * 30)
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R2 Score:", round(r2, 2))

#Display Coefficients

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance:")
print(coefficients)

print("\nIntercept:")
print(model.intercept_)

#Actual vs Predicted Values

results = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': y_pred
})

print("\nActual vs Predicted Sales:")
print(results.head(10))

#Visualization of Predictions

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

#Predict New Sales

# Example:
# TV = 150
# Radio = 25
# Newspaper = 30

new_data = pd.DataFrame({
    'TV': [150],
    'Radio': [25],
    'Newspaper': [30]
})

prediction = model.predict(new_data)

print("\nPredicted Sales for New Advertisement Budget:")
print("TV = 150")
print("Radio = 25")
print("Newspaper = 30")
print("Predicted Sales =", round(prediction[0], 2))

#End of Project
