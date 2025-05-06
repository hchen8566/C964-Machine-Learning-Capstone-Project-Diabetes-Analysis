# Hao Chen
# C964 Capstone Project: Machine Learning
# ID: 010771133


# importing required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# reading the diabetes dataset and saving it as diabetes_df
diabetes_df = pd.read_csv("diabetes.csv")

# replacing invalid values (0's) with nan and then removing these invalid data
diabetes_columns = ['Glucose', 'BloodPressure', 'BMI', 'Age']
diabetes_df[diabetes_columns] = diabetes_df[diabetes_columns].replace(0, np.nan)
diabetes_df = diabetes_df.dropna()

# the variables and outcome split
y = diabetes_df["Outcome"]
X = diabetes_df.drop("Outcome", axis=1)

# Split into train (60%), validation (20%), and test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "diabetes_model.pkl")
print("✅ Model saved as diabetes_model.pkl")

# Predict on validation data
val_predictions = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))
print("Classification Report:\n", classification_report(y_val, val_predictions))