import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset

data = pd.read_csv("Telco_Customer_Churn_Dataset.csv")

# Step 2: Data Preprocessing
# Let's check for missing values and handle them
data.isnull().sum()

# Drop unnecessary columns (like customerID)
data.drop(columns=['customerID'], axis=1, inplace=True)

# Convert categorical variables to numerical using pd.get_dummies()
data = pd.get_dummies(data, drop_first=True)

# Step 3: Split the data into features and target variable
X = data.drop(columns=['Churn_Yes'])  # Features
y = data['Churn_Yes']  # Target variable (1 if churned, 0 if not)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

