import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Step 1: Load the cleaned data (assuming it's already cleaned as per your previous steps)
file_path = '/Users/sajanbista/Desktop/MachineLearning Daily/to be done/messiGoal.csv'
messi_goals_cleaned = pd.read_csv('messiGoal.csv')

# Step 2: Prepare Features and Target
X = messi_goals_cleaned.iloc[ : :-1].values # Features
y = messi_goals_cleaned.iloc[:,-1].values # Target

# Convert categorical variables to numerical ones using one-hot encoding
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
le = LabelEncoder()
y= le.fit_transform(y)
# Step 3: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 4: Perform Feature Scaling (though RandomForest is less sensitive, it's good practice)
scaler = StandardScaler()

# Fit and transform the training data, transform the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with scaled data
rf_classifier.fit(X_train_scaled, y_train)

# Step 6: Evaluate the Model
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Additional evaluation with a classification report
print(classification_report(y_test, y_pred))

# Step 7: Visualize Feature Importance
importances = rf_classifier.feature_importances_

# Create a DataFrame to display features and their importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()
