import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("loadshedding.csv")

# Handling missing values
imputer = SimpleImputer(strategy="mean")  # You can use 'median' or 'most_frequent'
df.iloc[:, :] = imputer.fit_transform(df)

# Encoding categorical data (if any)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Dimensionality Reduction (PCA)
pca = PCA(n_components=2)  # Adjust components as needed
reduced_data = pca.fit_transform(scaled_data)

# Convert back to DataFrame
df_pca = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])

# Save the processed data
df_pca.to_csv("processed_data.csv", index=False)

print("Preprocessing completed. Processed data saved as 'processed_data.csv'.")
