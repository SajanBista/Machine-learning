from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import concurrent.futures
import threading


# initialize the FastAPI app
app = FastAPI(title="Polynomial Regression API")

# Loading and checking the dataset 

dataset = pd.read_csv("Position_Salaries.csv")
print(dataset)


# extract input features (X) and output labels (y)
X = dataset.iloc[:, 1:-1].values  # Level column
y = dataset.iloc[:, -1].values    # Salary column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Thread-safe dictionary to store the best trained model
lock = threading.Lock()
best_model_info = {
    "model": None,
    "poly_features": None,
    "degree": None,
    "score": None
}

def train_polynomial_regression(degree):
    """
    trains a polynomial regression model of the given degree.
    returns the model, polynomial features, and the score on test data.
    """
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    X_poly_test = poly_features.transform(X_test)
    score = model.score(X_poly_test, y_test)

    return degree, model, poly_features, score

@app.post("/train")
def train_models():
    """
    trains polynomial regression models for degrees 2 through 5,
    and stores the one with the highest R^2 score.
    """
    degrees = [2, 3, 4, 5]
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_polynomial_regression, d) for d in degrees]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                return {"error": str(e)}

    if not results:
        return {"error": "Model training failed."}

    # Select the model with the best score
    best_degree, best_model, best_poly_features, best_score = max(results, key=lambda x: x[3])

    with lock:
        best_model_info["model"] = best_model
        best_model_info["poly_features"] = best_poly_features
        best_model_info["degree"] = best_degree
        best_model_info["score"] = best_score

    return {
        "message": "Training completed successfully.",
        "best_degree": best_degree,
        "best_score": round(best_score, 4)
    }

class PredictRequest(BaseModel):
    """
    Request body model for prediction endpoint.
    Expects a single numeric 'level' field.
    """
    level: float

@app.post("/predict")
def predict_salary(request: PredictRequest):
    """
    Predicts the salary for a given level using the trained model.
    Requires that the /train endpoint has been called successfully.
    """
    with lock:
        model = best_model_info.get("model")
        poly = best_model_info.get("poly_features")

        if model is None or poly is None:
            return {"error": "Model not trained yet. Please call /train first."}

        try:
            new_level = np.array([[request.level]])
            new_poly = poly.transform(new_level)
            prediction = model.predict(new_poly)

            return {
                "input_level": request.level,
                "predicted_salary": round(prediction[0], 2),
                "model_degree": best_model_info["degree"]
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

@app.get("/")
def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Polynomial Regression API is running."}
