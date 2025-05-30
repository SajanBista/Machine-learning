"""from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import concurrent.futures
import threading

app = FastAPI(title="Polynomial Regression API")

@app.get("/")
def read_root():
    return {"message": "Polynomial Regression API is running!"}


# Loading dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# shared state for model
lock = threading.Lock()
best_model_info = {
    "model": None,
    "poly_features": None,
    "degree": None,
    "score": None
}


def train_polynomial_regression(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    X_poly_test = poly_features.transform(X_test)
    score = model.score(X_poly_test, y_test)

    return degree, model, poly_features, score

@app.post("/train")
def train_models():
    degrees = [2, 3, 4, 5]
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_polynomial_regression, d) for d in degrees]
        for future in concurrent.futures.as_completed(futures):
            try:
                degree, model, poly_features, score = future.result()
                results.append((degree, model, poly_features, score))
            except Exception as e:
                return {"error": str(e)}

    # Selecting the best model
    best_degree, best_model, best_poly_features, best_score = max(results, key=lambda x: x[3])
    
    with lock:
        best_model_info["model"] = best_model
        best_model_info["poly_features"] = best_poly_features
        best_model_info["degree"] = best_degree
        best_model_info["score"] = best_score

    return {
        "message": "Training completed.",
        "best_degree": best_degree,
        "best_score": round(best_score, 4)
    }


class PredictRequest(BaseModel):
    level: float


@app.post("/predict")
def predict_salary(request: PredictRequest):
    with lock:
        if best_model_info["model"] is None:
            return {"error": "Model not trained yet. Call /train first."}

        new_level = np.array([[request.level]])
        new_poly = best_model_info["poly_features"].transform(new_level)
        prediction = best_model_info["model"].predict(new_poly)

    return {
        "input_level": request.level,
        "predicted_salary": round(prediction[0], 2),
        "model_degree": best_model_info["degree"]
    }
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import concurrent.futures
import threading

# ✅ Only one FastAPI instance
app = FastAPI(title="Polynomial Regression API")

# Load dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thread-safe shared model info
lock = threading.Lock()
best_model_info = {
    "model": None,
    "poly_features": None,
    "degree": None,
    "score": None
}

def train_polynomial_regression(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    X_poly_test = poly_features.transform(X_test)
    score = model.score(X_poly_test, y_test)
    return degree, model, poly_features, score

@app.post("/train")
def train_models():
    degrees = [2, 3, 4, 5]
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_polynomial_regression, d) for d in degrees]
        for future in concurrent.futures.as_completed(futures):
            try:
                degree, model, poly_features, score = future.result()
                print(f"Trained degree {degree} with R2 score: {score}")
                results.append((degree, model, poly_features, score))
            except Exception as e:
                print(f"Error training degree: {str(e)}")
                return {"error": str(e)}

    if not results:
        return {"error": "All model training failed."}

    best_degree, best_model, best_poly_features, best_score = max(results, key=lambda x: x[3])
    
    with lock:
        best_model_info["model"] = best_model
        best_model_info["poly_features"] = best_poly_features
        best_model_info["degree"] = best_degree
        best_model_info["score"] = best_score

    return {
        "message": "Training completed.",
        "best_degree": best_degree,
        "best_score": round(best_score, 4)
    }


class PredictRequest(BaseModel):
    level: float

@app.post("/predict")
def predict_salary(request: PredictRequest):
    with lock:
        model = best_model_info.get("model")
        poly = best_model_info.get("poly_features")

        if model is None or poly is None:
            return {"error": "Model not trained yet. Call /train first."}

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


@app.get("/")  # Optional: simple homepage
def root():
    return {"message": "Polynomial Regression API is running!"}


"""shit will try again.."""