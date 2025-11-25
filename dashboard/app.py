from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(
    title="API Predicción Tiroides",
    description="Servicio de inferencia para el modelo de tumor de tiroides",
    version="1.0"
)

model = joblib.load("modelo_tiroides.pkl")

@app.post("/predict")
def predict(features: list):
    X = np.array(features).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    return {"probabilidad": float(prob)}