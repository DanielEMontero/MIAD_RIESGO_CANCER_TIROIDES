from fastapi import FastAPI
import joblib
import numpy as np
import os # Importar 'os'

# Obtener la ruta del directorio actual donde se encuentra app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_tiroides.pkl")

app = FastAPI(
    title="API Predicción Tiroides",
    description="Servicio de inferencia para el modelo de tumor de tiroides",
    version="1.0"
)

# Cargar el modelo usando la ruta absoluta calculada
model = joblib.load(MODEL_PATH) 

@app.post("/predict")
def predict(features: list):
    X = np.array(features).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    return {"probabilidad": float(prob)}
