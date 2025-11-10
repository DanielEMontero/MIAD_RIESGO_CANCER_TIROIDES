# -*- coding: utf-8 -*-
"""
Entrenamiento y registro de modelo con MLflow
Autor: Equipo DSA - MIAD
Nota:
 - Usa la clase IterarModelo para split y utilidades generales.
 - Preprocesa categóricas con one-hot encoding genérico.
 - Registra parámetros, métricas y tiempos en MLflow.
"""

import mlflow
import mlflow.xgboost
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from preparar_datos_utils import preparar_datos

import os

# Preparar datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "thyroid_cancer_risk_data.csv")
var_objetivo = "Diagnosis"
modelo, xTrain, xTest, yTrain, yTest = preparar_datos(file_path, var_objetivo, graficar=False)

# Definir modelo
xgb_model = xgb.XGBClassifier(
    n_estimators = 300,
    learning_rate = 0.1,
    max_depth = 5,
    subsample = 1.0,
    colsample_bytree = 0.8,
    random_state = 0
)

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment = mlflow.set_experiment("thyroid_classification")

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="XGBoost_15vars"):
    modelo_entrenado, metricas = modelo.ejecutar_modelo(
        xTrain, yTrain, xTest, yTest, xgb_model, "Extreme Gradient Boosting XGBoost", 
        imbalanceo=False, graficar=False
    )

    mlflow.log_params(xgb_model.get_params())
    for nombre, valor in metricas.items():
        mlflow.log_metric(nombre, valor)
    mlflow.xgboost.log_model(modelo_entrenado, "xgboost-model")

    df_metrics = pd.DataFrame(metricas, index=["XGBoost"])
    plt.figure(figsize=(6, 2))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="Blues")
    plt.title("Métricas XGBoost")
    plt.tight_layout()
    plt.savefig("metricas_xgb.png")
    mlflow.log_artifact("metricas_xgb.png")

    print("\n✅ Modelo XGBoost registrado en MLflow.")
