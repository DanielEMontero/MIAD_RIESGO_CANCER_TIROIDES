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
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from preparar_datos_utils import preparar_datos
from sklearn.ensemble import RandomForestClassifier

import os



# Preparar datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "thyroid_cancer_risk_data.csv")
var_objetivo = "Diagnosis"
modelo, xTrain, xTest, yTrain, yTest = preparar_datos(file_path, var_objetivo, graficar=False)

# Definir modelo
modelRF = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    random_state=0
)

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ajusta si usas otro servidor
experiment = mlflow.set_experiment("thyroid_classification")

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="RandomForest_15vars"):

    # Entrenar y evaluar
    modelo_entrenado, metricas = modelo.ejecutar_modelo(
        xTrain, yTrain, xTest, yTest, modelRF, "Random Forest", 
        imbalanceo=False, graficar=False
    )

    # Registrar parámetros
    mlflow.log_params(modelRF.get_params())

    # Registrar métricas
    for nombre, valor in metricas.items():
        mlflow.log_metric(nombre, valor)

    # Registrar modelo
    mlflow.sklearn.log_model(modelo_entrenado, "random-forest-model")

    # Registrar artefacto visual (heatmap de métricas)
    df_metrics = pd.DataFrame(metricas, index=["RandomForest"])
    plt.figure(figsize=(6, 2))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Métricas Random Forest")
    plt.tight_layout()
    plt.savefig("metricas_rf.png")
    mlflow.log_artifact("metricas_rf.png")

    print("\n✅ Modelo y métricas registrados en MLflow con éxito.")
