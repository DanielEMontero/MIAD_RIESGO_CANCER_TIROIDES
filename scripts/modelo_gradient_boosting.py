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
from sklearn.ensemble import GradientBoostingClassifier

# Preparar datos
file_path = "thyroid_cancer_risk_data.csv"
var_objetivo = "Diagnosis"
modelo, xTrain, xTest, yTrain, yTest = preparar_datos(file_path, var_objetivo, graficar=False)

# Definir modelo
modelGB = GradientBoostingClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 5,
    random_state = 0
)

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment = mlflow.set_experiment("thyroid_classification")

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="GradientBoosting_15vars"):
    modelo_entrenado, metricas = modelo.ejecutar_modelo(
        xTrain, yTrain, xTest, yTest, modelGB, "Gradient Boosting", 
        imbalanceo=False, graficar=False
    )

    mlflow.log_params(modelGB.get_params())
    for nombre, valor in metricas.items():
        mlflow.log_metric(nombre, valor)
    mlflow.sklearn.log_model(modelo_entrenado, "gradient-boosting-model")

    df_metrics = pd.DataFrame(metricas, index=["GradientBoosting"])
    plt.figure(figsize=(6, 2))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Métricas Gradient Boosting")
    plt.tight_layout()
    plt.savefig("metricas_gb.png")
    mlflow.log_artifact("metricas_gb.png")

    print("\n✅ Modelo Gradient Boosting registrado en MLflow.")


print("\n\033[1m🔹 Gradient Boosting con selección de variables (RFE): \033[0m")