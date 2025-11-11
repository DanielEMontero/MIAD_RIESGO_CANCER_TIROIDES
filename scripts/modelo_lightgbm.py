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
import mlflow.lightgbm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb

from preparar_datos_utils import preparar_datos

# Preparar datos
file_path = "thyroid_cancer_risk_data.csv"
var_objetivo = "Diagnosis"
modelo, xTrain, xTest, yTrain, yTest = preparar_datos(file_path, var_objetivo, graficar=False)

# Definir modelo
lgbm_model = lgb.LGBMClassifier(
    n_estimators = 300,
    learning_rate = 0.05,
    num_leaves = 31,
    subsample = 0.9,
    colsample_bytree = 0.9,
    random_state = 0
)

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment = mlflow.set_experiment("thyroid_classification")

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="LightGBM_15vars"):
    modelo_entrenado, metricas = modelo.ejecutar_modelo(
        xTrain, yTrain, xTest, yTest, lgbm_model, "LightGBM", 
        imbalanceo=False, graficar=False
    )

    mlflow.log_params(lgbm_model.get_params())
    for nombre, valor in metricas.items():
        mlflow.log_metric(nombre, valor)
    mlflow.lightgbm.log_model(modelo_entrenado, "lightgbm-model")

    df_metrics = pd.DataFrame(metricas, index=["LightGBM"])
    plt.figure(figsize=(6, 2))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="Greens")
    plt.title("Métricas LightGBM")
    plt.tight_layout()
    plt.savefig("metricas_lgbm.png")
    mlflow.log_artifact("metricas_lgbm.png")

    print("\n✅ Modelo LightGBM registrado en MLflow.")