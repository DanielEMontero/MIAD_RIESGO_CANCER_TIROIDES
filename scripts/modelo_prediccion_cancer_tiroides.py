import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from preparar_datos_utils import preparar_datos
from sklearn.ensemble import GradientBoostingClassifier
import joblib

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

# ENTRENAR el modelo
modelo_entrenado, metricas = modelo.ejecutar_modelo(
    xTrain, yTrain, xTest, yTest, modelGB, "Gradient Boosting",
    imbalanceo=False, graficar=False
)

joblib.dump(modelo_entrenado, "dashboard/modelo_tiroides.pkl")