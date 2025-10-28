import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


# Cargar el dataset
file_path = "thyroid_cancer_risk_data.csv"
dfWork = pd.read_csv(file_path)

# Exploración inicial
print("\n\033[1m Primeras 5 filas del dataset: \033[0m")
display(pd.DataFrame(dfWork.head()))

print("\n\033[1m Información del dataset: \033[0m")
dfWork.info()

# Eliminar columnas innecesarias
dfWork.drop(columns=["Patient_ID"], inplace=True)

# Separar variables predictoras y la variable de respuesta
VarResp = "Diagnosis"
xTotal = dfWork.drop(columns=[VarResp])  # Todas las columnas excepto 'Direction'
yTotal = dfWork[VarResp]  # Variable de respuesta

