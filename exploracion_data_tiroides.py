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

# Configuración global para gráficos
plt.rcParams.update({'font.size': 12})

# **Histogramas y piecharts de variables predictoras**
column_names = xTotal.columns
num_columns = len(column_names)
num_rows = int(np.ceil(num_columns / 3))  

fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, col in enumerate(column_names):
    valores_unicos = xTotal[col].nunique()
    
    if valores_unicos <= 8:  # Pie Chart
        data_counts = xTotal[col].value_counts()
        axes[i].pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        titulo = fr"PieChart $\bf{{{col}}}$"
    else:  # Histograma
        axes[i].hist(xTotal[col], bins='auto', alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_xlabel('Valores')
        axes[i].set_ylabel('Frecuencia')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        titulo = fr"Histograma $\bf{{{col}}}$"
    
    axes[i].set_title(titulo, fontsize=12)
    
# Ocultar gráficos vacíos si el número de variables no es múltiplo de 3
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
fig.suptitle("Gráficos Variables Predictoras", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Histogramas y boxplot entre cada variable predictora y la variable de respuesta (Y)
# Filtrar variables numéricas y categóricas
xTotal_numeric = xTotal.select_dtypes(include=['number'])  # Solo numéricas
categorical_cols = xTotal.select_dtypes(include=['object']).columns.tolist()  # Solo categóricas

# Contar total de variables
num_predictors = len(xTotal_numeric.columns) + len(categorical_cols)
num_rows = int(np.ceil(num_predictors / 3))  # Organizar en 3 columnas

# Crear la figura
fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))
axes = axes.flatten()

# Generar gráficos para variables numéricas (Correlogramas)
for i, col in enumerate(xTotal_numeric.columns):
    sns.histplot(data=dfWork, x=col, hue=VarResp, kde=True, bins=30, palette="coolwarm", ax=axes[i])
    axes[i].set_title(f"Histograma: {col} vs {VarResp}", fontsize=12)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel("Frecuencia", fontsize=10)

# Generar gráficos de barras y tablas de frecuencias para variables categóricas
for j, col in enumerate(categorical_cols, start=i + 1):
    sns.countplot(data=dfWork, x=col, hue=VarResp, palette="coolwarm", ax=axes[j])
    axes[j].set_title(f"Distribución de {col} según {VarResp}", fontsize=12)
    axes[j].set_xlabel(col, fontsize=10)
    axes[j].set_ylabel("Frecuencia", fontsize=10)
    axes[j].tick_params(axis='x', rotation=30)  # Rotar etiquetas si hay muchas categorías

# Ocultar gráficos vacíos si el número de variables no es múltiplo de 3
for k in range(j + 1, len(axes)):
    fig.delaxes(axes[k])

plt.suptitle(f"Histogramas y Gráficos de barras de Variables Predictoras vs Variable de Respuesta ({VarResp})", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
