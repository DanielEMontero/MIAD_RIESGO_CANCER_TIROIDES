# -*- coding: utf-8 -*-
"""
Módulo utilitario de preparación de datos
Autor: Equipo DSA - MIAD

Función: preparar_datos()
    Realiza la preparación completa:
    Exploración → Limpieza → Traducción → Codificación → Selección.
"""

# %% LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder

#from src.iterar_modelo import IterarModelo
from iterar_modelo import IterarModelo

try:
    from IPython.display import display
except ImportError:
    display = print

def preparar_datos(file_path, var_objetivo, graficar=True):
    modelo = IterarModelo(file_path, var_objetivo)
    
    # CARGAR Y EXPLORAR
    dfTotal = modelo.dfTotal.copy()                     # Extraer los dataframe
    modelo.explora_ini(dfTotal)                         # Exploracion inicial
    modelo.detectar_columnas_no_numericas(dfTotal)      # Detectar columnas con valores no numéricos
    modelo.valores_nulos(dfTotal)                       # Verificar valores nulos
    modelo.detectar_valores_mixtos(dfTotal)             # Detectar valores mixtos en columnas
    modelo.detectar_valores_infinitos(dfTotal)          # Buscar valores especiales (NaN, None, inf, -inf)
    
    # TRADUCCIÓN DE NOMBRES Y CATEGORÍAS
    nombres_espaniol = {
        "Age": "Edad",
        "Gender": "Genero",
        "Country": "Pais",
        "Ethnicity": "Etnicidad",
        "Family_History": "Historial_Familiar",
        "Radiation_Exposure": "Exposicion_Radiacion",
        "Iodine_Deficiency": "Deficiencia_Yodo",
        "Smoking": "Tabaquismo",
        "Obesity": "Obesidad",
        "Diabetes": "Diabetes",
        "TSH_Level": "Nivel_TSH",
        "T3_Level": "Nivel_T3",
        "T4_Level": "Nivel_T4",
        "Nodule_Size": "Tamanio_Nodo",
        "Thyroid_Cancer_Risk": "Riesgo_Cancer_Tiroides",
        "Diagnosis": "Diagnostico"
    }
    
    categorias_espanol = {
        "Genero": {"Male": "Masculino", "Female": "Femenino"},
        "Pais": {
            "Russia": "Rusia", "Germany": "Alemania", "Nigeria": "Nigeria", "India": "India",
            "UK": "Reino Unido", "South Korea": "Corea del Sur", "Brazil": "Brasil",
            "China": "China", "Japan": "Japon", "USA": "EEUU"
        },
        "Etnicidad": {
            "Caucasian": "Caucasico", "Hispanic": "Hispano", "Asian": "Asiatico",
            "African": "Africano", "Middle Eastern": "Oriente Medio"
        },
        "Historial_Familiar": {"No": "No", "Yes": "Si"},
        "Exposicion_Radiacion": {"No": "No", "Yes": "Si"},
        "Deficiencia_Yodo": {"No": "No", "Yes": "Si"},
        "Tabaquismo": {"No": "No", "Yes": "Si"},
        "Obesidad": {"No": "No", "Yes": "Si"},
        "Diabetes": {"No": "No", "Yes": "Si"},
        "Riesgo_Cancer_Tiroides": {"Low": "Bajo", "Medium": "Medio", "High": "Alto"},
        "Diagnostico": {"Benign": "Benigno", "Malignant": "Maligno"}
    }
    
    dfTotal.rename(columns=nombres_espaniol, inplace=True)      # Renombrar las columnas en el DataFrame
    var_objetivo = "Diagnostico"
    modelo.var_objetivo = var_objetivo
    
    for col, mapping in categorias_espanol.items():             # Aplicar la traducción en el DataFrame
        if col in dfTotal.columns:
            dfTotal[col] = dfTotal[col].replace(mapping)
    
    modelo.explora_ini(dfTotal)                     # Exploracion inicial
    
    # LIMPIEZA BÁSICA
    dfTotal.dropna(subset=[var_objetivo], inplace=True)     # Eliminar filas con valores nulos en la variable objetivo
    dfTotal = dfTotal.loc[:, dfTotal.nunique() > 1]         # Eliminación de columnas con valores únicos
    if "Patient_ID" in dfTotal.columns:
        dfTotal.drop(columns=["Patient_ID"], inplace=True)  # Eliminacion de varaibles innecesarias
    
    # SEPARAR Y CODIFICAR
    xTotal = dfTotal.drop(columns=[var_objetivo])
    yTotal = dfTotal[var_objetivo]
    xTrain, xTest, yTrain, yTest = modelo.separar_datos(xTotal, yTotal, 0.2)
    
    # Visualizar comportamiento de los datos
    if graficar:
        _explorar_datos(modelo, dfTotal, xTrain, yTrain, var_objetivo)
    
    # Filtrar variables numéricas y categóricas y ver comportamiento
    xTrain_numeric = xTrain.select_dtypes(include=['number'])  # Solo numéricas
    xTest_numeric = xTest.select_dtypes(include=['number'])  # Solo numéricas
    categorical_cols = xTrain.select_dtypes(include=['object']).columns.tolist()  # Solo categóricas
    modelo.visualizar_outliers(xTrain_numeric, "Outliers Originales")
    
    # Normalizar variables
    scaler = StandardScaler()
    xTrain_nscaled = xTrain_numeric.copy()
    xTest_nscaled = xTest_numeric.copy()
    xTrain_scaled = xTrain.copy()
    xTest_scaled = xTest.copy()
    
    # Redondear edad antes de normalizar (mantenerla como entera real)
    if 'Edad' in xTrain_numeric.columns:
        xTrain_numeric['Edad'] = xTrain_numeric['Edad'].round().astype('int64')
        xTest_numeric['Edad'] = xTest_numeric['Edad'].round().astype('int64')
    
    # Normalización con StandardScaler
    xTrain_nscaled = pd.DataFrame(
        scaler.fit_transform(xTrain_numeric),
        columns=xTrain_numeric.columns,
        index=xTrain.index)

    xTest_nscaled = pd.DataFrame(
        scaler.transform(xTest_numeric),
        columns=xTest_numeric.columns,
        index=xTest.index)

    modelo.visualizar_outliers(xTrain_nscaled, "Outliers despues de normalización")

    # Actualizar los dataframe
    xTrain_scaled.update(xTrain_nscaled)
    xTest_scaled.update(xTest_nscaled)

    # Codificar variables
    label_encoding_cols = ["Riesgo_Cancer_Tiroides"]        # Separar la variable que requiere Label Encoding
    # Variables que usarán One-Hot Encoding
    one_hot_encoding_cols = [col for col in categorical_cols if col not in label_encoding_cols]

    # Aplicar Label Encoding a "Riesgo_Cancer_Tiroides"
    xTrain_enc = xTrain_scaled.copy()
    xTest_enc = xTest_scaled.copy()

    le = LabelEncoder()
    xTrain_enc["Riesgo_Cancer_Tiroides"] = le.fit_transform(xTrain_enc["Riesgo_Cancer_Tiroides"])
    xTest_enc["Riesgo_Cancer_Tiroides"] = le.transform(xTest_enc["Riesgo_Cancer_Tiroides"])

    # Aplicar One-Hot Encoding al resto de variables categóricas
    xTrain_enc = pd.get_dummies(xTrain_enc, columns=one_hot_encoding_cols, drop_first=True)
    xTest_enc = pd.get_dummies(xTest_enc, columns=one_hot_encoding_cols, drop_first=True)
    xTest_enc = xTest_enc.reindex(columns=xTrain_enc.columns, fill_value=0)

    # Ajustar y transformar la variable de respuesta
    ylabel_encoder = LabelEncoder()
    yTrain_enc = ylabel_encoder.fit_transform(yTrain)
    yTest_enc = ylabel_encoder.transform(yTest)
    
    # SELECCIÓN DE VARIABLES
    varSel = modelo.seleccionar_variables(xTrain_enc, yTrain_enc, numSel=15, tipoSel=1)
    
    #  Filtrar solo las variables seleccionadas
    xTrain_sel = xTrain_enc[varSel]
    xTest_sel = xTest_enc[varSel]
    print(f"✅ Preparación completa. {len(varSel)} variables seleccionadas.")
    
    return modelo, xTrain_sel, xTest_sel, yTrain_enc, yTest_enc
  
def _explorar_datos(modelo, dfTotal, xTrain, yTrain, var_objetivo):
    modelo.graf_set(xTrain, colsGrid=4, titGraf="Gráficos de Variables Predictoras")
    modelo.graf_var(yTrain)

    # Filtrar variables numéricas y categóricas
    dfTotal_numeric = dfTotal.select_dtypes(include=['number'])  # Solo numéricas
    categorical_cols = dfTotal.select_dtypes(include=['object']).columns.tolist()  # Solo categóricas

    # Configuración global para gráficos
    plt.rcParams.update({'font.size': 12})
    col_per_row = 4
    num_predictors = len(dfTotal_numeric.columns) + len(categorical_cols)
    num_rows = int(np.ceil(num_predictors / col_per_row))
    
    plt.close('all')
    plt.suptitle(f"Histogramas y Gráficos de barras de Variables Predictoras vs Variable de Respuesta ({var_objetivo})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Crear la figura
    fig, axes = plt.subplots(num_rows, col_per_row, figsize=(24, 5 * num_rows))
    axes = axes.flatten()

    # Generar gráficos para variables numéricas (Correlogramas)
    for i, col in enumerate(dfTotal_numeric.columns):
        sns.histplot(data=dfTotal, x=col, hue=var_objetivo, kde=True, bins=30, palette="coolwarm", ax=axes[i])
        axes[i].set_title(f"Histograma: {col} vs {var_objetivo}", fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Frecuencia", fontsize=10)

    # Generar gráficos de barras con porcentajes de Maligno/Benigno en cada categoría
    for j, col in enumerate(categorical_cols, start=i + 1):    
        # Crear tabla de frecuencias normalizada para obtener porcentajes
        tab_freq = pd.crosstab(dfTotal[col], dfTotal[var_objetivo], normalize="index") * 100  # Convertir a porcentaje
        
        # Reindexar si la variable es categoría ordenada
        orden = dfTotal[col].cat.categories if isinstance(dfTotal[col].dtype, pd.CategoricalDtype) else None
        
        # Graficar barras apiladas con porcentajes
        tab_freq.plot(kind="bar", stacked=True, colormap="coolwarm", ax=axes[j])
        
        # Ajustes visuales
        axes[j].set_title(f"Distribución de {col} según {var_objetivo}", fontsize=12)
        axes[j].set_xlabel(col, fontsize=10)
        axes[j].set_ylabel("Porcentaje", fontsize=10)
        axes[j].set_xticklabels(tab_freq.index, rotation=30)
        axes[j].legend(title=var_objetivo)

    # Ocultar gráficos vacíos si el número de variables no es múltiplo
    for k in range(j + 1, len(axes)):
        fig.delaxes(axes[k])