# -*- coding: utf-8 -*-
"""
Módulo: iterar_modelo.py
Autor: Equipo 20 DSA - MIAD
Descripción:
    Clase utilitaria IterarModelo para exploración, preparación de datos,
    selección de variables, manejo de outliers, imputación y evaluación de modelos.
"""

# %% LIBRERIAS
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, wasserstein_distance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve, 
                             precision_score, f1_score, recall_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE, mutual_info_classif, f_classif, SelectKBest

try:
    from IPython.display import display
except ImportError:
    display = print

# %% DEFCLASE
class IterarModelo:
    def __init__(self, file_path, var_objetivo="Diagnosis"):
        """ Carga los datasets y los deja accesibles como atributos """
        self.var_objetivo = var_objetivo
        self.dfTotal = pd.read_csv(file_path)
    
    def explora_ini (self, df):
        # Función para exploracion inicial un set de datos
        
        # Exploración inicial
        print("\n\033[1m Primeras 5 filas del dataset: \033[0m")
        display(df.head())
        
        print("\n\033[1m Información del dataset: \033[0m")
        df.info()
        
        # Informacion de variables numéricas
        print("\n\033[1m Información del dataset: \033[0m")
        display(df.describe())
        
        # Revisar tipos de datos
        print("\n\033[1m Tipos de datos en el dataset: \033[0m")
        print(df.dtypes)
        
    def graf_set(self, df, colsGrid=5, titGraf="Gráficos de Variables"):
        # Función para graficar un set de datos
        nomCol = df.columns
        nCols = len(nomCol)
        nFilas = int(np.ceil(nCols / colsGrid))

        fig, axes = plt.subplots(nFilas, colsGrid, figsize=(5 * colsGrid, 5 * nFilas))
        axes = axes.flatten()

        for i, col in enumerate(nomCol):
            valores_unicos = df[col].nunique()

            if valores_unicos <= 8:  # Pie Chart
                data_counts = df[col].value_counts()
                
                # Verificar si la columna es de tipo 'category' y tiene un orden predefinido
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    data_counts = data_counts.reindex(df[col].cat.categories)  # Reordenar correctamente
                
                axes[i].pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', 
                            colors = fig.cm.Paired.colors, startangle=90, wedgeprops={'edgecolor': 'black'})
                titulo = f"Distribución de {col}"
            elif valores_unicos <= 16:  # Gráfico de Barras
                data_counts = df[col].value_counts()
                data_counts.plot(kind='bar', color='blue', alpha=0.7, edgecolor='black', ax=axes[i])
                axes[i].set_xlabel("Clases")
                axes[i].set_ylabel("Frecuencia")
                axes[i].set_xticklabels(data_counts.index, rotation=75)
                axes[i].grid(axis='y', linestyle='--', alpha=0.6)
                titulo = f"Distribución de {col}"
            else:  # Histograma
                axes[i].hist(df[col].dropna(), bins='auto', alpha=0.7, color='blue', edgecolor='black')
                axes[i].set_xlabel('Valores')
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, linestyle='--', alpha=0.6)
                titulo = f"Histograma de {col}"

            axes[i].set_title(titulo, fontsize=12)

        # Ocultar gráficos vacíos si el número de variables no es múltiplo de colsGrid
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(titGraf, fontsize=18, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        #plt.show()
        plt.close(fig)
        
    def graf_var(self, y, umbral=None):
        # Función para visualizar comportamiento de una variable
        categorias = y.nunique()
        
        titGraf = f'Distribución de la Variable {y.name}'

        if categorias <= 8:
            # Usar Pie Chart solo si existen pocas categorías
            fig = plt.figure(figsize=(6, 6))
            data_counts = y.value_counts()
            fig.pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', colors=["red", "green"])
            fig.title(titGraf, fontsize=16, fontweight='bold')
        elif categorias <= 16:
            # Usar un Gráfico de Barras si existen muchas clases
            fig = plt.figure(figsize=(10, 5))
            y.value_counts().plot(kind='bar', color='blue', alpha=0.7, edgecolor='black')
            fig.title(titGraf, fontsize=16, fontweight='bold')
            fig.xlabel("Clases")
            fig.ylabel("Frecuencia")
            plt.xticks(rotation=75)
            fig.grid(axis='y', linestyle='--', alpha=0.6)
        else:
            # Histograma con umbral opcional
            fig = plt.figure(figsize=(10, 5))
            bins = min(50, categorias)

            # Definir los bins manualmente
            counts, bin_edges = np.histogram(y, bins=bins)

            if umbral is None:
                # Histograma normal si no hay umbral
                fig.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color='blue', edgecolor='black', alpha=0.7)
            else:
                # Dividir las frecuencias en dos grupos
                counts_menor = np.where(bin_edges[:-1] <= umbral, counts, 0)  # Azul (≤ umbral)
                counts_mayor = np.where(bin_edges[:-1] > umbral, counts, 0)   # Rojo (> umbral)

                # Dibujar barras en dos colores con los mismos bins
                fig.bar(bin_edges[:-1], counts_menor, width=np.diff(bin_edges), color='blue', edgecolor='black', alpha=0.7, label="≤ Umbral")
                fig.bar(bin_edges[:-1], counts_mayor, width=np.diff(bin_edges), color='red', edgecolor='black', alpha=0.7, label="> Umbral")

                # Línea divisoria del umbral
                fig.axvline(x=umbral, color='black', linestyle='--', linewidth=2, label=f'Umbral: {umbral}')

            fig.title(titGraf, fontsize=16, fontweight='bold')
            fig.xlabel("Valores de la Variable")
            fig.ylabel("Frecuencia")
            fig.legend()
            fig.grid(axis='y', linestyle='--', alpha=0.6)
            
        #plt.show()
        fig.tight_layout()
        plt.close(fig)
    
    def detectar_columnas_no_numericas(self, df):
        # Función para detectar columnas con valores no numéricos
        for col in df.columns:
            try:
                pd.to_numeric(df[col])  # Intentar conversión
                print(f"La columna '{col}' contiene valores numéricos.")
            except ValueError:
                print(f"La columna '{col}' contiene valores no numéricos.")
                
    def valores_nulos(self, df):
        # Función para identificar valores nulos y graficarlos
        # Calcular cantidad y porcentaje de valores nulos
        valNul = df.isna().sum()
        valNul = valNul[valNul > 0].sort_values(ascending=False)
        
        total_filas = len(df)  # Total de filas en el DataFrame
        pctNul = (valNul / total_filas) * 100  # Convertir a porcentaje  
        
        if not valNul.empty:
            print("\n\033[1m Valores nulos por columna: \033[0m")
            dfvalNul = pd.DataFrame({
                "Columna": valNul.index,
                "Valores Nulos": valNul.values,
                "Porcentaje (%)": pctNul.values
            })
            display(dfvalNul)

            # Gráfico combinado de cantidad y porcentaje de valores nulos
            fig, ax1 = plt.subplots(figsize=(12, 5))

            # Barras: Cantidad de valores nulos
            color = 'red'
            valNul.plot(kind='bar', color=color, alpha=0.7, edgecolor='black', ax=ax1)
            ax1.set_xlabel("Variable")
            ax1.set_ylabel("Cantidad de Valores Nulos", color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
            ax1.grid(axis='y', linestyle='--', alpha=0.6)

            # Línea: Porcentaje de valores nulos
            ax2 = ax1.twinx()
            ax2.plot(pctNul, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=5)
            ax2.set_ylabel("Porcentaje de Datos Nulos (%)", color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            fig.title("Cantidad y Porcentaje de Valores Nulos por Columna", fontsize=14, fontweight='bold')
            #plt.show()
            fig.tight_layout()
            plt.close(fig)
        else:
            print("\n No hay valores nulos en el dataset.")
    
    def detectar_valores_mixtos(self, df):
        # Función para detectar valores mixtos en una columna
        for col in df.columns:
            tipos_unicos = df[col].apply(type).unique()
            if len(tipos_unicos) > 1:
                print(f"La columna '{col}' tiene múltiples tipos de datos: {tipos_unicos}")
            else:
                print(f"La columna '{col}' tiene un único tipo de dato: {tipos_unicos}")

    def detectar_valores_infinitos(self, df):
        # Función para detectar valores infinitos
        valores_inf = (df == np.inf).sum() + (df == -np.inf).sum()
        if valores_inf.sum() > 0:
            print("\nValores infinitos detectados en las siguientes columnas:")
            print(valores_inf[valores_inf > 0])
        else:
            print("\nNo hay valores infinitos en el dataset.")
                       
    def visualizar_outliers(self, df, titGraf="Distribución de outliers"):
        # Función para visualizar los outliers
        # Detectar valores extremos con el Rango Intercuartílico (IQR)
        for col in df.select_dtypes(include=[np.number]).columns:  # Solo columnas numéricas
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            print(f"\n Outliers en '{col}':\n", df[(df[col] < lower_bound) | (df[col] > upper_bound)][col])

        # Revisar estadísticas descriptivas
        print("\n Estadisticas de los datos:\n")
        print(df.describe())

        # Detectar valores extremos con Z-score
        # Aplicar Z-score solo a columnas numéricas
        z_scores = df.select_dtypes(include=[np.number]).apply(zscore)

        # Filtrar valores con Z-score mayor a 3 o menor a -3 (considerados outliers)
        outliers_zscore = (z_scores.abs() > 3)

        # Mostrar cuántos outliers tiene cada columna
        print("\n Valores extremos con Z-score:\n")
        print(outliers_zscore.sum())
        
        
        # Gráficos de Boxplot (para ver outliers)
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(data=df.select_dtypes(include=[np.number]))
        plt.xticks(rotation=90)
        fig.title(titGraf)
        #plt.show()
        fig.tight_layout()
        plt.close(fig)
        
        # Aplicar los criterios de IQR y Z-score
        iqr_mask, z_mask = self.outliers_mask(df)
        
        # Contar cuántos outliers hay en cada variable
        outliers_count = (iqr_mask | z_mask).sum()
        
        # Ordenar de mayor a menor
        outliers_count = outliers_count.sort_values(ascending=False)
        
        # Mostrar las 10 variables con más outliers
        print("\n Variables con más outliers:\n")
        print(outliers_count.head(10))
        
    def outliers_mask (self, df):
        Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        iqr_mask = (df < lower_bound) | (df > upper_bound)
        z_mask = (df.apply(zscore).abs() > 3)
        
        return iqr_mask, z_mask
    
    def eliminar_outliers(self, dfRef, dfComp, tipoMask = 1):
        # Crear máscaras
        Q1, Q3 = dfRef.quantile(0.25), dfRef.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        iqr_mask = (dfRef < lower_bound) | (dfRef > upper_bound)
        z_mask = (dfRef.apply(zscore).abs() > 3)                
        
        if tipoMask == 1:
            dfRef_cleaned = dfRef.mask(iqr_mask | z_mask, np.nan)
            dfComp_cleaned = dfComp.mask(iqr_mask | z_mask, np.nan)
        elif tipoMask == 2:
            dfRef_cleaned = dfRef.mask(iqr_mask | z_mask, np.nan)
            dfComp_cleaned = dfComp.mask(iqr_mask, np.nan)       
            dfComp_cleaned = dfComp.mask((dfComp < lower_bound) | (dfComp > upper_bound), np.nan)
        
        return dfRef_cleaned, dfComp_cleaned

    def ejecutar_imputacion(self, dfRef, dfComp, maxIter):
        # Función para imputar valores faltantes
        imp = IterativeImputer(max_iter=maxIter, random_state=0)
        dfRef_imputed = pd.DataFrame(imp.fit_transform(dfRef), columns=dfRef.columns)
        dfComp_imputed = pd.DataFrame(imp.transform(dfComp), columns=dfComp.columns)

        return dfRef_imputed, dfComp_imputed
    
    def dist_dataframes(self, col, dfRef, dfEval):
        # Función para calcular la distancia de Wasserstein entre dos DataFrames
        scaler = MinMaxScaler()
        
        # Eliminar NaN
        original = dfRef[col].dropna().values
        evaluado = dfEval[col].values

        # Normalizar los valores antes del cálculo
        original = scaler.fit_transform(original.reshape(-1, 1)).flatten()
        evaluado = scaler.transform(evaluado.reshape(-1, 1)).flatten()

        # Calcular distancia de Wasserstein
        return wasserstein_distance(original, evaluado)

    def iterar_imputacion(self, df, min, max, paso):
        """ Encuentra el mejor max_iter basado en la distancia de Wasserstein """
        resultados = []
        for maxIter in range(min, max, paso):
            print(f"\nEvaluando max_iter = {maxIter}...")
            df_imputed, _ = self.ejecutar_imputacion(df, df, maxIter)
            
            # Evaluar la distancia de Wasserstein
            distancias = {col: self.dist_dataframes(col, df, df_imputed) for col in df.columns}
            distancia_prom = np.mean(list(distancias.values()))

            resultados.append({"max_iter": maxIter, "Distancia": distancia_prom})
            print(f"max_iter = {maxIter}, Distancia Promedio: {distancia_prom:.5f}")
        
        # Convertir a DataFrame para mejor visualización
        df_resultados = pd.DataFrame(resultados)
        
        # Graficar la evolución de ambas opciones
        fig = plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_resultados, x="max_iter", y="Distancia", marker="o", color="red", label="Distancia")

        plt.xticks(range(10, 110, 10))
        fig.xlabel("max_iter")
        fig.ylabel("Distancia de Wasserstein")
        fig.title("Evolución de la imputación según max_iter")
        fig.grid(axis="y", linestyle="--", alpha=0.6)
        fig.legend()
        #plt.show()
        fig.tight_layout()
        plt.close(fig)
        
        mejor_max_iter = df_resultados.loc[df_resultados["Distancia"].idxmin(), "max_iter"]
        print(f"\n Mejor max_iter: {mejor_max_iter}")
        
        return mejor_max_iter
    
    def comparar_distribuciones(self, dfRef, dfEval):
        # Función para comparar distribuciones
        """ Comparación de distribuciones con barras lado a lado """
        dfMeanRef = dfRef.mean()
        dfMeanEval = dfEval.mean()
        variables = dfMeanRef.index
        x = np.arange(len(variables))  # Posiciones en el eje X

        width = 0.4  # Ancho de las barras

        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Barras originales (gris)
        ax.bar(x - width/2, dfMeanRef, width, color="gray", alpha=0.7, label="Original", edgecolor='black')

        # Barras imputadas
        ax.bar(x + width/2, dfMeanEval, width, color="blue", alpha=0.7, label="Evaluado", edgecolor='black')

        ax.set_xlabel("Variables")
        ax.set_ylabel("Frecuencia Promedio")
        ax.set_title("Comparación de Distribuciones")
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=90)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        #plt.show()
        fig.tight_layout()
        plt.close(fig)
        
    def visualizar_var (self, df, varSet):
        for var in varSet:
            fig = plt.figure(figsize=(12, 5))
            
            # Histograma
            fig.subplot(1, 2, 1)
            sns.histplot(df[var], bins=50, kde=True, color='blue')
            fig.title(f"Histograma de {var}")
            
            # Boxplot
            fig.subplot(1, 2, 2)
            sns.boxplot(x=df[var], color='red')
            fig.title(f"Boxplot de {var}")
            
            #plt.show()
            fig.tight_layout()
            plt.close(fig)

            # Ver si hay relación con otras variables
            correlaciones = df.corr()[var].sort_values(ascending=False)
            print(f"\n Variables con mas correlación para {var}:\n")
            print(correlaciones.head(10))  # Top 10 variables más correlacionadas
            
    def transformar_var (self, df, varSet, tipoTransf=1):
        for var in varSet:
            if tipoTransf == 1:          
                # Aplicar Winzorizacion
                df[var] = winsorize(df[var], limits=[0.01, 0.01])
            elif tipoTransf == 2:
                # Transformación logarítmica
                df[var] = np.log1p(df[var])  # log(x + 1) para evitar log(0)
            elif tipoTransf == 3:
                # Aplicar Winzorizacion
                df[var] = winsorize(df[var], limits=[0.01, 0.01])
                # Transformación logarítmica
                df[var] = np.log1p(df[var])  # log(x + 1) para evitar log(0)
        
        return df

    def normalizar_datos(self, dfRef, dfComp, varSet):
        """ Aplica normalización y escalado a las variables numéricas """
        cols_a_normalizar = [col for col in dfRef.columns if col not in varSet]
        scaler = StandardScaler()
        scaler_minmax = MinMaxScaler()
        
        dfRef_scaled = dfRef.copy()
        dfComp_scaled = dfComp.copy()
        
        # Normalización con StandardScaler
        dfRef_scaled[cols_a_normalizar] = pd.DataFrame(
            scaler.fit_transform(dfRef[cols_a_normalizar]),
            columns=cols_a_normalizar,
            index=dfRef.index)
        
        dfComp_scaled[cols_a_normalizar] = pd.DataFrame(
        scaler.transform(dfComp[cols_a_normalizar]), 
        columns=cols_a_normalizar, 
        index=dfComp.index)

        dfRef_scaled[varSet] = scaler_minmax.fit_transform(dfRef_scaled[varSet])
        dfComp_scaled[varSet] = scaler_minmax.transform(dfComp_scaled[varSet])
        
        return dfRef_scaled, dfComp_scaled
        
    def importancia_var_modelo (self, xSet_Orig, ySet, xSet_mod):
        # Entrenar un modelo de Random Forest para evaluar la importancia de las variables
        model = RandomForestClassifier(random_state=0)
        model.fit(xSet_Orig, ySet)  # Entrenar el modelo con xSet_Orig antes de la transformación

        # Obtener la importancia de las variables
        feature_importance = pd.Series(model.feature_importances_, index=xSet_Orig.columns).sort_values(ascending=False)

        # Entrenar el modelo con xSet_Mod después de la transformación
        model.fit(xSet_mod, ySet)

        # Obtener la nueva importancia de las variables
        feature_importance_transformed = pd.Series(model.feature_importances_, index=xSet_Orig.columns).sort_values(ascending=False)

        # Mostrar la comparación entre el dataset original y el transformado
        comparison = pd.DataFrame({
            "Antes": feature_importance,  
            "Después": feature_importance_transformed
        })

        # Ordenar por importancia en el dataset original
        comparison = comparison.sort_values("Antes", ascending=False)

        print("Comparación de importancia de variables antes y después de la transformación:")
        print(comparison.head(10))

        # Graficar la comparación
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 6))
        comparison.head(10).plot(kind="bar", figsize=(12, 6), color=["blue", "red"], alpha=0.7, edgecolor="black")
        fig.title("Comparación de Importancia de Variables (Antes vs. Después de la Transformación)")
        fig.xlabel("Variables")
        fig.ylabel("Importancia")
        plt.xticks(rotation=75)
        fig.legend(["Antes", "Después"])
        fig.grid(axis='y', linestyle='--', alpha=0.6)
        #plt.show()
        fig.tight_layout()
        plt.close(fig)
        
    def separar_datos (self, xSet, ySet, porcTest):
        # División en entrenamiento y prueba
        xSet_Train, xSet_Test, ySet_Train, ySet_Test = train_test_split(xSet, ySet, test_size=porcTest, random_state=0)
        
        return xSet_Train, xSet_Test, ySet_Train, ySet_Test
    
    def imbalanceo (self, xSet, ySet, k=5, graf=True):
        sm = SMOTE(random_state=10, k_neighbors=k) # k_neighbors=5 default
        xSet_res, ySet_res = sm.fit_resample(xSet, ySet)
        
        if graf:
            ySet_graf = pd.Series(ySet)
            ySet_res_graf = pd.Series(ySet_res)
            # Contar las observaciones por clase antes y después del balanceo
            clases = sorted(ySet_graf.unique())  # Asegurar orden correcto
            conteo_antes = ySet_graf.value_counts().sort_index()
            conteo_despues = ySet_res_graf.value_counts().sort_index()
    
            # Crear gráfico de barras agrupadas
            fig, ax = plt.subplots(figsize=(8, 5))
            width = 0.35  # Ancho de las barras
            x = np.arange(len(clases))  # Posiciones en el eje X
    
            bars1 = ax.bar(x - width/2, conteo_antes, width, label="Antes del balanceo", color="#1f77b4")  # Azul
            bars2 = ax.bar(x + width/2, conteo_despues, width, label="Después del balanceo (SMOTE)", color="#ff7f0e")  # Naranja
    
            # Agregar etiquetas de valores sobre las barras
            for bars in [bars1, bars2]:  
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.0f}', ha='center', va='bottom', fontsize=10)
    
            # Personalizar gráfico
            ax.set_title("Comparación de clases antes y después del balanceo", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(clases)
            ax.set_xlabel("Clase")
            ax.set_ylabel("Número de Observaciones")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
    
            # Mostrar gráfico
            #plt.show()
            fig.tight_layout()
            plt.close(fig)
            
            # Comparar cambio
            fig = plt.figure(figsize=(10, 5))
            sns.kdeplot(xSet.iloc[:, 0], label="Original", fill=True, alpha=0.5)
            sns.kdeplot(xSet_res.iloc[:, 0], label="SMOTE", fill=True, alpha=0.5)
            fig.title("Comparación de distribución antes y después de SMOTE")
            fig.legend()
            #plt.show()
            fig.tight_layout()
            plt.close(fig)
        
        return xSet_res, ySet_res
        
    def eval_model_clasif_test(self, model, model_name, xSet_Train, ySet_Train, xSet_Test, ySet_Test, graficar=True):
        # Función para entrenar y evaluar modelos en problemas multiclase

        # 1. Entrenar el modelo
        model.fit(xSet_Train, ySet_Train)

        # 2. Realizar predicciones
        yPred = model.predict(xSet_Test)
        
        # 3. Verificar si el modelo tiene `predict_proba()` o usar `decision_function()`
        if hasattr(model, "predict_proba"):  # Modelos con probabilidades
            yPredProb = model.predict_proba(xSet_Test)[:, 1]  # Probabilidad de la clase positiva
        elif hasattr(model, "decision_function"):  # Modelos sin `predict_proba()`
            yPredProb = model.decision_function(xSet_Test)
            # Escalar valores de `decision_function()` entre 0 y 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            if yPredProb.ndim == 1:
                yPredProb = scaler.fit_transform(yPredProb.reshape(-1, 1)).ravel()
            else:
                yPredProb = scaler.fit_transform(yPredProb)
        else:
            yPredProb = None  # Si no hay forma de calcular probabilidades
        
        # Verificar si hay valores NaN en las probabilidades predichas
        if np.isnan(yPredProb).any():
            print("Advertencia: Probabilidades predichas contienen valores NaN")
            yPredProb = np.nan_to_num(yPredProb) # Reemplazar NaN con 0
        
        # 4. Determinar si es un problema **binario** o **multiclase**
        clases_unicas = np.unique(ySet_Train)
        es_binario = len(clases_unicas) == 2  # Solo dos clases → binario

        # 5. Evaluar el desempeño
        accuracy = accuracy_score(ySet_Test, yPred) 
        precision = precision_score(ySet_Test, yPred, average="weighted")  # Promedio ponderado
        f1 = f1_score(ySet_Test, yPred, average="weighted")
        recall = recall_score(ySet_Test, yPred)
        conf_matrix = confusion_matrix(ySet_Test, yPred)

        # 6. Convertir matriz de confusión a DataFrame
        df_conf_matrix = pd.DataFrame(conf_matrix, 
                                      index=[f"Real {cls}" for cls in clases_unicas], 
                                      columns=[f"Pred {cls}" for cls in clases_unicas])
        
        if graficar:
            # 7. Graficar matriz de confusión
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(df_conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
            fig.xlabel("Predicción")
            fig.ylabel("Realidad")
            fig.title(f"Matriz de Confusión - {model_name}")
            #plt.show()
            fig.tight_layout()
            fig.savefig(f"{model_name}_confusion.png")
            try:
                mlflow.log_artifact(f"{model_name}_confusion.png")
            except Exception as e:
                print(f"[MLflow] No se pudo registrar artefacto ({model_name}): {e}")
            plt.close(fig)

        # 8. Cálculo del AUC y la curva ROC
        if es_binario:
            fpr, tpr, _ = roc_curve(ySet_Test, yPredProb)
            auc_macro = auc_weighted = roc_auc_score(ySet_Test, yPredProb)
        else:
            yTestBinarized = label_binarize(ySet_Test, classes=clases_unicas)
            auc_macro = roc_auc_score(yTestBinarized, yPredProb, multi_class="ovr", average="macro")
            auc_weighted = roc_auc_score(yTestBinarized, yPredProb, multi_class="ovr", average="weighted")

            # Graficar curva ROC para multiclase
            fpr, tpr, _ = roc_curve(yTestBinarized.ravel(), yPredProb.ravel())
        
        # 9. Mostrar resultados
        print(f"\n\033[1m **Desempeño del modelo {model_name}** \033[0m")
        print(f" Exactitud modelo - {model_name}: \n\033[1m {accuracy:.4f} \033[0m")
        print(f" Precisión promedio ponderada - {model_name}: \n\033[1m {precision:.4f} \033[0m")
        print(f" Recall modelo - {model_name}: \n\033[1m {recall:.4f} \033[0m")
        print(f" F1-score ponderado - {model_name}: \033[1m{f1:.4f}\033[0m")
        
        if es_binario:
            print(f" AUC-ROC - {model_name}: \n\033[1m {auc_macro:.4f} \033[0m")
        else:
            print(f"\n AUC Macro (Promedio de clases) - {model_name}: \n\033[1m {auc_macro:.4f} \033[0m")
            print(f" AUC Weighted (Ponderado por frecuencia de clases) - {model_name}: \n\033[1m {auc_weighted:.4f} \033[0m")
        
        # 10. Presentar matriz de confusión
        print(f"\n\033[1m Matriz de Confusión modelo - {model_name}: \033[0m")
        display(df_conf_matrix)
        
        if graficar:
            # 11. Graficar curva ROC
            fig = plt.figure(figsize=(6, 5))
            fig.plot(fpr, tpr, 'b', label=f'AUC = {auc_macro:.2f}')
            fig.legend(loc='lower right')
            fig.plot([0, 1], [0, 1], 'r--')  # Línea de referencia (azar)
            fig.xlim([-0.01, 1])
            fig.ylim([0, 1.05])
            fig.ylabel('Tasa de Verdaderos Positivos (TPR)')
            fig.xlabel('Tasa de Falsos Positivos (FPR)')
            fig.title(f'Curva ROC - {model_name}')
            fig.grid()
            #plt.show()
            fig.tight_layout()
            fig.savefig(f"{model_name}_ROC.png")
            try:
                mlflow.log_artifact(f"{model_name}_ROC.png")
            except Exception as e:
                print(f"[MLflow] No se pudo registrar artefacto ({model_name}): {e}")
            plt.close(fig)
        
        # 12. Graficar métricas
        metricas = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1, "AUC": auc_macro}
        
        if graficar:
            fig = plt.figure(figsize=(8, 6))
            bars = plt.bar(metricas.keys(), metricas.values(), color=["blue", "orange", "green", "red", "purple"])
            
            # Agregar etiquetas de valores sobre cada barra
            for bar in bars:
                height = bar.get_height()
                fig.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}', 
                         ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Personalizar gráfico
            fig.ylim(0, 1)  # Rango de 0 a 1 porque son métricas de clasificación
            fig.xlabel("Métrica")
            fig.ylabel("Valor")
            fig.title(f"Métricas de desempeño - {model_name}")
            fig.grid(axis='y', linestyle='--', alpha=0.6)
            
            #plt.show()
            fig.tight_layout()
            fig.savefig(f"{model_name}_desempenio.png")
            try:
                mlflow.log_artifact(f"{model_name}_desempenio.png")
            except Exception as e:
                print(f"[MLflow] No se pudo registrar artefacto ({model_name}): {e}")
            
            plt.close(fig)
        
        return model, metricas
        
    def afinar_modelo(self, model, xSet, ySet, scoring="roc_auc"):
        # Configurar validación cruzada con 5 folds estratificados
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Evaluar AUC usando Cross-Validation
        auc_scoresRF = cross_val_score(model, xSet, ySet, cv=cv, scoring="roc_auc")

        print(f"\n AUC estimado en CV: {auc_scoresRF.mean():.4f} ± {auc_scoresRF.std():.4f}")
        
    def ejecutar_modelo (self, xSet_Train, ySet_Train, xSet_Test, ySet_Test, model, modelName, imbalanceo=True, graficar=True):
        if imbalanceo: 
            xSet_res, ySet_res = self.imbalanceo(xSet_Train, ySet_Train, 5)
        else:
            xSet_res, ySet_res = xSet_Train.copy(), ySet_Train.copy()
        
        resMod = self.eval_model_clasif_test(
            model, modelName, xSet_res, ySet_res, xSet_Test, ySet_Test, graficar
            )
            
        return resMod
    
    def seleccionar_variables (self, xSet, ySet, numSel, tipoSel=1):
        if tipoSel == 1:
            # Seleccionar con RFE
            # 1. Definir el modelo base (Random Forest)
            estimator = RandomForestClassifier(random_state=42)  #se usa random Forest como el algoritmo predictivo

            # 2. Aplicar RFE para seleccionar numSel mejores características
            selector = RFE(estimator, n_features_to_select=numSel, step=5) #Se define step=1, pero lo puede cambiar para que sea más rápido

            # 3. Ajustar el selector a los datos
            selector = selector.fit(xSet, ySet) # esto puede tardar algunos minutos

            # Obtener las variables seleccionadas
            selector.support_  #Conjunto de variables seleccionadas
            varSel = xSet.columns[selector.support_]
            
        elif tipoSel == 2:
            # Selector mixto
            # Seleccionar con información mututa
            selector_mi = SelectKBest(mutual_info_classif, k=numSel).fit(xSet, ySet)
            varSel_MI = xSet.columns[selector_mi.get_support()]
            print("Variables seleccionadas (MI):", varSel_MI.tolist())

            # Seleccionar con ANOVA
            selector_f = SelectKBest(f_classif, k=numSel).fit(xSet, ySet)
            varSel_F = xSet.columns[selector_f.get_support()]
            print("Variables seleccionadas (F-Classif):", varSel_F.tolist())

            # Combinar selección de variables
            varSel = list(set(varSel_MI) | set(varSel_F))
            print("Variables seleccionadas en ambas técnicas:", varSel)
            
        return varSel