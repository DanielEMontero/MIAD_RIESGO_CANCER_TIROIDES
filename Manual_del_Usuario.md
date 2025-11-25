
# Riesgo de Cáncer de Tiroides  
## Manual del Usuario

**Autores:**  
Daniel Eduardo Montero Ramírez  
Óscar Javier Sánchez Ruiz  
Clara Elvira Sierra Ossorio  
Diego Alejandro Arias Díaz  

**Universidad de los Andes**  
Facultad de Ingeniería  
Maestría en Inteligencia Analítica de Datos  
Despliegue de Soluciones de Analítica  
Colombia — Noviembre 2025  

____________________________________________________________________________________

## 1. Objetivo

Este manual tiene como objetivo describir de manera clara y estructurada el procedimiento para estimar el riesgo de cáncer de tiroides a partir de un conjunto de datos clínicos, demográficos y paraclínicos; utilizando un modelo Gradient Boosting. En este documento se indican los pasos necesarios para ingresar la información requerida, interpretar los resultados obtenidos y comprender su funcionamiento. 

El resultado obtenido constituye una herramienta de apoyo para la valoración clínica, pero es importante mencionar que no es un método diagnóstico definitivo ni garantiza una precisión del 100%. El resultado obtenido debe interpretarse con criterio médico y siempre en combinación con la evaluación clínica completa, otros exámenes y el juicio profesional del médico tratante. 

___________________________________________________________________________________

## 2.	A quien va dirigido 

Este modelo está dirigido principalmente al personal de salud, incluyendo médicos generales, especialistas, residentes, profesionales de enfermería y demás integrantes del equipo asistencial que participan en la valoración y seguimiento de pacientes con patología tiroidea. 

Aunque el contenido puede ser consultado por pacientes o público general interesado, se aclara que la interpretación de los resultados y cualquier decisión diagnóstica o terapéutica debe ser realizada exclusivamente por un médico tratante. La información aquí presentada no reemplaza la evaluación clínica ni constituye un diagnóstico por sí misma. 

___________________________________________________________________________________
## 3.	Descripción general del funcionamiento 

El sistema de estimación de riesgo funciona mediante el ingreso de variables relevantes del paciente, tales como edad, género, antecedentes familiares, exposición a radiación, deficiencia de yodo, tabaquismo, obesidad, diabetes, niveles de TSH, niveles de T3, niveles de T4 y tamaño del nódulo. A partir de esta información, el modelo calcula un puntaje o probabilidad que representa el riesgo estimado de cáncer de tiroides. 

El proceso se desarrolla en tres pasos principales: 

- Ingreso de los datos: El usuario introduce la información requerida sobre el paciente, siguiendo los criterios definidos para cada variable. 

- Procesamiento del modelo de riesgo: El sistema utiliza el modelo gradient boosting el cual calcula una estimación del riesgo de cáncer de tiroides. 

- Visualización e interpretación del resultado: Se presenta la probabilidad de que este nódulo sea maligno si esta es superior a 0.5 se indicara toma de biopsia de lo contrario se indicara que no requiere biopsia inmediata, pero debe continuar en controles y seguimiento.  

____________________________________________________________________________________
## 4.	Requisitos para su uso:  

### 4.1 Requisitos técnicos del sistema: 
Para el funcionamiento del dashboard se requieren unos requisitos de software y hardware. 

#### Software: 
•	Navegador actualizado (Chrome, Edge, Firefox). 
•	Conexión estable a internet. 

#### Hardware: 
•	Computador con RAM de mínimo 4Gb. 

### 4.2 Requisitos de ingreso de datos:  
Para calcular el riesgo estimado de cáncer de tiroides, el usuario debe ingresar la información del paciente en los campos correspondientes. Cada variable debe ser registrada siguiendo los criterios clínicos establecidos para garantizar un resultado confiable. 

Los datos se dividen en cuatro grupos principales: Datos demográficos, Factores de riesgo, Paraclínicos, e Indicadores generados por el sistema. 

El usuario debe registrar la información básica del paciente:

#### Datos demográficos: 
•	Edad: ingresar la edad en años completos. 
•	Sexo: seleccionar masculino o femenino. 
#### Factores de riesgo personales: 
•	Antecedentes Familiares: antecedente familiar de cáncer tiroideo. 
•	Tabaquismo: consumo actual o previo de tabaco. 
•	Obesidad: clasificación del paciente como obeso según criterios clínicos. 
•	Diabetes: diagnóstico confirmado de diabetes mellitus. 
#### Paraclínicos: 
El usuario debe ingresar los resultados de los estudios endocrinos y ecográficos: 
•	TSH: nivel sérico de TSH o hormona estimulante de tiroides. 
•	T3: nivel sérico de T3 o triyodotironina. 
•	T4: nivel sérico de T4 o tiroxima. 
•	Tamaño del nodulo: tamaño del nódulo tiroideo, según el reporte ecográfico. 
____________________________________________________________________________________
## 5.	Descripción de las variables del modelo: 

### 5.1 Datos demográficos: 
- Edad: La edad del paciente en años. Es un factor relevante porque ciertos grupos etarios presentan mayor riesgo de malignidad tiroidea. 
- Sexo: Es el sexo biológico del paciente (masculino o femenino). El cáncer de tiroides es más frecuente en mujeres. 

### 5.2 Factores de riesgo: 
- Antecedentes Familiares:  Indica si existe historia de cáncer de tiroides en familiares de primer grado, ya que esto aumenta significativamente el riesgo. 
- Tabaquismo: Se coloca si el paciente fuma actualmente o fumó en el pasado. El tabaquismo aumenta el riesgo de cáncer tiroideo. 
- Obesidad: Se indica si el paciente tiene obesidad de acuerdo si su índice de masa corporal es superior a 30 kg/m². La obesidad se ha relacionado con aumento del riesgo de malignidad en varios tumores, incluido el tiroideo. 
- Diabetes: Se coloca si el paciente presenta diabetes mellitus de cualquier tipo. La diabetes puede aumentar el riesgo de cancer tiroideo. 

### 5.3 Paraclínicos: 
- Niveles Hormona estimulante de la tiroides de TSH (mIU/L): Se debe ingresa la concentración de hormona estimulante de la tiroides (TSH) en mili unidades internacionales por litro. Niveles anormales pueden indicar disfunción tiroidea y se relacionan con la presencia y comportamiento de nódulos. 
- Niveles de la hormona triyodotironina T3 (ng/mL): Se deben ingresar los niveles de triyodotironina en sangre en nanogramos por mililitro. Es importante para evaluar el estado funcional de la glándula tiroides. 
- Niveles de la hormona tiroxina T4 (µg/dL): Se deben ingresar los niveles de concentración de tiroxina en microgramos por decilitro. Es importante para el análisis funcional del eje tiroideo. 
- Tamaño del nódulo (cm): Medida del nódulo tiroideo en centímetros. Los nódulos más grandes pueden tener mayor probabilidad de malignidad. 

____________________________________________________________________________________
 
## 6.	Procedimiento paso a paso: 
 1.	Se debe instalar el tablero de acuerdo con las instrucciones del manual de instalación, lo que llevara a una url donde va a encontrar una interfaz con los espacios para llenar las variables, la imagen que se observa se encuentra en la figura 1.
    
![alt text](image.png)
 Figura1. Interfaz del tablero. 

Figura1. Interfaz del tablero. 

 2.	Estas se deben llenar con los valores de los pacientes. 
 3.	Se oprime el botón Analizar riesgo. 
 4.	Se obtienen los resultados y la probabilidad de malignidad de acuerdo con las características ingresadas 

## 7.	Interpretación de resultados: 

El resultado del modelo es la probabilidad de malignidad: 
•	Si la probabilidad de malignidad se encuentra superior al 50% el resultado será maligno y se sugiere valoración médica prioritaria, practicar estudios y controles, para descartar si el paciente tiene cáncer de tiroides. 
•	Si la probabilidad de malignidad es inferior al 50%, el resultado será benigno y no se sugiere biopsia inmediata, sin embargo, debe continuar con sus controles y seguimiento médico. 
 4.	Se obtienen los resultados y la probabilidad de malignidad de acuerdo con las características ingresadas 

