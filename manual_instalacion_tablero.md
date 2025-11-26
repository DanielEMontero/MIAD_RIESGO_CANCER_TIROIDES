# MANUAL DE DESPLIEGUE E INSTALACIÓN

**Sistema de Predicción de Cáncer de Tiroides**

Este documento describe el procedimiento técnico para desplegar la solución de predicción de cáncer de tiroides.

La solución plantea una arquitectura de microservicios contenerizados desplegados en AWS ECS, compuesta por dos servicios independientes:

- **API (Backend):** Expone el modelo de Machine Learning vía FastAPI.
- **Tablero (Frontend):** Interfaz de usuario en Streamlit que consume la API.

---

## 1. Estructura del Repositorio

El repositorio se ha actualizado para soportar la contenerización. La estructura relevante es:

```
README.md
requirements.txt
prueba_commit.txt
.dvcignore
.gitignore

dashboard/
├── app.py                  # Código fuente de la API (FastAPI)
├── tablero_tiroides.py     # Código fuente del Tablero (Streamlit)
├── dockerfile_api          # Definición de imagen Docker para la API
├── dockerfile_tablero      # Definición de imagen Docker para el Tablero
└── modelo_tiroides.pkl     # Modelo serializado (usado solo por la API)

data/
└── thyroid_cancer_risk_data.csv.dvc

scripts/
├── exploracion_data_tiroides.py
├── iterar_modelo.py
├── modelo_gradient_boosting.py
├── modelo_lightgbm.py
├── modelo_prediccion_cancer_tiroides.py
├── modelo_random_forest.py
├── modelo_xgboost.py
└── preparar_datos_utils.py

docs/
├── manual_usuario.md
└── manual_instalacion.md
```

---

## 2. Requisitos Previos

### Software Local

- Docker Desktop o Docker Engine (en ejecución).
- AWS CLI instalado y configurado con credenciales válidas (`aws configure`).
- Python 3.10+ (para pruebas locales opcionales).
- Git.

### Infraestructura Cloud (AWS)

- **Amazon ECR:** Dos repositorios creados (uno para la API, otro para el Tablero).
- **Amazon ECS:** Un clúster activo.
- **Task Definitions:** Definiciones de tarea configuradas para API y Tablero.

---

## 3. Configuración Inicial

Antes de realizar cualquier despliegue, asegúrese de tener las últimas versiones del código y los datos.

```bash
git clone [[URL_DEL_REPOSITORIO]](https://github.com/DanielEMontero/MIAD_RIESGO_CANCER_TIROIDES)
cd MIAD_RIESGO_CANCER_TIROIDES
```

---

## 4. Flujo de Despliegue de la API (Backend)

**Nota: Como se ha mostrado en el documento final, tanto el api como el tablero ya se encuentran desplegados en ECS por lo que se deberían usar las IP ya definidas allí ya que los servicios no se apagarán. En caso de requerir un nuevo despliegue, sigue estas instrucciones**

La API debe desplegarse primero, ya que el Tablero depende de su dirección IP para funcionar.

### 4.1. Construir la imagen

Desde la raíz del repositorio:

```bash
docker build -t api-tiroides -f dashboard/dockerfile_api .
```

### 4.2. Autenticación en ECR

Refresque el token de autenticación de Docker con AWS:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 275856882302.dkr.ecr.us-east-1.amazonaws.com
```

### 4.3. Etiquetar y Subir (Push)

```bash
docker tag api-tiroides:latest 275856882302.dkr.ecr.us-east-1.amazonaws.com/api-tiroides:latest
docker push 275856882302.dkr.ecr.us-east-1.amazonaws.com/api-tiroides:latest
```

### 4.4. Actualizar Servicio en ECS

1. Vaya a la consola de AWS ECS → Cluster → Servicio de la API.
2. Seleccione **Update** → Marque **Force new deployment**.
3. Espere a que la tarea esté en estado **RUNNING**.

**IMPORTANTE:** Copie la IP Pública nueva de la tarea de la API.

---

## 5. Flujo de Despliegue del Tablero (Frontend)

El tablero requiere conocer la ubicación de la API para enviar las predicciones.

### 5.1. Actualizar conexión a la API

Edite el archivo `dashboard/tablero_tiroides.py`. Busque la variable `url_api` y actualícela con la IP obtenida en el paso 4.4:

```python
# Ejemplo
url_api = "http://[NUEVA_IP_API]:8001/predict"
```

### 5.2. Construir la imagen

Desde la raíz del repositorio:

```bash
docker build -t tablero-tiroides -f dashboard/dockerfile_tablero .
```

### 5.3. Etiquetar y Subir (Push)

```bash
docker tag tablero-tiroides:latest  275856882302.dkr.ecr.us-east-1.amazonaws.com/tablero-tiroides:latest
docker push  275856882302.dkr.ecr.us-east-1.amazonaws.com/tablero-tiroides:latest
```

### 5.4. Actualizar Servicio en ECS

Vaya a la consola de AWS ECS → Cluster → Servicio del Tablero.  
Seleccione **Update** → Marque **Force new deployment**.

---

## 6. Acceso a la Aplicación

Una vez finalizado el despliegue del tablero:

1. Ubique la IP Pública de la tarea del servicio del Tablero en ECS.  
2. Abra su navegador y navegue a:

```
http://[IP_PUBLICA_TABLERO]:8501
```

---

## 7. Reentrenamiento del Modelo (MLOps)

Si desea actualizar el modelo predictivo (`modelo_tiroides.pkl`):

```bash
python scripts/modelo_prediccion_cancer_tiroides.py
```

Esto generará un nuevo archivo `.pkl` en la carpeta `dashboard/`.  

Para que los cambios surtan efecto, debe repetir obligatoriamente el **Paso 4 (Flujo de Despliegue de la API)**, ya que el modelo vive dentro del contenedor de la API.

---

## 8. Solución de Problemas Comunes

- **Error "Circuit Breaker" en ECS:** Generalmente indica que el contenedor falla al iniciar. Revise los logs en CloudWatch. Causas comunes: rutas de archivos incorrectas o falta de dependencias (ej. `python-multipart`).
- **Error de Conexión en el Tablero:** Verifique que la IP configurada en `tablero_tiroides.py` coincida con la IP pública actual de la tarea de la API en ejecución.
- **Permiso denegado en `docker push`:** Su token de sesión ha expirado. Repita el comando de autenticación `aws ecr get-login-password ...`.

