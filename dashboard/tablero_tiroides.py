import streamlit as st
import numpy as np
import requests

st.markdown(
    """
    <div style="
        background-color:#007BFF;
        padding:15px;
        border-radius:10px;
        text-align:center;
        color:white;
        font-size:30px;
        font-weight:bold;">
        PREDICCION TUMOR DE TIROIDES
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True) 

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
    tabaquismo = st.selectbox("Tabaquismo", ["Si", "No"])
    t4 = st.number_input("T4")
    antecedentes = st.selectbox("Antecedentes familiares", ["Si", "No"])

with col2:
    sexo = st.selectbox("Sexo", ["Masculino", "Femenino"])
    diabetes = st.selectbox("Diabetes", ["Si", "No"])
    tsh = st.number_input("TSH")

with col3:
    obesidad = st.selectbox("Obesidad", ["Si", "No"])
    t3 = st.number_input("T3")
    tam_nodulo = st.number_input("Tamaño del nódulo")

if st.button("Analizar riesgo"):

    # Se genera el vector de 15 características en el orden EXACTO que espera el modelo en la API.
    # Las variables no preguntadas (Riesgo_Cancer, País, Etnicidad, Yodo) se rellenan con cero.
    X = np.array([
        edad,                                  # 1. Edad
        tsh,                                   # 2. Nivel_TSH
        t3,                                    # 3. Nivel_T3
        t4,                                    # 4. Nivel_T4
        tam_nodulo,                            # 5. Tamanio_Nodo
        0,                                     # 6. Riesgo_Cancer_Tiroide (Asumido 0)
        1 if sexo=="Masculino" else 0,         # 7. Genero_Masculino
        0,                                     # 8. Pais_India (Asumido 0)
        0,                                     # 9. Etnicidad_Asiatico (Asumido 0)
        0,                                     # 10. Etnicidad_Caucasico (Asumido 0)
        1 if antecedentes=="Si" else 0,        # 11. Historial_Familiar_Si
        0,                                     # 12. Deficiencia_Vodo_Si (Asumido 0)
        1 if tabaquismo=="Si" else 0,          # 13. Tabaquismo_Si
        1 if obesidad=="Si" else 0,            # 14. Obesidad_Si
        1 if diabetes=="Si" else 0             # 15. Diabetes_Si
    ]).reshape(1,-1) # Se asegura que sea una matriz de 1x15 para el tolist()

    url_api = "http://98.94.83.51:8001/predict"

    payload = {
        "features": X.tolist()
    }
    
    # Intentar la conexión y capturar errores de red (e.g., IP incorrecta o caída)
    try:
        response = requests.post(url_api, json=X.tolist())
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión: No se pudo conectar a la API en {url_api}.")
        st.warning("Verifique que la IP y el puerto 8001 sean correctos para la tarea de la API.")
        st.stop()
    
    # Evaluar la respuesta del servidor (código 200 vs. errores 4xx/5xx)
    if response.status_code == 200:
        prob = response.json()["probabilidad"]
        resultado = "Maligno" if prob > 0.5 else "Benigno"

        st.write(f"**Resultado:** {resultado}  —  **Probabilidad:** {prob:.2%}")

        if resultado == "Benigno":
            st.success("No se requiere biopsia inmediata")
        else:
            st.error("Requiere valoración médica prioritaria")
    else:
        st.error(f"Error consultando la API: La API respondió con código {response.status_code}.")
        st.markdown(f"**Mensaje Técnico (JSON):** `{response.text}`")
        st.warning("Debe revisar los logs de la API en AWS CloudWatch para la causa raíz del error interno.")
