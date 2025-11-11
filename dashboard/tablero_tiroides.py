import streamlit as st
import joblib
import numpy as np
import random


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
    # model = joblib.load("models/modelo_tiroides.pkl")
    # X = np.array([[edad, 1 if sexo=="Femenino" else 0,
    #                1 if obesidad=="Si" else 0,
    #                1 if tabaquismo=="Si" else 0,
    #                1 if diabetes=="Si" else 0,
    #                t3, t4, tsh, tam_nodulo,
    #                1 if antecedentes=="Si" else 0]])
    # prob = model.predict_proba(X)[0][1]
    prob = random.uniform(0, 1)  #Comentar esta línea cuando ya tengamos el .pkl
    resultado = "Maligno" if prob > 0.5 else "Benigno"
    st.write(f"**Resultado:** {resultado}  -  **Probabilidad:** {prob:.2%}")

    if resultado == "Benigno":
        st.success("No se requiere biopsia inmediata")
    else:
        st.error("Requiere valoración médica prioritaria")