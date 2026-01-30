import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64

# ------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE RECURSOS
# ------------------------------------------------------

st.set_page_config(page_title="Creditum", layout="wide")

@st.cache_resource
def load_resources():
    # Cargar Modelo
    try:
        with open('model_final.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: No se encuentra el archivo 'model_final.pkl'.")
        model = None

    # Cargar Datos Internos (Usamos el ID como índice para búsquedas rápidas)
    try:
        datos_internos = pd.read_csv('datos_internos.csv')
        if 'SK_ID_CURR' in datos_internos.columns:
            datos_internos = datos_internos.set_index('SK_ID_CURR')
    except FileNotFoundError:
        st.warning("Aviso: 'datos_internos.csv' no encontrado. Se usarán solo datos del formulario.")
        datos_internos = None
        
    return model, datos_internos

model_final, datos_internos_df = load_resources()

# ------------------------------------------------------
# 2. LÓGICA DE NEGOCIO Y PREDICCIÓN
# ------------------------------------------------------

def process_single_prediction(datos_solicitante, raw_input_data):
    """
    Combina datos, aplica el modelo y gestiona el guardado.
    """
    if model_final is None:
        return None, "Modelo no cargado."

    # Crear DataFrame del formulario
    df_form = pd.DataFrame([datos_solicitante]).set_index('SK_ID_CURR')

    # MERGE ROBUSTO: El formulario sobreescribe la base interna
    if datos_internos_df is not None and df_form.index[0] in datos_internos_df.index:
        # Recuperamos datos del bureau
        bureau_data = datos_internos_df.loc[[df_form.index[0]]]
        # combine_first: toma valores de df_form, y si son NaN, busca en bureau_data
        datos_completos = df_form.combine_first(bureau_data).reset_index()
    else:
        datos_completos = df_form.reset_index()

    # Preparar X para el modelo
    columnas_modelo = list(model_final.feature_names_in_)
    X = datos_completos.copy()

    # Asegurar que todas las columnas necesarias existan
    for col in columnas_modelo:
        if col not in X.columns:
            X[col] = 0
            
    X = X[columnas_modelo]

    # PREDICCIÓN CON UMBRAL (Para mayor sensibilidad al riesgo)
    # Si el modelo soporta predict_proba, podemos ajustar la rigurosidad
    try:
        prob_impago = model_final.predict_proba(X)[0][1]
        umbral = 0.4  # AJUSTE AQUÍ: Menor valor = Más difícil de aprobar
        prediction = 1 if prob_impago > umbral else 0
    except:
        prediction = model_final.predict(X)[0]

    # Guardar en historial
    save_to_csv(raw_input_data, prediction)

    return prediction, datos_completos

def save_to_csv(raw_data, prediction):
    filename = 'historial_creditos.csv'
    current_id = int(raw_data['SK_ID_CURR'])
    
    # Evitar duplicados
    if os.path.exists(filename):
        df_h = pd.read_csv(filename)
        if current_id in df_h['SK_ID_CURR'].values:
            return False

    raw_data['TARGET'] = int(prediction)
    raw_data['FECHA'] = pd.Timestamp.now()
    df_new = pd.DataFrame([raw_data])
    
    header = not os.path.exists(filename)
    df_new.to_csv(filename, index=False, mode='a', header=header)
    return True

def get_mappings():
    return {
        'children': {'0':0, '1':1, '2':2, '3':3, '4 o más':4},
        'education': {
            'Lower secondary':0, 'Secondary / secondary special':1, 
            'Incomplete higher':2, 'Higher education':3, 'Academic degree':4
        }
    }

# ------------------------------------------------------
# 3. VISTAS Y NAVEGACIÓN
# ------------------------------------------------------

def go_to_page(page_name):
    st.session_state.page = page_name

def page_home():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-attachment: fixed;
    }
    .header-box {
        background-color: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px); border-radius: 20px;
        padding: 40px; margin: 20px auto; max-width: 700px;
        text-align: center; border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .custom-title { color: #1e1e1e; font-weight: 800; font-size: 3rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header-box"><h1 class="custom-title">Análisis de Riesgo Crediticio</h1></div>', unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        with st.container(border=True):
            if st.button("💳 Nueva Evaluación", use_container_width=True, type="primary"):
                go_to_page("request")
                st.rerun()
        with st.container(border=True):
            if st.button("👥 Quiénes Somos", use_container_width=True):
                go_to_page("about")
                st.rerun()

def page_credit_request():
    st.button("⬅️ Volver", on_click=go_to_page, args=("home",))
    st.title("Solicitud de Crédito")
    
    tab1, tab2 = st.tabs(["👤 Individual", "👥 Carga Masiva"])
    maps = get_mappings()

    with tab1:
        with st.form("form_individual"):
            c1, c2 = st.columns(2)
            with c1:
                id_curr = st.number_input("ID Solicitante", min_value=1000, step=1)
                nombre = st.text_input("Nombre")
                edad = st.slider("Edad", 18, 90, 30)
                estudios = st.selectbox("Educación", list(maps['education'].keys()))
            with c2:
                ingresos = st.number_input("Ingresos Mensuales", min_value=0.0, value=20000.0)
                credito = st.number_input("Monto Crédito", min_value=0.0, value=50000.0)
                genero = st.selectbox("Género", ["Masculino", "Femenino"])
                vivienda = st.selectbox("Vivienda", ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment'])
            
            submit = st.form_submit_button("Evaluar Riesgo")

        if submit:
            # Construir datos para el modelo (One-Hot simplificado)
            datos_modelo = {
                'SK_ID_CURR': id_curr,
                'AMT_INCOME_TOTAL': ingresos,
                'AMT_CREDIT': credito,
                'AGE_BINS': 1 if edad < 35 else 2 if edad < 50 else 3,
                'GENDER_M': 1 if genero == "Masculino" else 0,
                'LEVEL_EDUCATION_TYPE': maps['education'][estudios],
                'HOUSING_TYPE_House_or_apartment': 1 if vivienda == 'House / apartment' else 0
            }
            # Raw data para guardar
            raw = {'SK_ID_CURR': id_curr, 'NAME': nombre, 'AMT_INCOME_TOTAL': ingresos, 'AMT_CREDIT': credito}
            
            res, details = process_single_prediction(datos_modelo, raw)
            if res == 0:
                st.success(f"✅ CRÉDITO APROBADO para {nombre}")
            else:
                st.error(f"❌ CRÉDITO DENEGADO para {nombre}")

    with tab2:
        st.info("Copia y pega desde Excel o llena la tabla.")
        df_template = pd.DataFrame(columns=["SK_ID_CURR", "NAME", "AMT_INCOME", "AMT_CREDIT", "AGE", "GENDER"])
        ed_df = st.data_editor(df_template, num_rows="dynamic", use_container_width=True)
        
        if st.button("Procesar Lista"):
            results = []
            for _, row in ed_df.iterrows():
                d_mod = {
                    'SK_ID_CURR': row['SK_ID_CURR'],
                    'AMT_INCOME_TOTAL': row['AMT_INCOME'],
                    'AMT_CREDIT': row['AMT_CREDIT'],
                    'GENDER_M': 1 if row['GENDER'] == "Masculino" else 0
                }
                raw_row = row.to_dict()
                pred, _ = process_single_prediction(d_mod, raw_row)
                status = "Aprobado" if pred == 0 else "Denegado"
                results.append({"ID": row['SK_ID_CURR'], "Resultado": status})
            st.table(results)

# ------------------------------------------------------
# 4. EJECUCIÓN
# ------------------------------------------------------

if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "about":
    st.write("Sección 'Sobre Nosotros' en construcción.")
    if st.button("Volver"): go_to_page("home"); st.rerun()
elif st.session_state.page == "request":
    page_credit_request()
