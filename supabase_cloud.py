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

    # Cargar Datos Internos
    try:
        datos_internos = pd.read_csv('datos_internos.csv')
    except FileNotFoundError:
        st.error("Error: No se encuentra el archivo 'datos_internos.csv'.")
        datos_internos = None
        
    return model, datos_internos

model_final, datos_internos_df = load_resources()

# ------------------------------------------------------
# 2. FUNCIONES DE UTILIDAD Y LÓGICA DE NEGOCIO
# ------------------------------------------------------

def process_single_prediction(datos_solicitante, raw_input_data):
    """
    Procesa solicitud, predice y gestiona el guardado.
    """
    df_datos = pd.DataFrame([datos_solicitante])
    
    # --- CORRECCIÓN CLAVE: Merge prioritario ---
    # Si el ID existe en la base interna, combinamos pero el FORMULARIO tiene la prioridad
    if datos_internos_df is not None and datos_solicitante['SK_ID_CURR'] in datos_internos_df['SK_ID_CURR'].values:
        bureau_row = datos_internos_df[datos_internos_df['SK_ID_CURR'] == datos_solicitante['SK_ID_CURR']]
        # Usamos combine_first para que los datos del formulario "pisen" a los del CSV
        datos_completos = df_datos.set_index('SK_ID_CURR').combine_first(bureau_row.set_index('SK_ID_CURR')).reset_index()
    else:
        datos_completos = df_datos
    
    if datos_completos.empty:
        return None, "ID no encontrado en base interna (Bureau)"
    
    # Columnas del modelo
    columnas_modelo = list(model_final.feature_names_in_)
    X = datos_completos.copy()

    # Limpieza de columnas no usadas por el modelo
    for col in ['NAME', 'FECHA_REGISTRO']: 
        if col in X.columns: X = X.drop(col, axis=1)

    # Asegurar compatibilidad (rellenar faltantes con 0 y ordenar)
    for col in columnas_modelo:
        if col not in X.columns: X[col] = 0
    X = X[columnas_modelo]

    # --- CORRECCIÓN DE SENSIBILIDAD: Uso de Probabilidades ---
    try:
        # Si el modelo permite probabilidades, bajamos el umbral para ser más estrictos
        prob = model_final.predict_proba(X)[0][1] 
        prediction = 1 if prob > 0.4 else 0 # Si la prob de impago es > 40%, denegar.
    except:
        prediction = model_final.predict(X)[0]

    # Guardado en CSV
    save_to_csv(raw_input_data, prediction)

    return prediction, datos_completos

def save_to_csv(raw_data, prediction):
    filename = 'historial_creditos.csv'
    current_id = int(raw_data['SK_ID_CURR'])

    if os.path.exists(filename):
        try:
            df_history = pd.read_csv(filename)
            if current_id in df_history['SK_ID_CURR'].values:
                return False, "Duplicado"
        except: pass

    # Preparar fila
    raw_data['TARGET'] = int(prediction)
    raw_data['FECHA_REGISTRO'] = pd.Timestamp.now()
    df_new_row = pd.DataFrame([raw_data])

    if not os.path.exists(filename):
        df_new_row.to_csv(filename, index=False, mode='w')
    else:
        df_new_row.to_csv(filename, index=False, mode='a', header=False)
    return True, "OK"

def get_mappings():
    return {
        'children': {'0':0, '1':1, '2':2, '3':3, '4 o más':4},
        'education': {
            'Lower secondary':0, 'Secondary / secondary special':1, 
            'Incomplete higher':2, 'Higher education':3, 'Academic degree':4
        }
    }

# ------------------------------------------------------
# 3. VISTAS (PÁGINAS)
# ------------------------------------------------------

def go_to_page(page_name):
    st.session_state.page = page_name

def page_home():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-position: center; background-attachment: fixed;
    }
    .header-box {
        background-color: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px);
        border-radius: 20px; padding: 40px 20px; margin: 20px auto;
        max-width: 650px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .custom-title { color: #000000 !important; font-family: 'Inter', sans-serif; font-weight: 800; font-size: 3.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="header-box"><h2 class="custom-title">Análisis inteligente del riesgo crediticio</h2></div>', unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; font-size: 2.2rem;'>🚀 Evaluación</h2>", unsafe_allow_html=True)
            if st.button("💳 Solicitar Crédito", use_container_width=True, type="primary"):
                go_to_page("request")
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; font-size: 2.2rem;'>🏢 Sobre nosotros</h2>", unsafe_allow_html=True)
            if st.button("👥 Quiénes Somos", use_container_width=True):
                go_to_page("about")
                st.rerun()

def page_about():
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    st.markdown("<h1 style='font-size: 3rem;'>Sobre Nosotros</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 22px; line-height: 1.6; text-align: justify;">
    Somos una plataforma especializada en <b>analítica avanzada y evaluación de riesgo crediticio</b>.
    <br><br>
    Nuestra solución analiza de forma integral variables financieras, laborales y demográficas...
    </div>
    """, unsafe_allow_html=True)

def page_credit_request():
    st.markdown("""<style>button[data-baseweb="tab"] div { font-size: 20px !important; } label p { font-size: 1.2rem !important; font-weight: bold !important; } .stButton button { font-size: 1.3rem !important; height: 3em !important; }</style>""", unsafe_allow_html=True)
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    st.markdown("<h1 style='font-size: 3rem;'>Solicitud de Crédito</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👤 Individual", "👥 Múltiples Solicitantes"])
    maps = get_mappings()

    # --- CASO 1: INDIVIDUAL ---
    with tab1:
        st.markdown("<h2 style='font-size: 1.8rem;'>Formulario Individual</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            SK_ID_CURR = st.text_input('ID del solicitante', key="single_id")
            NAME = st.text_input('Nombre del solicitante', key="single_name")
            AGES = st.slider('Edad:', 18, 100, 30, key="single_age")
            GENDER = st.selectbox('Género:', ['Masculino', 'Femenino'], key="single_gender")
            CNT_CHILDREN_TXT = st.selectbox('Número de hijos:', ['0','1', '2', '3', '4 o más'], key="single_kids")
            NAME_EDUCATION_TYPE = st.selectbox('Nivel de estudios:', ['Lower secondary','Secondary / secondary special', 'Incomplete higher','Higher education','Academic degree'], key="single_edu")
            FAMILY_STATUS = st.selectbox('Situación familiar:', ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'], key="single_fam")
        with col2:
            HOUSING_TYPE = st.selectbox('Tipo de vivienda:', ['With parents', 'Rented apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment'], key="single_house")
            AMT_INCOME_TOTAL = st.number_input('Ingresos:', min_value=0.0, step=100.0, key="single_inc")
            INCOME_TYPE = st.selectbox('Fuente de ingresos:', ['Working', 'State servant', 'Commercial associate', 'Businessman', 'Maternity leave', 'Student', 'Unemployed', 'Pensioner'], key="single_inctype")
            YEARS_ACTUAL_WORK_TXT = st.text_input('Años trabajo actual (Vacío = 0):', key="single_work")
            AMT_CREDIT = st.number_input('Crédito solicitado:', min_value=0.0, step=100.0, key="single_cred")
        
        st.markdown("#### Documentación y Bienes")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            FLAG_OWN_REALTY = st.checkbox('Casa Propia', key="s_house")
            FLAG_OWN_CAR = st.checkbox('Coche Propio', key="s_car")
            FLAG_PHONE = st.checkbox('Teléfono', key="s_ph")
        with d_col2:
            FLAG_DNI = st.checkbox('DNI Entregado', key="s_dni")
            FLAG_PASAPORTE = st.checkbox('Pasaporte', key="s_pass")
            FLAG_CERTIFICADO_LABORAL = st.checkbox('Cert. Laboral', key="s_lab")
        with d_col3:
            FLAG_COMPROBANTE_DOM_FISCAL = st.checkbox('Comp. Domicilio', key="s_dom")
            FLAG_ESTADO_CUENTA_BANC = st.checkbox('Estado Cuenta', key="s_acc")
            FLAG_TARJETA_ID_FISCAL = st.checkbox('ID Fiscal', key="s_fis")

        if st.button('Evaluar Solicitud Individual'):
            if not SK_ID_CURR.isdigit(): st.error("ID inválido"); return
            
            # Binning de edad
            bins = [18, 34, 43, 54, 100]; labels = [1, 2, 3, 4]
            AGE_BINS = pd.cut([AGES], bins=bins, labels=labels, right=True, include_lowest=True).to_list()[0]

            datos_solicitante = {
                'SK_ID_CURR': int(SK_ID_CURR),
                'AMT_INCOME_TOTAL': float(AMT_INCOME_TOTAL),
                'AMT_CREDIT': float(AMT_CREDIT),
                'AGE_BINS': int(AGE_BINS),
                'GENDER_M': 1 if GENDER == 'Masculino' else 0,
                'GENDER_F': 1 if GENDER == 'Femenino' else 0,
                'CNT_CHILDREN': maps['children'][CNT_CHILDREN_TXT],
                'LEVEL_EDUCATION_TYPE': maps['education'][NAME_EDUCATION_TYPE],
                'YEARS_ACTUAL_WORK': float(YEARS_ACTUAL_WORK_TXT) if YEARS_ACTUAL_WORK_TXT.replace('.','',1).isdigit() else 0.0,
                'FLAG_OWN_CAR': int(FLAG_OWN_CAR), 'FLAG_OWN_REALTY': int(FLAG_OWN_REALTY),
                'FLAG_PHONE': int(FLAG_PHONE), 'FLAG_DNI': int(FLAG_DNI),
                'FLAG_PASAPORTE': int(FLAG_PASAPORTE), 'FLAG_COMPROBANTE_DOM_FISCAL': int(FLAG_COMPROBANTE_DOM_FISCAL),
                'FLAG_ESTADO_CUENTA_BANC': int(FLAG_ESTADO_CUENTA_BANC), 'FLAG_TARJETA_ID_FISCAL': int(FLAG_TARJETA_ID_FISCAL),
                'FLAG_CERTIFICADO_LABORAL': int(FLAG_CERTIFICADO_LABORAL)
            }
            # OHE manual para que sea sensible
            datos_solicitante['FAMILY_STATUS_Married'] = 1 if FAMILY_STATUS == 'Married' else 0
            datos_solicitante['HOUSING_TYPE_House_or_apartment'] = 1 if HOUSING_TYPE == 'House / apartment' else 0
            datos_solicitante['INCOME_TYPE_Alta_Estabilidad'] = 1 if INCOME_TYPE in ['Working', 'State servant'] else 0

            raw_input = {'SK_ID_CURR': SK_ID_CURR, 'NAME': NAME, 'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL, 'AMT_CREDIT': AMT_CREDIT}
            
            pred, details = process_single_prediction(datos_solicitante, raw_input)
            if pred is not None:
                if pred == 0: st.success("✅ **CRÉDITO APROBADO**")
                else: st.error("❌ **CRÉDITO DENEGADO**")
                with st.expander("Detalles"): st.dataframe(details)

    # --- CASO 2: MASIVO ---
    with tab2:
        st.markdown("<h2 style='font-size: 1.8rem;'>Carga Masiva</h2>", unsafe_allow_html=True)
        column_config = {
            "SK_ID_CURR": st.column_config.NumberColumn("ID", format="%d"),
            "GENDER": st.column_config.SelectboxColumn("Género", options=['Masculino', 'Femenino']),
            "CNT_CHILDREN": st.column_config.SelectboxColumn("Hijos", options=['0','1', '2', '3', '4 o más']),
            "EDUCATION": st.column_config.SelectboxColumn("Educación", options=list(maps['education'].keys())),
            "AMT_INCOME": st.column_config.NumberColumn("Ingresos"),
            "AMT_CREDIT": st.column_config.NumberColumn("Crédito")
        }
        df_template = pd.DataFrame(columns=["SK_ID_CURR", "NAME", "AGE", "GENDER", "CNT_CHILDREN", "EDUCATION", "AMT_INCOME", "AMT_CREDIT"])
        edited_df = st.data_editor(df_template, num_rows="dynamic", column_config=column_config, use_container_width=True)

        if st.button("Procesar Lista Completa"):
            results_log = []
            for index, row in edited_df.iterrows():
                try:
                    # Corrección: Ahora sí definimos los datos para cada fila
                    d_row = {
                        'SK_ID_CURR': int(row['SK_ID_CURR']),
                        'AMT_INCOME_TOTAL': float(row['AMT_INCOME']),
                        'AMT_CREDIT': float(row['AMT_CREDIT']),
                        'GENDER_M': 1 if row['GENDER'] == 'Masculino' else 0,
                        'LEVEL_EDUCATION_TYPE': maps['education'][row['EDUCATION']]
                    }
                    raw_row = {'SK_ID_CURR': row['SK_ID_CURR'], 'NAME': row['NAME'], 'AMT_INCOME_TOTAL': row['AMT_INCOME']}
                    pred, _ = process_single_prediction(d_row, raw_row)
                    results_log.append({"ID": row['SK_ID_CURR'], "Nombre": row['NAME'], "Resultado": "Aprobado" if pred == 0 else "Denegado"})
                except Exception as e:
                    results_log.append({"ID": row.get('SK_ID_CURR', 'N/A'), "Resultado": f"Error: {str(e)}"})
            st.table(pd.DataFrame(results_log))

# ------------------------------------------------------
# 4. ENRUTAMIENTO
# ------------------------------------------------------

if 'page' not in st.session_state: st.session_state.page = "home"

if st.session_state.page == "home": page_home()
elif st.session_state.page == "about": page_about()
elif st.session_state.page == "request": page_credit_request()
