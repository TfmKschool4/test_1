import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from supabase import create_client

# ------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE RECURSOS
# ------------------------------------------------------

st.set_page_config(page_title="Creditum", layout="wide", page_icon="🛡️")

# Función para cargar recursos (con caché para no recargar en cada interacción)
@st.cache_resource
def load_resources():
    # Cargar Modelo
    try:
        with open('model_final.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        # Creamos un modelo dummy para que la app no falle si no tienes el pkl a mano
        st.warning("Aviso: 'model_final.pkl' no encontrado. Usando modo demostración.")
        model = "DUMMY_MODEL" 

    # Cargar Datos Internos
    try:
        datos_internos = pd.read_csv('datos_internos.csv', index_col=0)
    except FileNotFoundError:
        st.warning("Aviso: 'datos_internos.csv' no encontrado. Usando datos vacíos.")
        datos_internos = pd.DataFrame()
        
    return model, datos_internos

# Función auxiliar para convertir imagen a base64 (para insertarla en HTML/CSS)
def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

# Conexión Supabase
try:
    url = os.environ.get('SUPABASE_URL', '')
    key = os.environ.get('SUPABASE_KEY', '')
    if url and key:
        supabase = create_client(url, key)
    else:
        supabase = None
except Exception:
    supabase = None

model_final, datos_internos_df = load_resources()

# ------------------------------------------------------
# 2. FUNCIONES DE UTILIDAD Y LÓGICA DE NEGOCIO
# ------------------------------------------------------

def process_single_prediction(datos_solicitante, raw_input_data):
    """
    Procesa una única solicitud (diccionario pre-procesado), hace el merge, predice y guarda.
    """
    if model_final == "DUMMY_MODEL":
        # Simulación para demo si falta el archivo
        return np.random.choice([0, 1]), pd.DataFrame([datos_solicitante])

    df_datos = pd.DataFrame([datos_solicitante])
    
    # Merge con datos internos
    if not datos_internos_df.empty:
        datos_completos = df_datos.merge(datos_internos_df, on='SK_ID_CURR', how='left')
        # Rellenar nulos si el ID no cruza, para no romper el modelo
        datos_completos.fillna(0, inplace=True)
    else:
        datos_completos = df_datos
        # Añadir columna faltante dummy si no hay csv interno
        datos_completos['DEF_30_CNT_SOCIAL_CIRCLE'] = 0

    # Columnas requeridas por el modelo (asegurar orden)
    columnas_modelo = [
        'SK_ID_CURR', 'NAME', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'LEVEL_EDUCATION_TYPE', 'AGE_BINS',
        'YEARS_ACTUAL_WORK', 'FLAG_PHONE', 'DEF_30_CNT_SOCIAL_CIRCLE',
        'FLAG_COMPROBANTE_DOM_FISCAL', 'FLAG_ESTADO_CUENTA_BANC', 'FLAG_PASAPORTE',
        'FLAG_TARJETA_ID_FISCAL', 'FLAG_DNI', 'FLAG_CERTIFICADO_LABORAL',
        'GENDER_F', 'GENDER_M', 'INCOME_TYPE_Alta_Estabilidad',
        'INCOME_TYPE_Baja_Estabilidad', 'INCOME_TYPE_Media_Estabilidad',
        'INCOME_TYPE_Pensionista', 'FAMILY_STATUS_Civil_marriage',
        'FAMILY_STATUS_Married', 'FAMILY_STATUS_Separated',
        'FAMILY_STATUS_Single_or_not_married', 'FAMILY_STATUS_Widow',
        'HOUSING_TYPE_Co_op_apartment', 'HOUSING_TYPE_House_or_apartment',
        'HOUSING_TYPE_Municipal_apartment', 'HOUSING_TYPE_Office_apartment',
        'HOUSING_TYPE_Rented_apartment', 'HOUSING_TYPE_With_parents'
    ]
    
    # Asegurar que todas las columnas existan, si falta alguna poner 0
    for col in columnas_modelo:
        if col not in datos_completos.columns:
            datos_completos[col] = 0
            
    datos_completos = datos_completos[columnas_modelo]
    
    # Predicción
    X = datos_completos.drop(['SK_ID_CURR', 'NAME'], axis=1)
    
    try:
        prediction = model_final.predict(X)[0] # Tomamos el valor escalar
    except Exception as e:
        return None, f"Error en predicción: {str(e)}"
    
    # Guardar en Supabase
    save_to_supabase(raw_input_data, datos_completos, prediction)
    
    return prediction, datos_completos

def save_to_supabase(raw_data, datos_completos, prediction):
    if not supabase: return

    try:
        new_loan_variables = {
            'SK_ID_CURR': int(raw_data['SK_ID_CURR']),
            'NAME': str(raw_data['NAME']),
            'CODE_GENDER': 'M' if raw_data['GENDER'] == 'Masculino' else 'F',
            'FLAG_OWN_REALTY': int(raw_data['FLAG_OWN_REALTY']),
            'CNT_CHILDREN': int(raw_data['CNT_CHILDREN_MAPPED']),
            'AMT_INCOME_TOTAL': float(raw_data['AMT_INCOME_TOTAL']),
            'AMT_CREDIT': float(raw_data['AMT_CREDIT']),
            'NAME_INCOME_TYPE': str(raw_data['INCOME_TYPE']),
            'NAME_EDUCATION_TYPE': str(raw_data['NAME_EDUCATION_TYPE']),
            'NAME_FAMILY_STATUS': str(raw_data['FAMILY_STATUS']),
            'NAME_HOUSING_TYPE': str(raw_data['HOUSING_TYPE']),
            'AGE': int(raw_data['AGE']),
            'YEARS_ACTUAL_WORK': float(raw_data['YEARS_ACTUAL_WORK']) if raw_data['YEARS_ACTUAL_WORK'] else None,
            'FLAG_PHONE': int(raw_data['FLAG_PHONE']),
            'DEF_30_CNT_SOCIAL_CIRCLE': int(datos_completos['DEF_30_CNT_SOCIAL_CIRCLE'].iloc[0]),
            'FLAG_COMPROBANTE_DOM_FISCAL': int(raw_data['FLAG_COMPROBANTE_DOM_FISCAL']),
            'FLAG_ESTADO_CUENTA_BANC': int(raw_data['FLAG_ESTADO_CUENTA_BANC']),
            'FLAG_PASAPORTE': int(raw_data['FLAG_PASAPORTE']),
            'FLAG_TARJETA_ID_FISCAL': int(raw_data['FLAG_TARJETA_ID_FISCAL']),
            'FLAG_DNI': int(raw_data['FLAG_DNI']),
            'FLAG_CERTIFICADO_LABORAL': int(raw_data['FLAG_CERTIFICADO_LABORAL']),
            'TARGET': int(prediction)
        }
        supabase.table('historical_loans').insert(new_loan_variables).execute()
    except Exception as e:
        print(f"Error Supabase: {e}")

# Mapeos auxiliares para transformar texto a números/dummies
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
    # Convertir logo a base64 para insertarlo en HTML
    logo_b64 = get_img_as_base64("logo.jpg")
    if not logo_b64:
        # Fallback si no encuentra la imagen
        logo_html = "<h1 class='custom-title'>Creditum</h1>"
    else:
        logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" class="logo-img" alt="Creditum Logo">'

    # --- 1. CSS ESTILO "GLASSMORPHISM" MEJORADO ---
    st.markdown("""
    <style>
    /* FONDO DE PANTALLA */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* CAPA OSCURA SUAVE */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.2); 
        z-index: -1;
    }

    /* CONTENEDOR CENTRAL (Glass Card) */
    .header-box {
        background: rgba(255, 255, 255, 0.85); /* Blanco translúcido */
        backdrop-filter: blur(12px);            /* Efecto desenfoque */
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.6); /* Borde sutil */
        border-radius: 24px;
        padding: 50px 30px;
        margin-bottom: 40px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15); /* Sombra suave */
        text-align: center;
        display: flex;
        flex-direction: column;
        align_items: center;
        justify_content: center;
    }

    /* ESTILO DE IMAGEN DEL LOGO */
    .logo-img {
        max-width: 400px;
        width: 100%;
        height: auto;
        margin-bottom: 15px;
        filter: drop-shadow(0px 4px 4px rgba(0,0,0,0.1));
    }

    /* SUBTÍTULO CON ALTO CONTRASTE */
    .custom-subtitle {
        color: #1a1a1a !important; /* Gris muy oscuro casi negro */
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.5rem;
        font-weight: 600; /* Letra más gruesa para leerse mejor */
        margin-top: 10px;
        text-shadow: 0px 0px 20px rgba(255,255,255, 0.8); /* Halo blanco para separar del fondo si es necesario */
    }
    
    /* MODIFICAR LAS TARJETAS DE ABAJO PARA QUE SEAN COHERENTES */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* BOTONES */
    .stButton > button {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- 2. CONTENIDO PRINCIPAL ---
    
    st.markdown(f"""
    <div class="header-box">
        {logo_html}
        <p class="custom-subtitle">Análisis inteligente del riesgo crediticio.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 3. BOTONES DE ACCIÓN ---
    col_spacer_left, col_action1, col_action2, col_spacer_right = st.columns([0.5, 2, 2, 0.5])

    with col_action1:
        with st.container(border=True):
            st.markdown("### 🏢 Sobre nosotros")
            if st.button("👥 Quiénes Somos", use_container_width=True):
                go_to_page("about")
                st.rerun()

    with col_action2:
        with st.container(border=True):
            st.markdown("### 🚀 Evaluación")
            st.success("Scoring individual o masivo en tiempo real.")
            if st.button("💳 Solicitar Crédito", use_container_width=True, type="primary"):
                go_to_page("request")
                st.rerun()
                
def page_about():
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    st.title("Sobre Nosotros")

    # 🔽 MODIFICADO: TEXTO CORPORATIVO
    st.markdown("""
    Somos una plataforma especializada en **analítica avanzada y evaluación de riesgo crediticio**, diseñada para apoyar la toma de decisiones financieras mediante el uso de **modelos predictivos**.

    Nuestra solución analiza de forma integral variables financieras, laborales y demográficas con el objetivo de **estimar la probabilidad de impago** de un solicitante y proporcionar recomendaciones objetivas, consistentes y escalables para la concesión de crédito.

    El sistema está pensado para integrarse en procesos reales de evaluación crediticia, permitiendo tanto el análisis **individual** como el **procesamiento masivo de solicitudes**, con trazabilidad de resultados y almacenamiento histórico de decisiones.

    Creemos en el uso responsable de la tecnología para impulsar **decisiones financieras más inteligentes, eficientes y basadas en datos**, reduciendo la incertidumbre y mejorando la gestión del riesgo.

    > *La tecnología al servicio de decisiones financieras más seguras y eficientes.*
    """)

def page_credit_request():
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    st.title("Solicitud de Crédito")

    tab1, tab2 = st.tabs(["👤 Individual", "👥 Múltiples Solicitantes"])

    # -------------------------
    # CASO 1: INDIVIDUAL
    # -------------------------
    with tab1:
        st.subheader("Formulario Individual")
        
        # --- INPUTS (Código original adaptado) ---
        col1, col2 = st.columns(2)
        
        with col1:
            SK_ID_CURR = st.text_input('ID del solicitante', key="single_id")
            NAME = st.text_input('Nombre del solicitante', key="single_name")
            AGES = st.slider('Edad:', 18, 100, 30, key="single_age")
            GENDER = st.selectbox('Género:', ['Masculino', 'Femenino'], key="single_gender")
            CNT_CHILDREN_TXT = st.selectbox('Número de hijos:', ['0','1', '2', '3', '4 o más'], key="single_kids")
            NAME_EDUCATION_TYPE = st.selectbox('Nivel de estudios:', 
                ['Lower secondary','Secondary / secondary special', 'Incomplete higher','Higher education','Academic degree'], key="single_edu")
            FAMILY_STATUS = st.selectbox('Situación familiar:',
                ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'], key="single_fam")
            
        with col2:
            HOUSING_TYPE = st.selectbox('Tipo de vivienda:',
                ['With parents', 'Rented apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment'], key="single_house")
            AMT_INCOME_TOTAL = st.number_input('Ingresos:', min_value=0.0, step=100.0, key="single_inc")
            INCOME_TYPE = st.selectbox('Fuente de ingresos:',
                ['Working', 'State servant', 'Commercial associate', 'Businessman', 'Maternity leave', 'Student', 'Unemployed', 'Pensioner'], key="single_inctype")
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
            # Validación simple
            errors = []
            if not SK_ID_CURR.isdigit(): errors.append("ID inválido")
            if not NAME or not NAME.replace(" ","").isalpha(): errors.append("Nombre inválido")
            
            if errors:
                for e in errors: st.error(e)
            else:
                # Procesamiento de variables
                maps = get_mappings()
                
                # Binning de edad
                labels = [1, 2, 3, 4]
                bins = [18, 34, 43, 54, 100]
                AGE_BINS = pd.cut([AGES], bins=bins, labels=labels, right=True, include_lowest=True).to_list()[0]

                # Construcción del diccionario de datos (One Hot Encoding Manual)
                datos_solicitante = {
                    'SK_ID_CURR': int(SK_ID_CURR),
                    'NAME': NAME,
                    'AGE_BINS': int(AGE_BINS),
                    'GENDER_M': 1 if GENDER == 'Masculino' else 0,
                    'GENDER_F': 1 if GENDER == 'Femenino' else 0,
                    'CNT_CHILDREN': maps['children'][CNT_CHILDREN_TXT],
                    'LEVEL_EDUCATION_TYPE': maps['education'][NAME_EDUCATION_TYPE],
                    'AMT_INCOME_TOTAL': float(AMT_INCOME_TOTAL),
                    'AMT_CREDIT': float(AMT_CREDIT),
                    'YEARS_ACTUAL_WORK': float(YEARS_ACTUAL_WORK_TXT) if YEARS_ACTUAL_WORK_TXT and YEARS_ACTUAL_WORK_TXT.replace('.','',1).isdigit() else np.nan,
                    'FLAG_OWN_CAR': int(FLAG_OWN_CAR), 'FLAG_OWN_REALTY': int(FLAG_OWN_REALTY),
                    'FLAG_PHONE': int(FLAG_PHONE), 'FLAG_DNI': int(FLAG_DNI),
                    'FLAG_PASAPORTE': int(FLAG_PASAPORTE), 'FLAG_COMPROBANTE_DOM_FISCAL': int(FLAG_COMPROBANTE_DOM_FISCAL),
                    'FLAG_ESTADO_CUENTA_BANC': int(FLAG_ESTADO_CUENTA_BANC), 'FLAG_TARJETA_ID_FISCAL': int(FLAG_TARJETA_ID_FISCAL),
                    'FLAG_CERTIFICADO_LABORAL': int(FLAG_CERTIFICADO_LABORAL)
                }

                # OHE Family Status
                fam_opts = ['Single / not married', 'Married', 'Civil marriage', 'Separated', 'Widow']
                for f in fam_opts:
                    col_name = f"FAMILY_STATUS_{f.replace(' / ', '_or_').replace(' ', '_')}" 
                    if f == 'Single / not married': k = 'FAMILY_STATUS_Single_or_not_married'
                    elif f == 'Civil marriage': k = 'FAMILY_STATUS_Civil_marriage'
                    else: k = f"FAMILY_STATUS_{f}"
                    datos_solicitante[k] = 1 if FAMILY_STATUS == f else 0

                # OHE Housing
                hous_opts = ['With parents', 'Rented apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment']
                for h in hous_opts:
                    if h == 'House / apartment': k = 'HOUSING_TYPE_House_or_apartment'
                    elif h == 'Co-op apartment': k = 'HOUSING_TYPE_Co_op_apartment' 
                    else: k = f"HOUSING_TYPE_{h.replace(' ', '_')}"
                    datos_solicitante[k] = 1 if HOUSING_TYPE == h else 0

                # OHE Income
                datos_solicitante['INCOME_TYPE_Alta_Estabilidad'] = 1 if INCOME_TYPE in ['Working', 'State servant'] else 0
                datos_solicitante['INCOME_TYPE_Media_Estabilidad'] = 1 if INCOME_TYPE in ['Commercial associate', 'Businessman'] else 0
                datos_solicitante['INCOME_TYPE_Baja_Estabilidad'] = 1 if INCOME_TYPE in ['Maternity leave', 'Student', 'Unemployed'] else 0
                datos_solicitante['INCOME_TYPE_Pensionista'] = 1 if INCOME_TYPE == 'Pensioner' else 0

                # Datos "raw" para guardar en Supabase (texto original)
                raw_input_data = {
                    'SK_ID_CURR': SK_ID_CURR, 'NAME': NAME, 'GENDER': GENDER, 'AGE': AGES,
                    'CNT_CHILDREN_MAPPED': maps['children'][CNT_CHILDREN_TXT],
                    'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE, 'FAMILY_STATUS': FAMILY_STATUS,
                    'HOUSING_TYPE': HOUSING_TYPE, 'INCOME_TYPE': INCOME_TYPE,
                    'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL, 'AMT_CREDIT': AMT_CREDIT,
                    'YEARS_ACTUAL_WORK': YEARS_ACTUAL_WORK_TXT,
                    'FLAG_OWN_REALTY': FLAG_OWN_REALTY, 'FLAG_OWN_CAR': FLAG_OWN_CAR,
                    'FLAG_PHONE': FLAG_PHONE, 'FLAG_DNI': FLAG_DNI, 'FLAG_PASAPORTE': FLAG_PASAPORTE,
                    'FLAG_COMPROBANTE_DOM_FISCAL': FLAG_COMPROBANTE_DOM_FISCAL, 'FLAG_ESTADO_CUENTA_BANC': FLAG_ESTADO_CUENTA_BANC,
                    'FLAG_TARJETA_ID_FISCAL': FLAG_TARJETA_ID_FISCAL, 'FLAG_CERTIFICADO_LABORAL': FLAG_CERTIFICADO_LABORAL
                }

                # Ejecutar predicción
                pred, details = process_single_prediction(datos_solicitante, raw_input_data)
                
                if pred is not None:
                    st.success(f"Solicitud procesada con éxito.")
                    if pred == 0:
                        st.success("✅ **CRÉDITO APROBADO** (Riesgo Bajo - Predicción: 0)")
                    else:
                        st.error("❌ **CRÉDITO DENEGADO** (Riesgo Alto - Predicción: 1)")
                    
                    with st.expander("Ver detalles técnicos"):
                        st.dataframe(details)
                else:
                    st.error(details) # Mensaje de error (ID no encontrado)

    # -------------------------
    # CASO 2: MÚLTIPLE (TABLA)
    # -------------------------
    with tab2:
        st.subheader("Carga Masiva de Solicitudes")
        st.info("Añada filas a la tabla a continuación. Puede copiar y pegar desde Excel.")

        # Configuración de columnas para el editor
        column_config = {
            "SK_ID_CURR": st.column_config.NumberColumn("ID Solicitante", min_value=0, step=1, format="%d"),
            "NAME": st.column_config.TextColumn("Nombre"),
            "AGE": st.column_config.NumberColumn("Edad", min_value=18, max_value=100),
            "GENDER": st.column_config.SelectboxColumn("Género", options=['Masculino', 'Femenino'], required=True),
            "CNT_CHILDREN": st.column_config.SelectboxColumn("Hijos", options=['0','1', '2', '3', '4 o más'], required=True),
            "EDUCATION": st.column_config.SelectboxColumn("Educación", options=['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree'], required=True),
            "FAMILY_STATUS": st.column_config.SelectboxColumn("Estado Civil", options=['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'], required=True),
            "HOUSING": st.column_config.SelectboxColumn("Vivienda", options=['With parents', 'Rented apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment'], required=True),
            "INCOME_TYPE": st.column_config.SelectboxColumn("Fuente Ingresos", options=['Working', 'State servant', 'Commercial associate', 'Businessman', 'Maternity leave', 'Student', 'Unemployed', 'Pensioner'], required=True),
            "AMT_INCOME": st.column_config.NumberColumn("Ingresos", min_value=0),
            "AMT_CREDIT": st.column_config.NumberColumn("Crédito", min_value=0),
            "YEARS_WORKED": st.column_config.NumberColumn("Años Trabajados", min_value=0),
            "OWN_REALTY": st.column_config.CheckboxColumn("Casa Propia"),
            "OWN_CAR": st.column_config.CheckboxColumn("Coche Propio"),
            "FLAG_PHONE": st.column_config.CheckboxColumn("Teléfono"),
            "FLAG_DNI": st.column_config.CheckboxColumn("DNI"),
            "FLAG_PASAPORTE": st.column_config.CheckboxColumn("Pasaporte"),
            "FLAG_CERTIFICADO_LABORAL": st.column_config.CheckboxColumn("Cert. Laboral"),
            "FLAG_COMPROBANTE_DOM_FISCAL": st.column_config.CheckboxColumn("Comp. Domicilio"),
            "FLAG_ESTADO_CUENTA_BANC": st.column_config.CheckboxColumn("Estado Cuenta"),
            "FLAG_TARJETA_ID_FISCAL": st.column_config.CheckboxColumn("ID Fiscal")
        }

        # DataFrame plantilla
        df_template = pd.DataFrame(columns=[
            "SK_ID_CURR", "NAME", "AGE", "GENDER", "CNT_CHILDREN", "EDUCATION", 
            "FAMILY_STATUS", "HOUSING", "INCOME_TYPE", "AMT_INCOME", "AMT_CREDIT", 
            "YEARS_WORKED", "OWN_REALTY", "OWN_CAR",
            "FLAG_PHONE", "FLAG_DNI", "FLAG_PASAPORTE",
            "FLAG_CERTIFICADO_LABORAL", "FLAG_COMPROBANTE_DOM_FISCAL",
            "FLAG_ESTADO_CUENTA_BANC", "FLAG_TARJETA_ID_FISCAL"
        ])

        edited_df = st.data_editor(df_template, num_rows="dynamic", column_config=column_config, use_container_width=True)

        if st.button("Procesar Lista Completa"):
            if edited_df.empty:
                st.warning("La tabla está vacía.")
            else:
                progress_bar = st.progress(0)
                results_log = []
                
                maps = get_mappings()
                bins = [18, 34, 43, 54, 100]
                labels = [1, 2, 3, 4]

                for index, row in edited_df.iterrows():
                    # Preparar datos fila por fila
                    try:
                        # 1. Definir docs_ok basado en los flags de la fila
                        # Si quieres ser estricto: docs_ok = 1 si TIENE DNI y PASAPORTE, por ejemplo.
                        # Aquí asumiremos que cada flag individual cuenta, y docs_ok era una variable auxiliar.
                        # La usamos como 1 para cumplir con los requerimientos del diccionario si faltan datos específicos.
                        
                        flag_dni = 1 if row.get('FLAG_DNI') else 0
                        flag_pass = 1 if row.get('FLAG_PASAPORTE') else 0
                        
                        # Mapeos básicos
                        age_bin = pd.cut([row['AGE']], bins=bins, labels=labels, right=True, include_lowest=True).to_list()[0]
                        
                        # Construir diccionario solicitante
                        d = {
                            'SK_ID_CURR': int(row['SK_ID_CURR']),
                            'NAME': row['NAME'],
                            'AGE_BINS': int(age_bin),
                            'GENDER_M': 1 if row['GENDER'] == 'Masculino' else 0,
                            'GENDER_F': 1 if row['GENDER'] == 'Femenino' else 0,
                            'CNT_CHILDREN': maps['children'][str(row['CNT_CHILDREN'])],
                            'LEVEL_EDUCATION_TYPE': maps['education'][row['EDUCATION']],
                            'AMT_INCOME_TOTAL': float(row['AMT_INCOME']),
                            'AMT_CREDIT': float(row['AMT_CREDIT']),
                            'YEARS_ACTUAL_WORK': float(row['YEARS_WORKED']) if pd.notnull(row['YEARS_WORKED']) else np.nan,
                            'FLAG_OWN_CAR': int(row['OWN_CAR']), 
                            'FLAG_OWN_REALTY': int(row['OWN_REALTY']),
                            'FLAG_PHONE': int(row.get('FLAG_PHONE', 0)), 
                            'FLAG_DNI': flag_dni, 
                            'FLAG_PASAPORTE': flag_pass, 
                            'FLAG_COMPROBANTE_DOM_FISCAL': int(row.get('FLAG_COMPROBANTE_DOM_FISCAL', 0)), 
                            'FLAG_ESTADO_CUENTA_BANC': int(row.get('FLAG_ESTADO_CUENTA_BANC', 0)), 
                            'FLAG_TARJETA_ID_FISCAL': int(row.get('FLAG_TARJETA_ID_FISCAL', 0)), 
                            'FLAG_CERTIFICADO_LABORAL': int(row.get('FLAG_CERTIFICADO_LABORAL', 0))
                        }

                        # OHE Family
                        fs = row['FAMILY_STATUS']
                        d['FAMILY_STATUS_Single_or_not_married'] = 1 if fs == 'Single / not married' else 0
                        d['FAMILY_STATUS_Married'] = 1 if fs == 'Married' else 0
                        d['FAMILY_STATUS_Civil_marriage'] = 1 if fs == 'Civil marriage' else 0
                        d['FAMILY_STATUS_Separated'] = 1 if fs == 'Separated' else 0
                        d['FAMILY_STATUS_Widow'] = 1 if fs == 'Widow' else 0

                        # OHE Housing
                        ht = row['HOUSING']
                        d['HOUSING_TYPE_House_or_apartment'] = 1 if ht == 'House / apartment' else 0
                        d['HOUSING_TYPE_Co_op_apartment'] = 1 if ht == 'Co-op apartment' else 0
                        d['HOUSING_TYPE_With_parents'] = 1 if ht == 'With parents' else 0
                        d['HOUSING_TYPE_Rented_apartment'] = 1 if ht == 'Rented apartment' else 0
                        d['HOUSING_TYPE_Municipal_apartment'] = 1 if ht == 'Municipal apartment' else 0
                        d['HOUSING_TYPE_Office_apartment'] = 1 if ht == 'Office apartment' else 0

                        # OHE Income
                        it = row['INCOME_TYPE']
                        d['INCOME_TYPE_Alta_Estabilidad'] = 1 if it in ['Working', 'State servant'] else 0
                        d['INCOME_TYPE_Media_Estabilidad'] = 1 if it in ['Commercial associate', 'Businessman'] else 0
                        d['INCOME_TYPE_Baja_Estabilidad'] = 1 if it in ['Maternity leave', 'Student', 'Unemployed'] else 0
                        d['INCOME_TYPE_Pensionista'] = 1 if it == 'Pensioner' else 0

                        # Raw data para supabase
                        raw_input = {
                            'SK_ID_CURR': row['SK_ID_CURR'], 'NAME': row['NAME'], 'GENDER': row['GENDER'], 
                            'AGE': row['AGE'], 'CNT_CHILDREN_MAPPED': maps['children'][str(row['CNT_CHILDREN'])],
                            'NAME_EDUCATION_TYPE': row['EDUCATION'], 'FAMILY_STATUS': fs,
                            'HOUSING_TYPE': ht, 'INCOME_TYPE': it,
                            'AMT_INCOME_TOTAL': row['AMT_INCOME'], 'AMT_CREDIT': row['AMT_CREDIT'],
                            'YEARS_ACTUAL_WORK': row['YEARS_WORKED'],
                            'FLAG_OWN_REALTY': row['OWN_REALTY'], 'FLAG_OWN_CAR': row['OWN_CAR'],
                            'FLAG_PHONE': d['FLAG_PHONE'], 'FLAG_DNI': d['FLAG_DNI'], 'FLAG_PASAPORTE': d['FLAG_PASAPORTE'],
                            'FLAG_COMPROBANTE_DOM_FISCAL': d['FLAG_COMPROBANTE_DOM_FISCAL'], 
                            'FLAG_ESTADO_CUENTA_BANC': d['FLAG_ESTADO_CUENTA_BANC'],
                            'FLAG_TARJETA_ID_FISCAL': d['FLAG_TARJETA_ID_FISCAL'], 
                            'FLAG_CERTIFICADO_LABORAL': d['FLAG_CERTIFICADO_LABORAL']
                        }

                        # Predecir
                        pred, _ = process_single_prediction(d, raw_input)
                        
                        status = "✅ Aprobado" if pred == 0 else "❌ Denegado" if pred == 1 else "⚠️ ID no encontrado"
                        results_log.append({"ID": row['SK_ID_CURR'], "Nombre": row['NAME'], "Resultado": status})
                    
                    except Exception as e:
                        results_log.append({"ID": row['SK_ID_CURR'], "Nombre": row['NAME'], "Resultado": f"Error: {str(e)}"})

                    progress_bar.progress((index + 1) / len(edited_df))
                
                st.success("Proceso completado.")
                st.table(pd.DataFrame(results_log))

# ------------------------------------------------------
# 4. ENRUTAMIENTO PRINCIPAL
# ------------------------------------------------------

if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "about":
    page_about()
elif st.session_state.page == "request":
    page_credit_request()
