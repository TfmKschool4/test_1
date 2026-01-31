import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from supabase import create_client

# ------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE RECURSOS
# ------------------------------------------------------

st.set_page_config(page_title="Creditum", layout="wide")

# Función para cargar recursos (con caché para no recargar en cada interacción)
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
        datos_internos = pd.read_csv('datos_internos.csv', index_col=0)
    except FileNotFoundError:
        st.error("Error: No se encuentra el archivo 'datos_internos.csv'.")
        datos_internos = None
        
    return model, datos_internos

# Conexión Supabase
try:
    url = os.environ['SUPABASE_URL']
    key = os.environ['SUPABASE_KEY']
    supabase = create_client(url, key)
except KeyError:
    st.warning("Variables de entorno de Supabase no configuradas. La funcionalidad de guardado no funcionará.")
    supabase = None

model_final, datos_internos_df = load_resources()

# ------------------------------------------------------
# 2. FUNCIONES DE UTILIDAD Y LÓGICA DE NEGOCIO
# ------------------------------------------------------

def process_single_prediction(datos_solicitante, raw_input_data):
    """
    Procesa solicitud, predice y gestiona el guardado en CSV.
    """
    df_datos = pd.DataFrame([datos_solicitante])
    
    # Merge con datos internos (Bureau)
    datos_completos = df_datos.merge(datos_internos_df, on='SK_ID_CURR')
    
    if datos_completos.empty:
        return None, "ID no encontrado en base interna (Bureau)"
    
    # Columnas del modelo (sin ID ni NAME)
    columnas_modelo = list(model_final.feature_names_in_)

    # Construir X asegurando compatibilidad total con el modelo
    X = datos_completos.copy()

    # Eliminar columnas que el modelo no usa
    for col in ['SK_ID_CURR', 'NAME']:
        if col in X.columns:
            X = X.drop(col, axis=1)

    # Añadir columnas faltantes
    for col in columnas_modelo:
        if col not in X.columns:
            X[col] = 0

    # Eliminar columnas extra y ordenar
    X = X[columnas_modelo]

    # Predicción
    prediction = model_final.predict(X)[0]

    # Guardado en CSV
    saved_ok, saved_msg = save_to_csv(raw_input_data, datos_completos, prediction)

    return prediction, datos_completos

# ------------------------------------------------------
# FUNCIÓN DE GUARDADO EN CSV (REEMPLAZA A SUPABASE)
# ------------------------------------------------------

def save_to_csv(raw_data, datos_completos, prediction):
    filename = 'historial_creditos.csv'
    current_id = int(raw_data['SK_ID_CURR'])

    # 1. Verificar duplicados
    if os.path.exists(filename):
        df_history = pd.read_csv(filename)
        if current_id in df_history['SK_ID_CURR'].values:
            return False, f"El ID {current_id} ya existe en el historial."

    # 2. Mapear datos (Asegúrate de que las llaves coincidan con raw_input_data del formulario)
    new_loan_variables = {
        'SK_ID_CURR': current_id,
        'NAME': raw_data.get('NAME', 'N/A'),
        'GENERO': raw_data.get('GENDER', 'N/A'),
        'EDAD': raw_data.get('AGE', 0),
        'INGRESOS': raw_data.get('AMT_INCOME_TOTAL', 0),
        'CREDITO_SOLICITADO': raw_data.get('AMT_CREDIT', 0),
        'RESULTADO_PREDICCION': "APROBADO" if prediction == 0 else "DENEGADO",
        'FECHA_REGISTRO': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    df_new_row = pd.DataFrame([new_loan_variables])

    # 3. Guardar
    try:
        if not os.path.exists(filename):
            df_new_row.to_csv(filename, index=False, mode='w')
        else:
            df_new_row.to_csv(filename, index=False, mode='a', header=False)
        return True, "Datos guardados en el historial local."
    except Exception as e:
        return False, f"Error al guardar: {str(e)}"
        
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

import os # <--- ASEGÚRATE DE IMPORTAR ESTO AL PRINCIPIO JUNTO A LOS OTROS IMPORTS
import base64

def page_home():
# --- 1. CSS ESTILO REFINADO ---
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

    /* CONTENEDOR PRINCIPAL (Efecto Cristal Transparente) */
    .header-box {
        background-color: rgba(255, 255, 255, 0.7); /* Reducimos la opacidad a 0.7 */
        backdrop-filter: blur(10px); /* Esto crea el efecto de vidrio esmerilado */
        -webkit-backdrop-filter: blur(10px); 
        border-radius: 20px;
        padding: 40px 20px;
        margin: 20px auto;
        max-width: 650px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); /* Sombra más sutil */
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3); /* Borde suave */
    }

    .logo-img {
        max-width: 120px;
        height: auto;
        margin-bottom: 15px;
        /* Quitamos el filtro pesado para que se vea limpio */
    }

    .custom-title {
        color: #000000 !important; /* Negro puro para máximo contraste sobre el cristal */
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.5); /* Sutil relieve */
    }
    </style>

    """, unsafe_allow_html=True)

    # --- 2. LÓGICA DE IMAGEN ---
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            return None

    img_b64 = get_base64_image("logo.png") 
    logo_html = f'<img src="data:image/png;base64,{img_b64}" class="logo-img">' if img_b64 else ""

    # --- 3. RENDERIZADO DEL HEADER ---
    st.markdown(f"""
    <div class="header-box">
        {logo_html}
        <h2 class="custom-title">Análisis inteligente del riesgo crediticio</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- 4. BOTONES DE ACCIÓN CENTRADOS Y APILADOS ---
    # Usamos columnas para centrar el bloque de botones en el medio de la página
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        # BLOQUE 1: EVALUACIÓN (Arriba)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; font-size: 2.2rem;'>🚀 Evaluación</h2>", unsafe_allow_html=True)
            if st.button("💳 Solicitar Crédito", use_container_width=True, type="primary"):
                go_to_page("request")
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)

        # BLOQUE 2: SOBRE NOSOTROS (Debajo)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; font-size: 2.2rem;'>🏢 Sobre nosotros</h2>", unsafe_allow_html=True)
            if st.button("👥 Quiénes Somos", use_container_width=True):
                go_to_page("about")
                st.rerun()
                
def page_about():
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    
    # Título más grande
    st.markdown("<h1 style='font-size: 3rem;'>Sobre Nosotros</h1>", unsafe_allow_html=True)

    # Contenedor con tamaño de letra personalizado
    # Ajusta '22px' al tamaño que prefieras
    st.markdown("""
    <div style="font-size: 22px; line-height: 1.6; text-align: justify;">
    
    Somos una plataforma especializada en <b>analítica avanzada y evaluación de riesgo crediticio</b>, 
    diseñada para apoyar la toma de decisiones financieras mediante el uso de <b>modelos predictivos</b>.
    <br><br>
    Nuestra solución analiza de forma integral variables financieras, laborales y demográficas con el 
    objetivo de <b>estimar la probabilidad de impago</b> de un solicitante y proporcionar recomendaciones 
    objetivas, consistentes y escalables para la concesión de crédito.
    <br><br>
    El sistema está pensado para integrarse en procesos reales de evaluación crediticia, permitiendo 
    tanto el análisis <b>individual</b> como el <b>procesamiento masivo de solicitudes</b>, con 
    trazabilidad de resultados y almacenamiento histórico de decisiones.
    <br><br>
    Creemos en el uso responsable de la tecnología para impulsar <b>decisiones financieras más 
    inteligentes, eficientes y basadas en datos</b>, reduciendo la incertidumbre y mejorando la gestión del riesgo.
    
    <blockquote style="font-size: 24px; font-style: italic; border-left: 5px solid #ff4b4b; padding-left: 15px; margin-top: 20px;">
    "La tecnología al servicio de decisiones financieras más seguras y eficientes."
    </blockquote>
    
    </div>
    """, unsafe_allow_html=True)

def page_credit_request():
    # --- CSS ESPECÍFICO PARA ESTA PÁGINA ---
    st.markdown("""
    <style>
        /* Tamaño de los títulos de las pestañas (Tabs) */
        button[data-baseweb="tab"] div {
            font-size: 20px !important;
        }
        /* Tamaño de las etiquetas de los campos (Labels) */
        label p {
            font-size: 1.2rem !important;
            font-weight: bold !important;
        }
        /* Tamaño del texto dentro de los campos de entrada */
        input {
            font-size: 1.1rem !important;
        }
        /* Tamaño del texto de los botones de esta página */
        .stButton button {
            font-size: 1.3rem !important;
            height: 3em !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    
    # Título principal aumentado
    st.markdown("<h1 style='font-size: 3rem;'>Solicitud de Crédito</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👤 Individual", "👥 Múltiples Solicitantes"])

    # -------------------------
    # CASO 1: INDIVIDUAL
    # -------------------------
    with tab1:
        # Subtítulo aumentado
        st.markdown("<h2 style='font-size: 1.8rem;'>Formulario Individual</h2>", unsafe_allow_html=True)
        
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
                # NOTA: Para no repetir código extenso, simplificamos la lógica de asignación
                # asumiendo que el modelo espera exactamente los nombres de columnas de tu código original.
                
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
                    col_name = f"FAMILY_STATUS_{f.replace(' / ', '_or_').replace(' ', '_')}" # Ajuste manual para coincidir con tu key original si difiere
                    # Usando tus keys exactas:
                    if f == 'Single / not married': k = 'FAMILY_STATUS_Single_or_not_married'
                    elif f == 'Civil marriage': k = 'FAMILY_STATUS_Civil_marriage'
                    else: k = f"FAMILY_STATUS_{f}"
                    datos_solicitante[k] = 1 if FAMILY_STATUS == f else 0

                # OHE Housing
                hous_opts = ['With parents', 'Rented apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment']
                for h in hous_opts:
                    if h == 'House / apartment': k = 'HOUSING_TYPE_House_or_apartment'
                    elif h == 'Co-op apartment': k = 'HOUSING_TYPE_Co_op_apartment' # Ajuste guion
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
        st.markdown("<h2 style='font-size: 1.8rem;'>Carga Masiva de Solicitudes</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.2rem;'>Añada filas a la tabla a continuación. Puede copiar y pegar desde Excel.</p>", unsafe_allow_html=True)

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
            # Simplificamos algunos flags documentales para que la tabla no sea kilométrica, 
            # asumiendo True por defecto o añadiendo solo los críticos. Añade más si es necesario.
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
                    try:
                        # 1. Mapeos básicos y validación de edad
                        age_val = row['AGE'] if pd.notnull(row['AGE']) else 18
                        age_bin = pd.cut([age_val], bins=bins, labels=labels, right=True, include_lowest=True).to_list()[0]
                        
                        # 2. Construir diccionario solicitante (Para el modelo)
                        # Usamos int(row['COL']) para los checkboxes porque el modelo suele esperar 0 o 1
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
                            'YEARS_ACTUAL_WORK': float(row['YEARS_WORKED']) if pd.notnull(row['YEARS_WORKED']) else 0.0,
                            'FLAG_OWN_CAR': int(row['OWN_CAR']), 
                            'FLAG_OWN_REALTY': int(row['OWN_REALTY']),
                            'FLAG_PHONE': int(row['FLAG_PHONE']), 
                            'FLAG_DNI': int(row['FLAG_DNI']), 
                            'FLAG_PASAPORTE': int(row['FLAG_PASAPORTE']), 
                            'FLAG_COMPROBANTE_DOM_FISCAL': int(row['FLAG_COMPROBANTE_DOM_FISCAL']), 
                            'FLAG_ESTADO_CUENTA_BANC': int(row['FLAG_ESTADO_CUENTA_BANC']), 
                            'FLAG_TARJETA_ID_FISCAL': int(row['FLAG_TARJETA_ID_FISCAL']), 
                            'FLAG_CERTIFICADO_LABORAL': int(row['FLAG_CERTIFICADO_LABORAL'])
                        }

                        # 3. OHE Family y Housing (Igual que tenías, pero asegurando los nombres)
                        fs = row['FAMILY_STATUS']
                        d['FAMILY_STATUS_Single_or_not_married'] = 1 if fs == 'Single / not married' else 0
                        d['FAMILY_STATUS_Married'] = 1 if fs == 'Married' else 0
                        d['FAMILY_STATUS_Civil_marriage'] = 1 if fs == 'Civil marriage' else 0
                        d['FAMILY_STATUS_Separated'] = 1 if fs == 'Separated' else 0
                        d['FAMILY_STATUS_Widow'] = 1 if fs == 'Widow' else 0

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

                        # 4. Raw data para guardar (Se usa para el historial CSV)
                        raw_input_data = d.copy() # Simplificamos usando d o mapeando strings de nuevo

                        # 5. Predecir
                        pred, _ = process_single_prediction(d, raw_input_data)
                        
                        status = "✅ Aprobado" if pred == 0 else "❌ Denegado" if pred == 1 else "⚠️ Error ID"
                        results_log.append({"ID": row['SK_ID_CURR'], "Nombre": row['NAME'], "Resultado": status})
                    
                    except Exception as e:
                        results_log.append({"ID": row.get('SK_ID_CURR', 'N/A'), "Nombre": row.get('NAME', 'N/A'), "Resultado": f"Error: {str(e)}"})

                    progress_bar.progress((index + 1) / len(edited_df))
                
                st.success("Proceso completado.")
                st.table(pd.DataFrame(results_log))


# ------------------------------------------------------
# FUNCIONES DE GESTIÓN DE HISTORIAL (CSV)
# ------------------------------------------------------

def delete_id_from_csv(id_to_delete):
    filename = 'historial_creditos.csv'
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            # Aseguramos que el ID sea numérico para la comparación
            id_int = int(id_to_delete)
            if id_int in df['SK_ID_CURR'].values:
                df = df[df['SK_ID_CURR'] != id_int]
                df.to_csv(filename, index=False)
                return True, f"ID {id_to_delete} eliminado correctamente."
            return False, "ID no encontrado en el historial."
        except ValueError:
            return False, "Por favor, introduce un ID numérico válido."
    return False, "No existe archivo de historial."

def clear_history():
    filename = 'historial_creditos.csv'
    if os.path.exists(filename):
        os.remove(filename)
        return True
    return False

def render_history_section():
    """Renderiza el historial y las opciones de borrado dentro de un expander"""
    with st.expander("📊 Ver historial de solicitudes (CSV)"):
        if os.path.exists('historial_creditos.csv'):
            historial_df = pd.read_csv('historial_creditos.csv')
            st.dataframe(historial_df, use_container_width=True)
            
            st.divider()
            st.subheader("🛠️ Gestión de Registros")
            
            col_del1, col_del2 = st.columns([2, 1])
            
            with col_del1:
                id_borrar = st.text_input("ID específico a eliminar", key="input_del_id")
                if st.button("Eliminar Registro Individual", use_container_width=True):
                    success, msg = delete_id_from_csv(id_borrar)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            
            with col_del2:
                st.write("Zona de Peligro")
                if st.button("🗑️ Borrar TODO el historial", type="secondary", use_container_width=True):
                    if clear_history():
                        st.success("Historial eliminado.")
                        st.rerun()
                    else:
                        st.error("No hay historial que borrar.")
        else:
            st.info("Aún no hay registros guardados en el historial local.")

# ------------------------------------------------------
# VISTAS ACTUALIZADAS
# ------------------------------------------------------

def page_credit_request():
    # Estilos CSS (se mantienen igual que tu código)
    st.markdown("""<style>...</style>""", unsafe_allow_html=True)
    
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    st.markdown("<h1 style='font-size: 3rem;'>Solicitud de Crédito</h1>", unsafe_allow_html=True)

    # --- NUEVA UBICACIÓN DEL HISTORIAL ---
    # Se muestra aquí para que sea visible tanto en Individual como en Múltiple
    render_history_section()
    
    st.divider()

    tab1, tab2 = st.tabs(["👤 Individual", "👥 Múltiples Solicitantes"])

    with tab1:
        st.markdown("<h2 style='font-size: 1.8rem;'>Formulario Individual</h2>", unsafe_allow_html=True)
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
                # NOTA: Para no repetir código extenso, simplificamos la lógica de asignación
                # asumiendo que el modelo espera exactamente los nombres de columnas de tu código original.
                
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
                    col_name = f"FAMILY_STATUS_{f.replace(' / ', '_or_').replace(' ', '_')}" # Ajuste manual para coincidir con tu key original si difiere
                    # Usando tus keys exactas:
                    if f == 'Single / not married': k = 'FAMILY_STATUS_Single_or_not_married'
                    elif f == 'Civil marriage': k = 'FAMILY_STATUS_Civil_marriage'
                    else: k = f"FAMILY_STATUS_{f}"
                    datos_solicitante[k] = 1 if FAMILY_STATUS == f else 0

                # OHE Housing
                hous_opts = ['With parents', 'Rented apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment']
                for h in hous_opts:
                    if h == 'House / apartment': k = 'HOUSING_TYPE_House_or_apartment'
                    elif h == 'Co-op apartment': k = 'HOUSING_TYPE_Co_op_apartment' # Ajuste guion
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

    with tab2:
        st.markdown("<h2 style='font-size: 1.8rem;'>Carga Masiva de Solicitudes</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.2rem;'>Añada filas a la tabla a continuación. Puede copiar y pegar desde Excel.</p>", unsafe_allow_html=True)

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
            # Simplificamos algunos flags documentales para que la tabla no sea kilométrica, 
            # asumiendo True por defecto o añadiendo solo los críticos. Añade más si es necesario.
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
                    try:
                        # 1. Mapeos básicos y validación de edad
                        age_val = row['AGE'] if pd.notnull(row['AGE']) else 18
                        age_bin = pd.cut([age_val], bins=bins, labels=labels, right=True, include_lowest=True).to_list()[0]
                        
                        # 2. Construir diccionario solicitante (Para el modelo)
                        # Usamos int(row['COL']) para los checkboxes porque el modelo suele esperar 0 o 1
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
                            'YEARS_ACTUAL_WORK': float(row['YEARS_WORKED']) if pd.notnull(row['YEARS_WORKED']) else 0.0,
                            'FLAG_OWN_CAR': int(row['OWN_CAR']), 
                            'FLAG_OWN_REALTY': int(row['OWN_REALTY']),
                            'FLAG_PHONE': int(row['FLAG_PHONE']), 
                            'FLAG_DNI': int(row['FLAG_DNI']), 
                            'FLAG_PASAPORTE': int(row['FLAG_PASAPORTE']), 
                            'FLAG_COMPROBANTE_DOM_FISCAL': int(row['FLAG_COMPROBANTE_DOM_FISCAL']), 
                            'FLAG_ESTADO_CUENTA_BANC': int(row['FLAG_ESTADO_CUENTA_BANC']), 
                            'FLAG_TARJETA_ID_FISCAL': int(row['FLAG_TARJETA_ID_FISCAL']), 
                            'FLAG_CERTIFICADO_LABORAL': int(row['FLAG_CERTIFICADO_LABORAL'])
                        }

                        # 3. OHE Family y Housing (Igual que tenías, pero asegurando los nombres)
                        fs = row['FAMILY_STATUS']
                        d['FAMILY_STATUS_Single_or_not_married'] = 1 if fs == 'Single / not married' else 0
                        d['FAMILY_STATUS_Married'] = 1 if fs == 'Married' else 0
                        d['FAMILY_STATUS_Civil_marriage'] = 1 if fs == 'Civil marriage' else 0
                        d['FAMILY_STATUS_Separated'] = 1 if fs == 'Separated' else 0
                        d['FAMILY_STATUS_Widow'] = 1 if fs == 'Widow' else 0

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

                        # 4. Raw data para guardar (Se usa para el historial CSV)
                        raw_input_data = d.copy() # Simplificamos usando d o mapeando strings de nuevo

                        # 5. Predecir
                        pred, _ = process_single_prediction(d, raw_input_data)
                        
                        status = "✅ Aprobado" if pred == 0 else "❌ Denegado" if pred == 1 else "⚠️ Error ID"
                        results_log.append({"ID": row['SK_ID_CURR'], "Nombre": row['NAME'], "Resultado": status})
                    
                    except Exception as e:
                        results_log.append({"ID": row.get('SK_ID_CURR', 'N/A'), "Nombre": row.get('NAME', 'N/A'), "Resultado": f"Error: {str(e)}"})

                    progress_bar.progress((index + 1) / len(edited_df))
                
                st.success("Proceso completado.")
                st.table(pd.DataFrame(results_log))


# ------------------------------------------------------
# ENRUTAMIENTO PRINCIPAL (LIMPIO)
# ------------------------------------------------------

if 'page' not in st.session_state:
    st.session_state.page = "home"

# Eliminamos el checkbox global que estaba fuera de las funciones
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "about":
    page_about()
elif st.session_state.page == "request":
    page_credit_request()
