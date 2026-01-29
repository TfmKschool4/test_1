import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from supabase import create_client

# ------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE RECURSOS
# ------------------------------------------------------

st.set_page_config(page_title="Credit Scoring Risk", layout="wide")

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
    Procesa una única solicitud (diccionario pre-procesado), hace el merge, predice y guarda.
    """
    df_datos = pd.DataFrame([datos_solicitante])
    
    # Merge con datos internos
    datos_completos = df_datos.merge(datos_internos_df, on='SK_ID_CURR')
    
    if datos_completos.empty:
        return None, "ID no encontrado en base interna"

    # Columnas requeridas por el modelo
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
    
    datos_completos = datos_completos[columnas_modelo]
    
    # Predicción
    X = datos_completos.drop(['SK_ID_CURR', 'NAME'], axis=1)
    prediction = model_final.predict(X)[0] # Tomamos el valor escalar
    
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
        # No mostramos success aquí para no saturar si es masivo, se maneja fuera
    except Exception as e:
        st.error(f"Error Supabase ID {raw_data['SK_ID_CURR']}: {e}")

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
    st.markdown("<h1 style='text-align: center;'>Credit Scoring Risk</h1>", unsafe_allow_html=True)
    
    # He eliminado la línea que causaba error y dejado solo el código que muestra la imagen
    st.markdown(
        """
        <div style="display: flex; justify_content: center; margin-bottom: 30px;">
            <img src="https://img.freepik.com/vector-gratis/ilustracion-concepto-puntuacion-credito_114360-16474.jpg" width="400">
        </div>
        """, 
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("Bienvenido a la plataforma de evaluación de riesgo crediticio.")
        
        # Botones grandes
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👥 Sobre Nosotros", use_container_width=True):
                go_to_page("about")
                st.rerun()
        with c2:
            if st.button("💳 Solicitar Crédito", use_container_width=True):
                go_to_page("request")
                st.rerun()
                
def page_about():
    st.button("⬅️ Volver al Inicio", on_click=go_to_page, args=("home",))
    st.title("Sobre Nosotros")

    # 🔽 MODIFICADO: TEXTO CORPORATIVO
    st.markdown("""
    Somos una plataforma especializada en **analítica avanzada y evaluación de riesgo crediticio**, diseñada para apoyar la toma de decisiones financieras mediante el uso de **modelos predictivos y técnicas de Machine Learning**.

    Nuestra solución analiza de forma integral variables financieras, laborales y demográficas con el objetivo de **estimar la probabilidad de impago** de un solicitante y proporcionar recomendaciones objetivas, consistentes y escalables para la concesión de crédito.

    El sistema está pensado para integrarse en procesos reales de evaluación crediticia, permitiendo tanto el análisis **individual** como el **procesamiento masivo de solicitudes**, con trazabilidad de resultados y almacenamiento histórico de decisiones.

    Creemos en el uso responsable de la tecnología para impulsar **decisiones financieras más inteligentes, eficientes y basadas en datos**, reduciendo la incertidumbre y mejorando la gestión del riesgo.

    > *La tecnología al servicio de decisiones financieras más seguras y eficientes.*
    """)
    #  - Opcional

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
                    # Preparar datos fila por fila
                    try:
                        # 1. Mapeos básicos
                        age_bin = pd.cut([row['AGE']], bins=bins, labels=labels, right=True, include_lowest=True).to_list()[0]
                        raw_input = {
                        'FLAG_PHONE': row['FLAG_PHONE'],
                        'FLAG_DNI': row['FLAG_DNI'],
                        'FLAG_PASAPORTE': row['FLAG_PASAPORTE'],
                        'FLAG_CERTIFICADO_LABORAL': row['FLAG_CERTIFICADO_LABORAL'],
                        'FLAG_COMPROBANTE_DOM_FISCAL': row['FLAG_COMPROBANTE_DOM_FISCAL'],
                        'FLAG_ESTADO_CUENTA_BANC': row['FLAG_ESTADO_CUENTA_BANC'],
                        'FLAG_TARJETA_ID_FISCAL': row['FLAG_TARJETA_ID_FISCAL'],
                        }

                        
                        # Construir diccionario solicitante (misma lógica que individual)
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
                            'FLAG_OWN_CAR': int(row['OWN_CAR']), 'FLAG_OWN_REALTY': int(row['OWN_REALTY']),
                            # Asumimos que si marcó "Docs OK", tiene todo. Si no, ajustar según necesidad.
                            'FLAG_PHONE': 1, 'FLAG_DNI': docs_ok, 'FLAG_PASAPORTE': docs_ok, 
                            'FLAG_COMPROBANTE_DOM_FISCAL': docs_ok, 'FLAG_ESTADO_CUENTA_BANC': docs_ok, 
                            'FLAG_TARJETA_ID_FISCAL': docs_ok, 'FLAG_CERTIFICADO_LABORAL': docs_ok
                        }

                        # OHE Family (Lógica simplificada para tabla)
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
                            'FLAG_PHONE': 1, 'FLAG_DNI': docs_ok, 'FLAG_PASAPORTE': docs_ok,
                            'FLAG_COMPROBANTE_DOM_FISCAL': docs_ok, 'FLAG_ESTADO_CUENTA_BANC': docs_ok,
                            'FLAG_TARJETA_ID_FISCAL': docs_ok, 'FLAG_CERTIFICADO_LABORAL': docs_ok
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
