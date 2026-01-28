import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

with open('model_final.pkl', 'rb') as file:
    model_final = pickle.load(file)

from supabase import create_client

url = os.environ['SUPABASE_URL']
key = os.environ['SUPABASE_KEY']
supabase = create_client(url, key)


datos_internos = pd.read_csv('datos_internos.csv', index_col=0)

def main():
    st.title('Formulario del préstamo')
    
    # ID del cliente
    SK_ID_CURR = st.text_input('ID del solicitante')
    if SK_ID_CURR == '':
        st.warning('Debes introducir un ID.')
    else:
        try:
            SK_ID_CURR_int = int(SK_ID_CURR)  # Intentamos convertir a entero
            if SK_ID_CURR_int < 0:
                st.error('El ID debe ser un número entero positivo o 0.')
            else:
                st.success(f'ID del solicitante: {SK_ID_CURR_int}')
        except ValueError:
            st.error('El ID debe ser un número entero válido, no texto.')

    # Nombre del cliente
    NAME = st.text_input('Nombre del solicitante')
    if NAME == '':
        st.warning('Debes introducir un nombre.')
    else:
        if NAME.isalpha():
            st.success(f'El solicitante es: {NAME}')
        else:
            st.error('Debes introducir un nombre válido.')

    # Edad del cliente
    AGES = st.slider(
        'Edad:',
        min_value = 18,
        max_value = 100,
        value = 18,
        step = 1
    )
    labels = [1, 2, 3, 4]
    bins = [18, 34, 43, 54, 100]

    AGE_BINS = pd.cut([AGES], bins = bins, labels = labels, right=True, include_lowest=True).to_list()[0]

    # Género del cliente
    GENDER = st.selectbox(
        'Género del solicitante:',
        ['Masculino', 'Femenino']
    )
    GENDER_M = 1 if GENDER == 'Masculino' else 0
    GENDER_F = 1 if GENDER == 'Femenino' else 0
    
    CODE_GENDER = 'M' if GENDER == 'Masculino' else 'F'

    # Hijos
    CNT_CHILDREN = st.selectbox(
        'Número de hijos:',
        ['0','1', '2', '3', '4 o más']
    )
    map_children = {
        '0':0,
        '1':1,
        '2':2,
        '3':3,
        '4 o más':4
    }
    CNT_CHILDREN = map_children[CNT_CHILDREN]

    # Nivel de estudios
    NAME_EDUCATION_TYPE = st.selectbox(
        'Nivel de estudios:',
        ['Lower secondary',
        'Secondary / secondary special', 
        'Incomplete higher',
        'Higher education',
        'Academic degree']
    )
    map_studies = {
        'Lower secondary':0,
        'Secondary / secondary special':1,
        'Incomplete higher':2,
        'Higher education':3,
        'Academic degree':4
    }
    LEVEL_EDUCATION_TYPE = map_studies[NAME_EDUCATION_TYPE]

    FAMILY_STATUS = st.selectbox(
        'Situación familiar del solicitante:',
        ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    )
    FAMILY_STATUS_Single_or_not_married = 1 if FAMILY_STATUS == 'Single / not married' else 0
    FAMILY_STATUS_Married = 1 if FAMILY_STATUS == 'Married' else 0
    FAMILY_STATUS_Civil_marriage = 1 if FAMILY_STATUS == 'Civil marriage' else 0
    FAMILY_STATUS_Separated = 1 if FAMILY_STATUS == 'Separated' else 0
    FAMILY_STATUS_Widow = 1 if FAMILY_STATUS == 'Widow' else 0

    HOUSING_TYPE = st.selectbox(
        'Tipo de vivienda del solicitante:',
        ['With parents', 'Rented apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment']
    )

    HOUSING_TYPE_With_parents = 1 if HOUSING_TYPE == 'With parents' else 0
    HOUSING_TYPE_Rented_apartment = 1 if HOUSING_TYPE == 'Rented apartment' else 0
    HOUSING_TYPE_House_or_apartment = 1 if HOUSING_TYPE == 'House / apartment' else 0
    HOUSING_TYPE_Municipal_apartment = 1 if HOUSING_TYPE == 'Municipal apartment' else 0
    HOUSING_TYPE_Office_apartment = 1 if HOUSING_TYPE == 'Office apartment' else 0
    HOUSING_TYPE_Co_op_apartment = 1 if HOUSING_TYPE == 'Co-op apartment' else 0

    # Ingresos
    AMT_INCOME_TOTAL = st.number_input(
        'Ingresos del solicitante:',
        min_value = 0.00,
        value = 0.00,
        step = 100.00
        )

    # Fuente de ingresos
    INCOME_TYPE = st.selectbox(
        'Indica la fuente de ingresos del solitante:',
        ['Working', 'State servant', 'Commercial associate', 'Businessman', 'Maternity leave', 'Student', 'Unemployed', 'Pensioner']
    )
    INCOME_TYPE_Alta_Estabilidad = 1 if INCOME_TYPE == 'Working' or INCOME_TYPE == 'State servant' else 0
    INCOME_TYPE_Media_Estabilidad = 1 if INCOME_TYPE == 'Commercial associate' or INCOME_TYPE == 'Businessman' else 0
    INCOME_TYPE_Baja_Estabilidad = 1 if INCOME_TYPE == 'Maternity leave' or INCOME_TYPE == 'Student' or INCOME_TYPE == 'Unemployed' else 0
    INCOME_TYPE_Pensionista = 1 if INCOME_TYPE == 'Pensioner' else 0

    # Años en su trabajo actual
    YEARS_ACTUAL_WORK = st.text_input('Años en su actual puesto de trabajo:')
    if YEARS_ACTUAL_WORK == '':
        st.warning('Dejar el campo vacío implica que el solicitante es desempleado o jubilado.')
        # YEARS_ACTUAL_WORK = np.nan
    else:
        try:
            YEARS_ACTUAL_WORK = float(YEARS_ACTUAL_WORK)
            if YEARS_ACTUAL_WORK < 0:
                st.error('Debes escribir un número positivo.')
            else:
                pass
        except ValueError:
            st.error('Debes escribir un número entero positivo, no texto.')

    # Casa propia
    FLAG_OWN_REALTY = st.checkbox('¿El cliente posee casa propia?')
    
    # Carro propio
    FLAG_OWN_CAR = st.checkbox('¿El cliente posee carro propio?')

    # Teléfono
    FLAG_PHONE = st.checkbox('¿Ha proporcionado su número de teléfono?')

    # DNI
    FLAG_DNI = st.checkbox('¿Ha entregado el DNI?')

    # Pasaporte
    FLAG_PASAPORTE = st.checkbox('¿Ha entregado el pasaporte?')

    # Comprobante de domicilio fiscal
    FLAG_COMPROBANTE_DOM_FISCAL = st.checkbox('¿Ha entregado su comprobante de domicilio fiscal?')

    # Comprobante del estado de la cuenta bancaria
    FLAG_ESTADO_CUENTA_BANC = st.checkbox('¿Ha entregado el estado de su cuenta bancaria?')

    # Tarjeta de Identificación Fiscal
    FLAG_TARJETA_ID_FISCAL = st.checkbox('¿Ha entregado su tarjeta de identificación fiscal?')

    # Certificado laboral
    FLAG_CERTIFICADO_LABORAL = st.checkbox('¿Ha entregado su certificado laboral?')

    # Importe del crédito
    AMT_CREDIT = st.number_input(
        'Crédito solicitado:',
        min_value = 0.0,
        value = 0.0,
        step = 100.0
    )

    if st.button('Guardar datos'):
        # Ver si hay errores:
        errors = []
        
        #Error del ID. 
        try:
            SK_ID_CURR_int = int(SK_ID_CURR)
            if SK_ID_CURR_int < 0:
                errors.append('El ID del cliente debe ser un número entero positivo o 0.')
        except ValueError:
            errors.append('El ID del cliente debe ser un número entero válido y no puede estar vacío.')
        
        # Error del NAME.
        if NAME == '' or not NAME.isalpha():
            errors.append('El nombre del cliente debe ser válido y no puede estar vacío')

        if len(errors) > 0:
            st.markdown('#### **Se han encontrado los siguientes errores:**')
            for error in errors:
                st.error(error)
        else:
            datos_solicitante = {
                    'SK_ID_CURR': int(SK_ID_CURR),
                    'NAME': str(NAME),
                    'AGE_BINS': int(AGE_BINS),
                    'GENDER_M': int(GENDER_M),
                    'GENDER_F': int(GENDER_F),
                    'CNT_CHILDREN': int(CNT_CHILDREN),
                    'LEVEL_EDUCATION_TYPE': int(LEVEL_EDUCATION_TYPE),
                    'FAMILY_STATUS_Single_or_not_married': int(FAMILY_STATUS_Single_or_not_married),
                    'FAMILY_STATUS_Married': int(FAMILY_STATUS_Married),
                    'FAMILY_STATUS_Civil_marriage': int(FAMILY_STATUS_Civil_marriage),
                    'FAMILY_STATUS_Separated': int(FAMILY_STATUS_Separated),
                    'FAMILY_STATUS_Widow': int(FAMILY_STATUS_Widow),
                    'HOUSING_TYPE_With_parents': int(HOUSING_TYPE_With_parents),
                    'HOUSING_TYPE_Rented_apartment': int(HOUSING_TYPE_Rented_apartment),
                    'HOUSING_TYPE_House_or_apartment': int(HOUSING_TYPE_House_or_apartment),
                    'HOUSING_TYPE_Municipal_apartment': int(HOUSING_TYPE_Municipal_apartment),
                    'HOUSING_TYPE_Office_apartment': int(HOUSING_TYPE_Office_apartment),
                    'HOUSING_TYPE_Co_op_apartment': int(HOUSING_TYPE_Co_op_apartment),
                    'AMT_INCOME_TOTAL': float(AMT_INCOME_TOTAL),
                    'INCOME_TYPE_Alta_Estabilidad': int(INCOME_TYPE_Alta_Estabilidad),
                    'INCOME_TYPE_Media_Estabilidad': int(INCOME_TYPE_Media_Estabilidad),
                    'INCOME_TYPE_Baja_Estabilidad': int(INCOME_TYPE_Baja_Estabilidad),
                    'INCOME_TYPE_Pensionista': int(INCOME_TYPE_Pensionista),
                    'YEARS_ACTUAL_WORK': np.nan if YEARS_ACTUAL_WORK == '' else float(YEARS_ACTUAL_WORK),
                    'FLAG_OWN_REALTY': int(FLAG_OWN_REALTY),
                    'FLAG_OWN_CAR': int(FLAG_OWN_CAR),
                    'FLAG_PHONE': int(FLAG_PHONE),
                    'FLAG_DNI': int(FLAG_DNI),
                    'FLAG_PASAPORTE': int(FLAG_PASAPORTE),
                    'FLAG_COMPROBANTE_DOM_FISCAL': int(FLAG_COMPROBANTE_DOM_FISCAL),
                    'FLAG_ESTADO_CUENTA_BANC': int(FLAG_ESTADO_CUENTA_BANC),
                    'FLAG_TARJETA_ID_FISCAL': int(FLAG_TARJETA_ID_FISCAL),
                    'FLAG_CERTIFICADO_LABORAL': int(FLAG_CERTIFICADO_LABORAL),
                    'AMT_CREDIT': float(AMT_CREDIT)
                }
            df_datos = pd.DataFrame([datos_solicitante])
            st.success('Se han guardado los datos correctamente.')
            st.dataframe(df_datos)

            # Hacemos el merge con datos_internos. Si el ID no es correcto, lanza un error y no ejecuta la predicción.
            datos_completos = df_datos.merge(datos_internos, on='SK_ID_CURR')
            if datos_completos.empty:
                st.markdown('#### **Se han encontrado los siguientes errores:**')
                st.error('El ID del solicitante no se encuentra en la base de datos interna. No se ha podido realizar la predicción.')
                return

            columnas = [
            'SK_ID_CURR',
            'NAME',
            'FLAG_OWN_REALTY',
            'CNT_CHILDREN',
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT',
            'LEVEL_EDUCATION_TYPE',
            'AGE_BINS',
            'YEARS_ACTUAL_WORK',
            'FLAG_PHONE',
            'DEF_30_CNT_SOCIAL_CIRCLE',
            'FLAG_COMPROBANTE_DOM_FISCAL',
            'FLAG_ESTADO_CUENTA_BANC',
            'FLAG_PASAPORTE',
            'FLAG_TARJETA_ID_FISCAL',
            'FLAG_DNI',
            'FLAG_CERTIFICADO_LABORAL',
            'GENDER_F',
            'GENDER_M',
            'INCOME_TYPE_Alta_Estabilidad',
            'INCOME_TYPE_Baja_Estabilidad',
            'INCOME_TYPE_Media_Estabilidad',
            'INCOME_TYPE_Pensionista',
            'FAMILY_STATUS_Civil_marriage',
            'FAMILY_STATUS_Married',
            'FAMILY_STATUS_Separated',
            'FAMILY_STATUS_Single_or_not_married',
            'FAMILY_STATUS_Widow',
            'HOUSING_TYPE_Co_op_apartment',
            'HOUSING_TYPE_House_or_apartment',
            'HOUSING_TYPE_Municipal_apartment',
            'HOUSING_TYPE_Office_apartment',
            'HOUSING_TYPE_Rented_apartment',
            'HOUSING_TYPE_With_parents'
            ]

            datos_completos = datos_completos[columnas]
            st.dataframe(datos_completos)

            # Predicción
            X = datos_completos.drop(['SK_ID_CURR', 'NAME'], axis=1)
            prediction = model_final.predict(X)

            st.write(f'La predicción ha sido: {prediction}')

            # Creo la fila que se añadirá a historial_loans.csv
            new_loan_variables = {
                'SK_ID_CURR':int(SK_ID_CURR),
                'NAME':str(NAME),
                'CODE_GENDER':str(CODE_GENDER),
                'FLAG_OWN_REALTY':int(FLAG_OWN_REALTY),
                'CNT_CHILDREN':int(CNT_CHILDREN),
                'AMT_INCOME_TOTAL':float(AMT_INCOME_TOTAL),
                'AMT_CREDIT':float(AMT_CREDIT),
                'NAME_INCOME_TYPE':str(INCOME_TYPE),
                'NAME_EDUCATION_TYPE':str(NAME_EDUCATION_TYPE),
                'NAME_FAMILY_STATUS':str(FAMILY_STATUS),
                'NAME_HOUSING_TYPE':str(HOUSING_TYPE),
                'AGE':int(AGES),
                'YEARS_ACTUAL_WORK': float(YEARS_ACTUAL_WORK) if YEARS_ACTUAL_WORK != '' else None,
                'FLAG_PHONE':int(FLAG_PHONE),
                'DEF_30_CNT_SOCIAL_CIRCLE':int(datos_completos['DEF_30_CNT_SOCIAL_CIRCLE'][0]),
                'FLAG_COMPROBANTE_DOM_FISCAL':int(FLAG_COMPROBANTE_DOM_FISCAL),
                'FLAG_ESTADO_CUENTA_BANC':int(FLAG_ESTADO_CUENTA_BANC),
                'FLAG_PASAPORTE':int(FLAG_PASAPORTE),
                'FLAG_TARJETA_ID_FISCAL':int(FLAG_TARJETA_ID_FISCAL),
                'FLAG_DNI':int(FLAG_DNI),
                'FLAG_CERTIFICADO_LABORAL':int(FLAG_CERTIFICADO_LABORAL),
                'TARGET':int(prediction)
            }
            new_loan = pd.DataFrame([new_loan_variables])

            st.dataframe(new_loan)

            try:
                response = supabase.table('historical_loans').insert(new_loan_variables).execute()
                
                if response.data is None:
                    st.error('No se ha podido guardar en Supabase.')
                else:
                    st.success('Los datos se han guardado correctamente en Supabase.')

            except Exception as e:
                st.error(f'Ocurrió un error al guardar en Supabase: {e}')

main()
