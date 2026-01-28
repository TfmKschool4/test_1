import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from supabase import create_client

# =========================
# CARGA DEL MODELO
# =========================
with open("model_final.pk", "rb") as file:   # ⚠️ revisa el nombre si falla
    model_final = pickle.load(file)

# =========================
# SUPABASE
# =========================
url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
supabase = create_client(url, key)

# =========================
# DATOS INTERNOS
# =========================
datos_internos = pd.read_csv("datos_internos.csv", index_col=0)


def main():
    st.title("Formulario del préstamo")

    # =========================
    # ID CLIENTE
    # =========================
    SK_ID_CURR = st.text_input("ID del solicitante")

    if SK_ID_CURR == "":
        st.warning("Debes introducir un ID.")
    else:
        try:
            SK_ID_CURR_int = int(SK_ID_CURR)
            if SK_ID_CURR_int < 0:
                st.error("El ID debe ser un número entero positivo o 0.")
            else:
                st.success(f"ID del solicitante: {SK_ID_CURR_int}")
        except ValueError:
            st.error("El ID debe ser un número entero válido.")

    # =========================
    # NOMBRE
    # =========================
    NAME = st.text_input("Nombre del solicitante")

    if NAME == "":
        st.warning("Debes introducir un nombre.")
    elif NAME.isalpha():
        st.success(f"El solicitante es: {NAME}")
    else:
        st.error("Debes introducir un nombre válido.")

    # =========================
    # EDAD
    # =========================
    AGES = st.slider("Edad:", 18, 100, 18, 1)

    labels = [1, 2, 3, 4]
    bins = [18, 34, 43, 54, 100]
    AGE_BINS = pd.cut([AGES], bins=bins, labels=labels, include_lowest=True)[0]

    # =========================
    # GÉNERO
    # =========================
    GENDER = st.selectbox("Género del solicitante:", ["Masculino", "Femenino"])
    GENDER_M = int(GENDER == "Masculino")
    GENDER_F = int(GENDER == "Femenino")
    CODE_GENDER = "M" if GENDER == "Masculino" else "F"

    # =========================
    # HIJOS
    # =========================
    CNT_CHILDREN = st.selectbox("Número de hijos:", ["0", "1", "2", "3", "4 o más"])
    CNT_CHILDREN = {"0": 0, "1": 1, "2": 2, "3": 3, "4 o más": 4}[CNT_CHILDREN]

    # =========================
    # EDUCACIÓN
    # =========================
    NAME_EDUCATION_TYPE = st.selectbox(
        "Nivel de estudios:",
        [
            "Lower secondary",
            "Secondary / secondary special",
            "Incomplete higher",
            "Higher education",
            "Academic degree",
        ],
    )

    LEVEL_EDUCATION_TYPE = {
        "Lower secondary": 0,
        "Secondary / secondary special": 1,
        "Incomplete higher": 2,
        "Higher education": 3,
        "Academic degree": 4,
    }[NAME_EDUCATION_TYPE]

    # =========================
    # ESTADO FAMILIAR
    # =========================
    FAMILY_STATUS = st.selectbox(
        "Situación familiar:",
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    )

    FAMILY_STATUS_Single_or_not_married = int(FAMILY_STATUS == "Single / not married")
    FAMILY_STATUS_Married = int(FAMILY_STATUS == "Married")
    FAMILY_STATUS_Civil_marriage = int(FAMILY_STATUS == "Civil marriage")
    FAMILY_STATUS_Separated = int(FAMILY_STATUS == "Separated")
    FAMILY_STATUS_Widow = int(FAMILY_STATUS == "Widow")

    # =========================
    # VIVIENDA
    # =========================
    HOUSING_TYPE = st.selectbox(
        "Tipo de vivienda:",
        [
            "With parents",
            "Rented apartment",
            "House / apartment",
            "Municipal apartment",
            "Office apartment",
            "Co-op apartment",
        ],
    )

    HOUSING_TYPE_With_parents = int(HOUSING_TYPE == "With parents")
    HOUSING_TYPE_Rented_apartment = int(HOUSING_TYPE == "Rented apartment")
    HOUSING_TYPE_House_or_apartment = int(HOUSING_TYPE == "House / apartment")
    HOUSING_TYPE_Municipal_apartment = int(HOUSING_TYPE == "Municipal apartment")
    HOUSING_TYPE_Office_apartment = int(HOUSING_TYPE == "Office apartment")
    HOUSING_TYPE_Co_op_apartment = int(HOUSING_TYPE == "Co-op apartment")

    # =========================
    # INGRESOS
    # =========================
    AMT_INCOME_TOTAL = st.number_input("Ingresos:", min_value=0.0, step=100.0)

    INCOME_TYPE = st.selectbox(
        "Fuente de ingresos:",
        [
            "Working",
            "State servant",
            "Commercial associate",
            "Businessman",
            "Maternity leave",
            "Student",
            "Unemployed",
            "Pensioner",
        ],
    )

    INCOME_TYPE_Alta_Estabilidad = int(INCOME_TYPE in ["Working", "State servant"])
    INCOME_TYPE_Media_Estabilidad = int(INCOME_TYPE in ["Commercial associate", "Businessman"])
    INCOME_TYPE_Baja_Estabilidad = int(INCOME_TYPE in ["Maternity leave", "Student", "Unemployed"])
    INCOME_TYPE_Pensionista = int(INCOME_TYPE == "Pensioner")

    # =========================
    # AÑOS TRABAJADOS
    # =========================
    YEARS_ACTUAL_WORK = st.text_input("Años en el trabajo actual:")

    if YEARS_ACTUAL_WORK == "":
        YEARS_ACTUAL_WORK = np.nan
    else:
        try:
            YEARS_ACTUAL_WORK = float(YEARS_ACTUAL_WORK)
        except ValueError:
            st.error("Debes introducir un número válido.")
            return

    # =========================
    # CHECKBOX
    # =========================
    FLAG_OWN_REALTY = st.checkbox("Casa propia")
    FLAG_OWN_CAR = st.checkbox("Carro propio")
    FLAG_PHONE = st.checkbox("Teléfono")
    FLAG_DNI = st.checkbox("DNI")
    FLAG_PASAPORTE = st.checkbox("Pasaporte")
    FLAG_COMPROBANTE_DOM_FISCAL = st.checkbox("Comprobante domicilio")
    FLAG_ESTADO_CUENTA_BANC = st.checkbox("Estado cuenta bancaria")
    FLAG_TARJETA_ID_FISCAL = st.checkbox("Tarjeta fiscal")
    FLAG_CERTIFICADO_LABORAL = st.checkbox("Certificado laboral")

    # =========================
    # CRÉDITO
    # =========================
    AMT_CREDIT = st.number_input("Crédito solicitado:", min_value=0.0, step=100.0)

    if st.button("Guardar datos"):
        st.success("Formulario completado correctamente.")


if __name__ == "__main__":
    main()
