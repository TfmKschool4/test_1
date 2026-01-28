import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Credit Risk Scoring (Plantilla)", page_icon="💳", layout="wide")

# -----------------------------
# Estado para navegación
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = None  # None | "single" | "bulk"

# -----------------------------
# Función dummy de scoring (sin modelo real)
# -----------------------------
def dummy_pd_score(data: dict) -> float:
    """
    Devuelve una probabilidad de impago (PD) simulada en [0,1].
    (Solo para plantilla: reemplazar por tu model_final.predict_proba)
    """
    income = float(data.get("AMT_INCOME_TOTAL", 0) or 0)
    credit = float(data.get("AMT_CREDIT", 0) or 0)
    years_work = float(data.get("YEARS_ACTUAL_WORK", 0) or 0)

    ratio = credit / (income + 1e-6)

    # Heurística simple: ratio alto => más PD, más años trabajando => menos PD
    pd_ = 0.15 + 0.10 * min(ratio, 10) - 0.01 * min(years_work, 30)
    pd_ = float(np.clip(pd_, 0.01, 0.95))
    return pd_

def pd_to_score(pd, base_score=600, pdo=50):
    odds = (1 - pd) / pd
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(20)
    return float(offset + factor * np.log(odds))

# -----------------------------
# Pantalla 0: elegir modo
# -----------------------------
def choose_mode():
    st.title("Entrada de solicitudes (Plantilla)")
    st.write("¿Vas a rellenar datos para **1 persona** o para **varias personas**?")

    option = st.radio("Modo:", ["1 persona", "Varias personas"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continuar", use_container_width=True):
            st.session_state.mode = "single" if option == "1 persona" else "bulk"
            st.rerun()
    with col2:
        st.info("Plantilla sin base de datos. Resultados simulados para mostrar el flujo.")

# -----------------------------
# Modo 1: formulario (basado en el tuyo)
# -----------------------------
def single_form():
    st.title("Formulario del préstamo (1 persona)")

    # ID del cliente
    SK_ID_CURR = st.text_input("ID del solicitante")
    NAME = st.text_input("Nombre del solicitante")

    # Edad
    AGES = st.slider("Edad:", min_value=18, max_value=100, value=18, step=1)
    labels = [1, 2, 3, 4]
    bins = [18, 34, 43, 54, 100]
    AGE_BINS = pd.cut([AGES], bins=bins, labels=labels, right=True, include_lowest=True).to_list()[0]

    # Género
    GENDER = st.selectbox("Género del solicitante:", ["Masculino", "Femenino"])
    GENDER_M = 1 if GENDER == "Masculino" else 0
    GENDER_F = 1 if GENDER == "Femenino" else 0
    CODE_GENDER = "M" if GENDER == "Masculino" else "F"

    # Hijos
    CNT_CHILDREN = st.selectbox("Número de hijos:", ["0", "1", "2", "3", "4 o más"])
    CNT_CHILDREN = {"0": 0, "1": 1, "2": 2, "3": 3, "4 o más": 4}[CNT_CHILDREN]

    # Estudios
    NAME_EDUCATION_TYPE = st.selectbox(
        "Nivel de estudios:",
        ["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"],
    )
    LEVEL_EDUCATION_TYPE = {
        "Lower secondary": 0,
        "Secondary / secondary special": 1,
        "Incomplete higher": 2,
        "Higher education": 3,
        "Academic degree": 4,
    }[NAME_EDUCATION_TYPE]

    # Familia
    FAMILY_STATUS = st.selectbox(
        "Situación familiar del solicitante:",
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    )

    # Vivienda
    HOUSING_TYPE = st.selectbox(
        "Tipo de vivienda del solicitante:",
        ["With parents", "Rented apartment", "House / apartment", "Municipal apartment", "Office apartment", "Co-op apartment"],
    )

    # Ingresos
    AMT_INCOME_TOTAL = st.number_input("Ingresos del solicitante:", min_value=0.00, value=0.00, step=100.00)

    # Fuente de ingresos
    INCOME_TYPE = st.selectbox(
        "Indica la fuente de ingresos del solicitante:",
        ["Working", "State servant", "Commercial associate", "Businessman", "Maternity leave", "Student", "Unemployed", "Pensioner"],
    )

    # Años en su trabajo actual
    YEARS_ACTUAL_WORK = st.text_input("Años en su actual puesto de trabajo (vacío si no aplica):")

    # Flags
    FLAG_OWN_REALTY = st.checkbox("¿El cliente posee casa propia?")
    FLAG_PHONE = st.checkbox("¿Ha proporcionado su número de teléfono?")
    FLAG_DNI = st.checkbox("¿Ha entregado el DNI?")
    FLAG_PASAPORTE = st.checkbox("¿Ha entregado el pasaporte?")
    FLAG_COMPROBANTE_DOM_FISCAL = st.checkbox("¿Ha entregado su comprobante de domicilio fiscal?")
    FLAG_ESTADO_CUENTA_BANC = st.checkbox("¿Ha entregado el estado de su cuenta bancaria?")
    FLAG_TARJETA_ID_FISCAL = st.checkbox("¿Ha entregado su tarjeta de identificación fiscal?")
    FLAG_CERTIFICADO_LABORAL = st.checkbox("¿Ha entregado su certificado laboral?")

    # Crédito solicitado
    AMT_CREDIT = st.number_input("Crédito solicitado:", min_value=0.0, value=0.0, step=100.0)

    st.divider()

    if st.button("Procesar (simulado)", use_container_width=True):
        # Validaciones básicas solo al pulsar
        errors = []
        try:
            sk_id_int = int(SK_ID_CURR)
            if sk_id_int < 0:
                errors.append("El ID debe ser entero positivo o 0.")
        except Exception:
            errors.append("El ID debe ser un entero válido y no puede estar vacío.")

        if NAME == "" or (not NAME.isalpha()):
            errors.append("El nombre debe ser válido (solo letras) y no puede estar vacío.")

        years_work_value = 0.0
        if YEARS_ACTUAL_WORK != "":
            try:
                years_work_value = float(YEARS_ACTUAL_WORK)
                if years_work_value < 0:
                    errors.append("Años de trabajo debe ser positivo.")
            except Exception:
                errors.append("Años de trabajo debe ser numérico (o vacío).")

        if errors:
            st.error("Corrige los errores:")
            for e in errors:
                st.write("•", e)
            st.stop()

        # Construimos “fila” como en tu estructura
        datos_solicitante = {
            "SK_ID_CURR": int(SK_ID_CURR),
            "NAME": str(NAME),
            "AGE_BINS": int(AGE_BINS),
            "AGE": int(AGES),
            "CODE_GENDER": CODE_GENDER,
            "GENDER_M": int(GENDER_M),
            "GENDER_F": int(GENDER_F),
            "CNT_CHILDREN": int(CNT_CHILDREN),
            "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
            "LEVEL_EDUCATION_TYPE": int(LEVEL_EDUCATION_TYPE),
            "NAME_FAMILY_STATUS": FAMILY_STATUS,
            "NAME_HOUSING_TYPE": HOUSING_TYPE,
            "NAME_INCOME_TYPE": INCOME_TYPE,
            "AMT_INCOME_TOTAL": float(AMT_INCOME_TOTAL),
            "AMT_CREDIT": float(AMT_CREDIT),
            "YEARS_ACTUAL_WORK": float(years_work_value) if YEARS_ACTUAL_WORK != "" else None,
            "FLAG_OWN_REALTY": int(FLAG_OWN_REALTY),
            "FLAG_PHONE": int(FLAG_PHONE),
            "FLAG_DNI": int(FLAG_DNI),
            "FLAG_PASAPORTE": int(FLAG_PASAPORTE),
            "FLAG_COMPROBANTE_DOM_FISCAL": int(FLAG_COMPROBANTE_DOM_FISCAL),
            "FLAG_ESTADO_CUENTA_BANC": int(FLAG_ESTADO_CUENTA_BANC),
            "FLAG_TARJETA_ID_FISCAL": int(FLAG_TARJETA_ID_FISCAL),
            "FLAG_CERTIFICADO_LABORAL": int(FLAG_CERTIFICADO_LABORAL),
        }

        df = pd.DataFrame([datos_solicitante])
        st.success("Datos capturados correctamente.")
        st.dataframe(df, use_container_width=True)

        # Resultados simulados
        pd_score = dummy_pd_score({
            "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
            "AMT_CREDIT": AMT_CREDIT,
            "YEARS_ACTUAL_WORK": years_work_value,
        })
        score = pd_to_score(pd_score)
        threshold = 0.5
        decision = int(pd_score >= threshold)

        st.subheader("Resultado (simulado)")
        c1, c2, c3 = st.columns(3)
        c1.metric("PD (prob. impago)", f"{pd_score:.2%}")
        c2.metric("Score", f"{score:.0f}")
        c3.metric("Decisión", "❌ Riesgo alto" if decision else "✅ Riesgo bajo")

        if pd_score < 0.2:
            st.success("Riesgo bajo: Aprobación recomendada ✅")
        elif pd_score < 0.4:
            st.warning("Riesgo medio: Revisión manual 🟡")
        else:
            st.error("Riesgo alto: Rechazo recomendado ❌")

    st.divider()
    if st.button("⬅️ Volver", use_container_width=True):
        st.session_state.mode = None
        st.rerun()

# -----------------------------
# Modo 2: varias personas (tabla)
# -----------------------------
def bulk_table():
    st.title("Carga múltiple (varias personas)")
    st.write("Rellena varias solicitudes en la tabla. Resultados simulados.")

    cols = [
        "SK_ID_CURR", "NAME", "AGE", "CODE_GENDER",
        "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "YEARS_ACTUAL_WORK",
        "FLAG_OWN_REALTY", "FLAG_PHONE", "FLAG_DNI", "FLAG_PASAPORTE",
        "FLAG_COMPROBANTE_DOM_FISCAL", "FLAG_ESTADO_CUENTA_BANC",
        "FLAG_TARJETA_ID_FISCAL", "FLAG_CERTIFICADO_LABORAL",
    ]

    n = st.number_input("Número de solicitantes", min_value=2, max_value=200, value=5, step=1)
    df = pd.DataFrame([{c: None for c in cols} for _ in range(int(n))])
    df.index = range(1, int(n) + 1)  # 👈 ahora se verá 1..n
    edited = st.data_editor(df, use_container_width=True, num_rows="fixed")


    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Validar", use_container_width=True):
            errors = []
            if edited["SK_ID_CURR"].isna().any():
                errors.append("Hay SK_ID_CURR vacíos.")
            if edited["NAME"].isna().any():
                errors.append("Hay NAME vacíos.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.success("Validación básica OK.")

    with col2:
        if st.button("⚙️ Procesar (simulado)", use_container_width=True):
            # Calcula PD/Score fila a fila de forma simulada
            out = edited.copy()
            out["PD"] = out.apply(
                lambda r: dummy_pd_score({
                    "AMT_INCOME_TOTAL": r.get("AMT_INCOME_TOTAL", 0),
                    "AMT_CREDIT": r.get("AMT_CREDIT", 0),
                    "YEARS_ACTUAL_WORK": r.get("YEARS_ACTUAL_WORK", 0),
                }),
                axis=1
            )
            out["SCORE"] = out["PD"].apply(pd_to_score)
            out["DECISION"] = out["PD"].apply(lambda p: "❌ Riesgo alto" if p >= 0.5 else "✅ Riesgo bajo")

            st.subheader("Resultados (simulados)")
            st.dataframe(out, use_container_width=True)

            st.download_button(
                "Descargar resultados CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="resultados_scoring_simulados.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col3:
        if st.button("⬅️ Volver", use_container_width=True):
            st.session_state.mode = None
            st.rerun()

# -----------------------------
# Router
# -----------------------------
if st.session_state.mode is None:
    choose_mode()
elif st.session_state.mode == "single":
    single_form()
else:
    bulk_table()
