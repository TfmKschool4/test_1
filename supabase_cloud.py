def page_home():
    # --- FUNCIÓN AUXILIAR PARA EL LOGO ---
    def get_img_as_base64(file):
        try:
            with open(file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except FileNotFoundError:
            return None

    # Cargar logo (asegúrate de que 'logo.png' esté en la misma carpeta)
    img_b64 = get_img_as_base64("logo.png")
    
    # Si la imagen existe, crear la etiqueta HTML, si no, dejar vacío
    if img_b64:
        img_html = f'<img src="data:image/png;base64,{img_b64}" class="logo-img">'
    else:
        img_html = "" # Fallback por si no encuentra la imagen

    # --- 1. CSS ESTILO "GLASSMORPHISM" Y MEJORA DE CONTRASTE ---
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
        background-color: rgba(0, 0, 0, 0.1);
        z-index: -1;
    }

    /* CONTENEDOR DEL TÍTULO (Caja blanca) */
    .header-box {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
    }

    /* ESTILO DEL LOGO */
    .logo-img {
        max-width: 150px; /* Ajusta el tamaño del logo aquí */
        margin-bottom: 15px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* TÍTULO PRINCIPAL */
    .custom-title {
        color: #000000 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-top: 0px;
        margin-bottom: 10px;
    }

    .custom-subtitle {
        color: #141414 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* --- MEJORA DE CONTRASTE PARA TARJETAS INFERIORES --- */
    
    /* Fondo blanco más opaco para las tarjetas de acción */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.95) !important; /* Más blanco */
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* Forzar color NEGRO y aumentar peso en los títulos (h3) "Sobre nosotros" y "Evaluación" */
    h3 {
        color: #000000 !important;
        font-weight: 900 !important; /* Extra negrita */
        text-shadow: 0px 0px 10px rgba(255,255,255, 1); /* Halo blanco para legibilidad */
    }

    /* Forzar color negro en los textos descriptivos dentro de las tarjetas */
    div[data-testid="stMarkdownContainer"] p {
        color: #1a1a1a !important; /* Negro casi puro */
        font-weight: 600 !important; /* Semi-negrita */
    }
    
    /* Estilo específico para el mensaje de éxito verde para que se lea bien */
    .stAlert {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- 2. CONTENIDO PRINCIPAL ---
    
    # Inyectamos el HTML que incluye el LOGO (si existe) y el Título
    st.markdown(f"""
    <div class="header-box">
        {img_html}
        <h1 class="custom-title">Creditum</h1>
        <p class="custom-subtitle">Análisis inteligente del riesgo crediticio.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 3. BOTONES DE ACCIÓN ---
    col_spacer_left, col_action1, col_action2, col_spacer_right = st.columns([0.5, 2, 2, 0.5])

    with col_action1:
        with st.container(border=True):
            st.markdown("### 🏢 Sobre nosotros")
            # Un pequeño espacio invisible para separar título de botón
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
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
