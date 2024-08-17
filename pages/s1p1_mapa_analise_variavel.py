import streamlit as st
import streamlit.components.v1 as components
from copy import deepcopy
from page_classes import S1P1_MapaExploratorio
from my_utilities import report_page_top, report_page_bottom, make_map, PAGE_COUNT

report_page_top("mapa exploratorio", S1P1_MapaExploratorio, "1. Análise Estatística Exploratória", 1/PAGE_COUNT)
st.subheader('Seção 1 - Mapa de Análise da Variável Alvo')
st.markdown('''O mapa de análise da variável alvo apresenta uma análise geoespacial dos municípios do estado de
             Pernambuco. As diferentes tonalidades de cores no mapa representam as variações nos níveis da variável
             de escolha. As áreas em tons mais escuros indicam um desempenho superior, enquanto as áreas em tons
             mais claros refletem um desempenho inferior. Esta visualização detalhada é crucial para identificar
             regiões que necessitam de intervenções mais intensivas, ajudando a direcionar políticas públicas e
             recursos de forma mais eficiente.''')

with st.spinner('Gerando mapa...'):
    df = deepcopy(st.session_state["base reader"].crunched_database_average)
    name_col = st.session_state["base reader"].name_columns[0]
    output_col = st.session_state["base reader"].output_columns[0]
    if st.session_state["mapa exploratorio"].map is None:
        map, img = make_map(df, name_col, output_col)
        st.session_state["mapa exploratorio"].map = deepcopy(map)
        st.session_state["mapa exploratorio"].img_path = img
    components.html(st.session_state["mapa exploratorio"].map._repr_html_(), height=600)
    st.info(f'Figura 1 - Mapa de Análise da Variável Alvo')

report_page_bottom("mapa exploratorio", "pages/s0p1_repetir_planilha.py", "pages/s1p2_analise_estatistica.py")