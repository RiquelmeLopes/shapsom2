import streamlit as st
from copy import deepcopy
from page_classes import S2P7_HeatmapFiltro
from my_utilities import report_page_top, report_page_bottom, display_heatmaps, PAGE_COUNT

def S2P7_set_default_parameters():
    st.session_state["heatmap filtro"].filtro_min = st.session_state["filtro min"] = 0
    st.session_state["heatmap filtro"].filtro_max = st.session_state["filtro max"] = 70

def S2P7_load_parameters():
    st.session_state["filtro min"] = st.session_state["heatmap filtro"].filtro_min
    st.session_state["filtro max"] = st.session_state["heatmap filtro"].filtro_max

def S2P7_update_parameters(filtro_min, filtro_max) -> bool:
    changed = False

    if st.session_state["heatmap filtro"].filtro_min != filtro_min:
        st.session_state["heatmap filtro"].filtro_min = filtro_min
        changed = True

    if st.session_state["heatmap filtro"].filtro_max != filtro_max:
        st.session_state["heatmap filtro"].filtro_max = filtro_max
        changed = True

    return changed

report_page_top("heatmap filtro", S2P7_HeatmapFiltro, "2. Análise por Agrupamentos", 10/PAGE_COUNT, set_default_parameters=S2P7_set_default_parameters, load_parameters=S2P7_load_parameters)
st.subheader('Seção 5 - Filtro de Triagem')
st.markdown('''Esta seção, assim como na seção 2, traz uma análise visual da base de dados, porém agora em uma fatia dos dados
            escolida pelo usuário.
            Essa visualização é útil para analizar de forma mais detalhada elementos de interesse da base de dados.''')
st.markdown('''Como essa seção funciona:
            Ela usa os valores fornecidos pelo usuário nos campos abaixo para filtrar
            a última coluna da base (saída), exibindo as tabelas e mapas de calor para
            o conjuto de dados cujo o valor da coluna de saída esteja dentro do intervalo
            de valores fornecido pelo usuário.''')

with st.form("filter_form"):
    filtro_min, filtro_max = st.slider('Defina o intervalo (porcentagem)', 0, 100, (st.session_state["filtro min"], st.session_state["filtro max"]), step=1)
    if st.form_submit_button("**Aplicar intervalo**"):
        S2P7_update_parameters(filtro_min, filtro_max)

avg, std = display_heatmaps(st.session_state["heatmap filtro"].filtro_min / 100, st.session_state["heatmap filtro"].filtro_max / 100, numero_secao=5)
st.session_state["heatmap filtro"].avg_df = deepcopy(avg)
st.session_state["heatmap filtro"].std_df = deepcopy(std)
report_page_bottom("heatmap filtro", "pages/s2p6_analise_grupos.py", "pages/s2p8_anomalias.py")