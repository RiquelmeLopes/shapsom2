import streamlit as st
import pandas as pd
from copy import deepcopy
from page_classes import S2P9_TabelaRegioes
from my_utilities import report_page_top, report_page_bottom, REGIONS_TABLE, PAGE_COUNT

report_page_top("tabela regioes", S2P9_TabelaRegioes, "2. Análise por Agrupamentos", 12/PAGE_COUNT)

st.subheader('Seção 7 - Identificação de Mesorregiões e Microrregiões')
st.markdown('''Essa seção traz uma tabela com todos os municípios de Pernambuco, identificando suas mesorregiões
             e microrregiões e dando um índice para elas, que é o índice utilizado nos Mapas de Calor.''')

with st.spinner("Carregando tabela..."):
    if st.session_state["tabela regioes"].df is None:
        df = deepcopy(REGIONS_TABLE)
        df.index = pd.RangeIndex(start=1, stop=len(df)+1, step=1)
        st.session_state["tabela regioes"].df = deepcopy(df)
    st.dataframe(st.session_state["tabela regioes"].df, hide_index=False, use_container_width=True)
    st.info('Tabela 7.1 - Lista de Mesorregiões e Microrregiões')

if st.session_state["base reader"].output_columns:
    report_page_bottom("tabela regioes", "pages/s2p8_anomalias.py", "pages/s2report.py")
else:
    report_page_bottom("tabela regioes", "pages/s2p2_mapa_som.py", "")