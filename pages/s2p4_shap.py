import streamlit as st
import pandas as pd
from copy import deepcopy
from page_classes import S2P4_Shap
from my_utilities import report_page_top, report_page_bottom, calculate_shap, PAGE_COUNT

report_page_top("shap", S2P4_Shap, "2. Análise por Agrupamentos", 8/PAGE_COUNT)

st.subheader('Seção 3.1 - Análise de agrupamentos com SHAP')
st.markdown('''Nesta seção, apresentamos os grupos identificados e as variáveis que mais influenciaram na formação desses grupos. Um "agrupamento" reúne dados que são mais semelhantes em termos de suas características globais. Esses grupos são utilizados na aplicação de IA através de bases de dados (tabelas) fornecidas pela área usuária para o processamento com Redes Neurais Artificiais. "Agrupamento" é o processo de reunir, por exemplo, municípios, com base em suas semelhanças, visando realizar triagens para guiar auditorias.''')

with st.spinner("Calculando SHAP..."):
    if st.session_state["shap"].df is None:
        df: pd.DataFrame = deepcopy(st.session_state["base reader"].crunched_database_average)
        som_df: pd.DataFrame = deepcopy(st.session_state["mapa som"].df)
        shap_df = pd.DataFrame()
        shap_df["Nome do Fator"] = st.session_state["base reader"].input_columns

        for i, (_, cluster_df) in enumerate(som_df.groupby("Grupo")):
            sub_df = df[df[st.session_state["base reader"].name_columns[0]].isin(cluster_df["Municípios"].values)]
            shap_vals = calculate_shap(sub_df, st.session_state["base reader"].input_columns, st.session_state["base reader"].output_columns)
            shap_df[f"Grupo {i+1}"] = list(shap_vals)

        st.session_state["shap"].df = deepcopy(shap_df)
    
    shap_df = deepcopy(st.session_state["shap"].df)
    shap_df = shap_df.style.applymap(lambda val : f'color: {"blue" if val > 0 else ("red" if val < 0 else "")}', subset=shap_df.columns[1:])
    st.dataframe(shap_df, use_container_width=True)
    st.info('Tabela 3.1.1 - Influência dos fatores em cada grupo')
report_page_bottom("shap", "pages/s2p3_heatmap.py", "pages/s2p5_arvore.py")