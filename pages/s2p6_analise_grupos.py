import streamlit as st
import streamlit.components.v1 as components
from copy import deepcopy
from page_classes import S2P6_AnaliseGrupos
from my_utilities import report_page_top, report_page_bottom, make_map, PAGE_COUNT
import pandas as pd
import numpy as np

report_page_top("analise grupos", S2P6_AnaliseGrupos, "2. Análise por Agrupamentos", 9/PAGE_COUNT)
st.subheader('Seção 4 - Diferenças entre Agrupamentos')
st.markdown('''A análise comparativa entre os agrupamentos é conduzida combinando todas as informações
            da "Análise de Agrupamento" (Seção 3), organizando-as em uma disposição paralela. Isso tem o
            objetivo de destacar de forma mais clara as disparidades nas estruturas dos agrupamentos.''')

df: pd.DataFrame = deepcopy(st.session_state["base reader"].crunched_database_average)
som_df: pd.DataFrame = deepcopy(st.session_state["mapa som"].df)
shap_df = deepcopy(st.session_state["shap"].df)
name_column = st.session_state["base reader"].name_columns[0]
output_column = st.session_state["base reader"].output_columns[0]

group_amnt = len(set(list(som_df["Grupo"].values)))
correclty_cached = all([
    len(st.session_state["analise grupos"].output_averages) == group_amnt,
    len(st.session_state["analise grupos"].municipio_dfs) == group_amnt,
    len(st.session_state["analise grupos"].shap_dfs) == group_amnt,
    len(st.session_state["analise grupos"].maps) == group_amnt,
    len(st.session_state["analise grupos"].image_paths) == group_amnt
])

if not correclty_cached:
    st.session_state["analise grupos"].output_averages = []
    st.session_state["analise grupos"].municipio_dfs = []
    st.session_state["analise grupos"].shap_dfs = []
    st.session_state["analise grupos"].maps = []
    st.session_state["analise grupos"].image_paths = []

    for i, (_, cluster_df) in enumerate(som_df.groupby("Grupo")):
        with st.spinner(f'Carregando dados do Grupo {i+1}...'):
            st.subheader(f'Grupo {i+1}')
            sub_df = df[df[name_column].isin(cluster_df["Municípios"].values)]
            avg_val = round(float(np.average(sub_df[output_column].values)), 2)
            st.session_state["analise grupos"].output_averages.append(avg_val)
            st.text(f'Média de {output_column} do grupo {i+1}: {avg_val}')

            st.text('Municípios do grupo:')
            df_sub_municipios = sub_df[[name_column, output_column]]
            st.session_state["analise grupos"].municipio_dfs.append(deepcopy(df_sub_municipios))
            st.dataframe(df_sub_municipios, use_container_width=True)
            st.info(f"**Tabela 4.{i+1}.1 - Municípios do Grupo {i+1}**")

            _variables = list(shap_df["Nome do Fator"].values)
            _values = list(shap_df[f"Grupo {i+1}"].values)
            _argmax = int(np.argmax(_values))
            _argmin = int(np.argmin(_values))

            st.text('Influências dos Fatores:')
            influence_df = pd.DataFrame()
            if _values[_argmax] > 0 and _values[_argmin] < 0:
                influence_df["Nome do Fator"] = [_variables[_argmax], _variables[_argmin]]
                influence_df[f"Grupo {i+1}"] = [_values[_argmax], _values[_argmin]]
            elif _values[_argmax] > 0:
                influence_df["Nome do Fator"] = [_variables[_argmax]]
                influence_df[f"Grupo {i+1}"] = [_values[_argmax]]
            elif _values[_argmin] < 0:
                influence_df["Nome do Fator"] = [_variables[_argmin]]
                influence_df[f"Grupo {i+1}"] = [_values[_argmin]]
            else:
                influence_df["Nome do Fator"] = []
                influence_df[f"Grupo {i+1}"] = []

            st.session_state["analise grupos"].shap_dfs.append(deepcopy(influence_df))
            influence_df = influence_df.style.applymap(lambda val : f'color: {"blue" if val > 0 else ("red" if val < 0 else "")}', subset=influence_df.columns[1:])
            st.dataframe(influence_df, use_container_width=True, hide_index=True)
            st.info(f"**Tabela 4.{i+1}.2 - Fatores Que Mais Influenciam Positivamente e Negativamente no Grupo {i+1}**")

            m, img_path = make_map(sub_df, name_column, output_column, color=str(cluster_df["Cor"].values[0]))
            st.session_state["analise grupos"].maps.append(deepcopy(m))
            st.session_state["analise grupos"].image_paths.append(img_path)
            components.html(m._repr_html_(), height=600)
            st.info(f"**Figura 4.{i+1} - Mapa de Municípios do Grupo {i+1}**")
else:
    iterables = deepcopy(list(zip(
        st.session_state["analise grupos"].output_averages,
        st.session_state["analise grupos"].municipio_dfs,
        st.session_state["analise grupos"].shap_dfs,
        st.session_state["analise grupos"].maps
    )))
    for i,(avg_val, df_sub_municipios, influence_df, m) in enumerate(iterables):
        with st.spinner(f'Carregando dados do Grupo {i+1}...'):
            st.subheader(f'Grupo {i+1}')
            st.text(f'Média de {output_column} do grupo {i+1}: {avg_val}')

            st.text('Municípios do grupo:')
            st.dataframe(df_sub_municipios, use_container_width=True)
            st.info(f"**Tabela 4.{i+1}.1 - Municípios do Grupo {i+1}**")

            influence_df = influence_df.style.applymap(lambda val : f'color: {"blue" if val > 0 else ("red" if val < 0 else "")}', subset=influence_df.columns[1:])
            st.dataframe(influence_df, use_container_width=True, hide_index=True)
            st.info(f"**Tabela 4.{i+1}.2 - Fatores Que Mais Influenciam Positivamente e Negativamente no Grupo {i+1}**")

            components.html(m._repr_html_(), height=600)
            st.info(f"**Figura 4.{i+1} - Mapa de Municípios do Grupo {i+1}**")

report_page_bottom("analise grupos", "pages/s2p5_arvore.py", "pages/s2p7_heatmap_filtro.py")