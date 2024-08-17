import streamlit as st
import pandas as pd
from copy import deepcopy
from page_classes import S2P8_Anomalias, S3P1_RelatorioIndividual
from my_utilities import report_page_top, report_page_bottom, calculate_shap, PAGE_COUNT
import numpy as np

def S2P8_set_default_parameters():
    st.session_state["anomalias"].porcentagem = st.session_state["porcentagem"] = 10
    st.session_state["relatorio individual"] = S3P1_RelatorioIndividual()

def S2P8_load_parameters():
    st.session_state["porcentagem"] = st.session_state["anomalias"].porcentagem

def S2P8_update_parameters(porcentagem) -> bool:
    if st.session_state["anomalias"].porcentagem != porcentagem:
        st.session_state["anomalias"].porcentagem = porcentagem
        return True
    else:
        return False

report_page_top("anomalias", S2P8_Anomalias, "2. Análise por Agrupamentos", 11/PAGE_COUNT, set_default_parameters=S2P8_set_default_parameters, load_parameters=S2P8_load_parameters)
st.subheader('Seção 6 - Anomalias')
st.write("""A análise de anomalias foi conduzida utilizando um Mapa Auto-Organizável (SOM) para identificar pontos de
          dados que se desviam significativamente do padrão observado. Com as coordenadas dos pontos no SOM, o centroide
          do mapa foi calculado. Este centroide é determinado utilizando a mediana das coordenadas x e y de todos os
          pontos, o que fornece uma medida menos sensível a outliers em comparação com a média. Então, são calculadas
          as distâncias dos pontos para o centroide do mapa. Pontos que apresentaram distâncias significativamente
          maiores em relação ao centroide foram identificados como anômalos. Estes pontos fora do cluster principal
          sugerem comportamentos ou características discrepantes dos dados normais, destacando-se por estarem afastados
          do padrão usual.""")

st.altair_chart(st.session_state["mapa som"].map, use_container_width=True)
st.info('Figura 6.1 - Mapa SOM')

with st.form("anomalia_form"):
    porcentagem = st.slider("Defina a porcentagem das anomalias", min_value=0, max_value=100, value=st.session_state["porcentagem"])
    if st.form_submit_button("**Identificar anomalias**"):
        S2P8_update_parameters(porcentagem)

with st.spinner("Identificando anomalias..."):
    if st.session_state["anomalias"].df is None:
        crunched_df: pd.DataFrame = deepcopy(st.session_state["base reader"].crunched_database_average)
        som_df: pd.DataFrame = deepcopy(st.session_state["mapa som"].df)
        
        name_cols = st.session_state["base reader"].name_columns
        input_cols = st.session_state["base reader"].input_columns #Fatores
        output_cols = st.session_state["base reader"].output_columns

        city_names = crunched_df[name_cols[0]].values.tolist()
        output_vals = crunched_df[output_cols[0]].values.tolist()
        individual_shaps = calculate_shap(crunched_df, input_cols, output_cols, return_average=False).tolist()

        som_order = som_df["Municípios"].values.tolist()
        adjusted_order = [som_order.index(c) for c in city_names]
        city_names = [x for _, x in sorted(zip(adjusted_order, city_names))]
        output_vals = [x for _, x in sorted(zip(adjusted_order, output_vals))]
        individual_shaps = [x for _, x in sorted(zip(adjusted_order, individual_shaps))] #Influência
        individual_inputs = [x for _, x in sorted(zip(adjusted_order, crunched_df[input_cols].values.tolist()))] #Valor

        centroid_x = float(np.median(som_df["x"].values))
        centroid_y = float(np.median(som_df["y"].values))
        center_distance = [float(np.sqrt((x-centroid_x)**2 + (y-centroid_y)**2)) for x,y in zip(som_df["x"].values, som_df["y"].values)]
        som_df["Distância do Centroide"] = center_distance
        som_df[output_cols[0]] = output_vals
        som_df["Fator mais influente"] = [f"{input_cols[int(np.argmax(values))]} ({round(max(values), 3)})" for values in individual_shaps]
        som_df["Fator menos influente"] = [f"{input_cols[int(np.argmin(values))]} ({round(min(values), 3)})" for values in individual_shaps]

        for municipio, valores, influencias in zip(som_order, individual_inputs, individual_shaps):
            df = pd.DataFrame()
            df["Fator"] = input_cols
            df["Valor"] = valores
            df["Influência"] = influencias

            _self_df = som_df[som_df['Municípios'] == municipio]
            _grupo = _self_df["Grupo"].values[0]
            _avg_grupo = float(np.average(som_df[som_df['Grupo'] == _grupo]["Nota"].values))
            _nota_municipio = _self_df["Nota"].values[0]
            _my_x = _self_df["x"].values[0]
            _my_y = _self_df["y"].values[0]
            _num_closest_ones = 3
            _distances = [float(np.sqrt((x-_my_x)**2 + (y-_my_y)**2)) for x,y in zip(som_df["x"].values, som_df["y"].values)]
            _closest_ones = np.argsort(_distances).flatten().tolist()[:(_num_closest_ones+1)]
            _closest_ones = [som_order[i] for i in _closest_ones if som_order[i] != municipio][:_num_closest_ones]
            municipio_dict = {
                "dados": df,
                "grupo": _grupo,
                "nota_media_grupo": _avg_grupo,
                "nota_individual": _nota_municipio,
                "output": output_cols[0],
                "vizinhos": _closest_ones
            }
            st.session_state["relatorio individual"].municipios[municipio] = deepcopy(municipio_dict)

        df = som_df.sort_values(by='Distância do Centroide', ascending=False).drop(['Cor', "Nota"], axis=1)
        st.session_state["anomalias"].df = deepcopy(df)

    df = deepcopy(st.session_state["anomalias"].df).drop(['x', "y"], axis=1)
    df = df[:int(len(df) * st.session_state["anomalias"].porcentagem / 100)]
    st.dataframe(df, use_container_width=True)
    st.info('Tabela 6.1 - Resultados de Anomalias')

report_page_bottom("anomalias", "pages/s2p7_heatmap_filtro.py", "pages/s2p9_tabela_regioes.py")