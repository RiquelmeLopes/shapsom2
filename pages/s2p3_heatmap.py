import streamlit as st
from page_classes import S2P3_Heatmap
from my_utilities import report_page_top, report_page_bottom, display_heatmaps, PAGE_COUNT
from copy import deepcopy

report_page_top("heatmap", S2P3_Heatmap, "2. Análise por Agrupamentos", 7/PAGE_COUNT)
st.subheader('Seção 2 - Visão dos Dados e Gráficos de Mapas de Calor')
st.markdown('''Esta seção traz uma análise visual da base de dados, fornecendo mapas de calor para a média
            (*Gráfico 1*) e desvio padrão (*Gráfico 2*) dos fatores disponibilizados para cada um dos municípios.
            Mapa de Calor, também conhecido como Heatmap, é uma visualização gráfica que usa cores para representar a intensidade dos valores
            em uma matriz de dados. Cada célula da matriz é colorida de acordo com seu valor, facilitando a identificação de
            padrões, tendências e anomalias nos dados.
            **Média**: É a soma de todos os valores de um conjunto dividida pelo número de valores.
            Representa o valor médio
            **Desvio padrão**: Mede a dispersão dos valores em relação à média. Mostra o quanto os valores variam da média.''')
st.markdown('''Importante:
            Nos gráficos referentes aos Mapas de Calor:
            As linhas representam os municípios, que estão em ordem alfabética;
            As colunas representam os fatores selecionados pelo usuário na base de dados''')

avg, std = display_heatmaps(0.0, 1.0)
st.session_state["heatmap"].avg_df = deepcopy(avg)
st.session_state["heatmap"].std_df = deepcopy(std)
report_page_bottom("heatmap", "pages/s2p2_mapa_som.py", "pages/s2p4_shap.py")