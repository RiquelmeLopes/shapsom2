import streamlit as st
from copy import deepcopy
import plotly.express as px
from page_classes import S1P3_GraficoDispersao
from my_utilities import report_page_top, report_page_bottom, generate_random_string, PAGE_COUNT
from threading import Lock
import os

globals()["lock"] = Lock()

def S1P3_set_default_parameters():
    st.session_state["variavel_dispersao"] = list(st.session_state["base reader"].input_columns + st.session_state["base reader"].output_columns)[-1]

def S1P3_load_parameters():
    st.session_state["variavel_dispersao"] = st.session_state["grafico dispersao"].variavel_dispersao

def S1P3_update_parameters(variavel_dispersao):
    st.session_state["grafico dispersao"].variavel_dispersao = variavel_dispersao

report_page_top("grafico dispersao", S1P3_GraficoDispersao, "1. Análise Estatística Exploratória", 3/PAGE_COUNT, set_default_parameters=S1P3_set_default_parameters, load_parameters=S1P3_load_parameters)
st.subheader('Seção 3 - Gráfico de Dispersão')
st.markdown('''O gráfico de dispersão faz parte de uma análise estatística mais ampla apresentada no relatório, que visa 
                explorar a variabilidade e o desempenho geral dos municípios. Ele permite identificar quais municípios
                apresentam desempenhos extremos, tanto positivos quanto negativos, e como os valores da nossa variável alvo estão dispersos
                em relação à media. Esta visualização facilita uma identificação mais superficial das áreas que necessitam de maior atenção e recursos.''')

opcoes = list(st.session_state["base reader"].input_columns + st.session_state["base reader"].output_columns)
nc = st.session_state["base reader"].name_columns[0]
oc = st.selectbox('Selecione a variável', opcoes, index=opcoes.index(st.session_state["variavel_dispersao"]))

with st.spinner('Gerando gráfico de dispersão...'):
    if st.session_state["grafico dispersao"].img_path:
        os.remove(st.session_state["grafico dispersao"].img_path)
    df = deepcopy(st.session_state["base reader"].crunched_database_average)
    dfmc = df.groupby(nc)[oc].apply(lambda x: x.mode().iloc[0]).reset_index()
    dfmc[dfmc.columns[-1]] = dfmc[dfmc.columns[-1]].round(2)
    dfmc_copy = dfmc[dfmc.columns[-1]].describe().to_frame().T
    fig = px.scatter(dfmc, y=oc, x=nc, color=oc, color_continuous_scale='icefire_r')
    fig.update_layout(coloraxis_colorbar=dict(title=None))
    rand_imgname = os.path.join("tempfiles", f"{generate_random_string(10)}.png")
    with globals()["lock"]:
        fig.write_image(rand_imgname, engine="kaleido")
    st.session_state["grafico dispersao"].map = deepcopy(fig)
    st.session_state["grafico dispersao"].img_path = rand_imgname
    st.plotly_chart(st.session_state["grafico dispersao"].map, use_container_width=True)
    st.info(f'Gráfico 1 - Gráfico de Dispersão da Distribuição da Variável Selecionada por Município')

report_page_bottom("grafico dispersao", "pages/s1p2_analise_estatistica.py", "pages/s1report.py", update_parameters=lambda : S1P3_update_parameters(oc))