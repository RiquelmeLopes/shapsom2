import streamlit as st
from copy import deepcopy
from page_classes import S2P2_MapaSom
import altair as alt
import math
from my_utilities import report_page_top, report_page_bottom, create_map, HEX_SHAPE, clear_cached_data, PAGE_COUNT

def S2P2_set_default_parameters():
    st.session_state["mapa som"].sigma = st.session_state["sigma"] = 4 + int(math.ceil(5 * len(st.session_state["base reader"].crunched_database_average) / 186))
    st.session_state["mapa som"].size = st.session_state["size"] = 10 + int(math.ceil(20 * len(st.session_state["base reader"].crunched_database_average) / 186))
    st.session_state["mapa som"].lr = st.session_state["lr"] = -3.0
    st.session_state["mapa som"].epochs = st.session_state["epochs"] = 15000
    st.session_state["mapa som"].cluster_distance = st.session_state["cluster_distance"] = 1.5
    st.session_state["mapa som"].topology = st.session_state["topology"] = "Hexagonal"
    st.session_state["mapa som"].output_influences = st.session_state["output_influences"] = True

def S2P2_load_parameters():
    st.session_state["sigma"] = st.session_state["mapa som"].sigma
    st.session_state["size"] = st.session_state["mapa som"].size
    st.session_state["lr"] = st.session_state["mapa som"].lr
    st.session_state["epochs"] = st.session_state["mapa som"].epochs
    st.session_state["cluster_distance"] = st.session_state["mapa som"].cluster_distance
    st.session_state["topology"] = st.session_state["mapa som"].topology
    st.session_state["output_influences"] = st.session_state["mapa som"].output_influences

def S2P2_update_parameters(sigma, size, lr, epochs, cluster_distance, topology, output_influences) -> bool:
    changed = False

    if st.session_state["mapa som"].sigma != sigma:
        st.session_state["mapa som"].sigma = sigma
        changed = True

    if st.session_state["mapa som"].size != size:
        st.session_state["mapa som"].size = size
        changed = True

    if st.session_state["mapa som"].lr != lr:
        st.session_state["mapa som"].lr = lr
        changed = True

    if st.session_state["mapa som"].epochs != epochs:
        st.session_state["mapa som"].epochs = epochs
        changed = True

    if st.session_state["mapa som"].cluster_distance != cluster_distance:
        st.session_state["mapa som"].cluster_distance = cluster_distance
        changed = True

    if st.session_state["mapa som"].topology != topology:
        st.session_state["mapa som"].topology = topology
        changed = True

    if st.session_state["mapa som"].output_influences != output_influences:
        st.session_state["mapa som"].output_influences = output_influences
        changed = True

    return changed

report_page_top("mapa som", S2P2_MapaSom, "2. Análise por Agrupamentos", 6/PAGE_COUNT, set_default_parameters=S2P2_set_default_parameters, load_parameters=S2P2_load_parameters)
st.subheader('Seção 2 - Mapa SOM')
st.markdown('''Um Mapa SOM, ou Mapa Auto-Organizável, é uma técnica de aprendizado não supervisionado usada para
             visualizar e organizar dados complexos em uma representação bidimensional.''')

som_iter = create_map(
    deepcopy(st.session_state["base reader"].crunched_database_average),
    cluster_distance=st.session_state["mapa som"].cluster_distance,
    lr=10**st.session_state["mapa som"].lr,
    epochs=st.session_state["mapa som"].epochs,
    size=st.session_state["mapa som"].size,
    sigma=st.session_state["mapa som"].sigma,
    label_column=st.session_state["base reader"].name_columns[0],
    output_column=st.session_state["base reader"].output_columns[0] if st.session_state["base reader"].output_columns else "",
    variable_columns=st.session_state["base reader"].input_columns,
    interval_epochs=st.session_state["mapa som"].epochs//20,
    output_influences=st.session_state["mapa som"].output_influences,
    topology="rectangular" if st.session_state["mapa som"].topology == "Retangular" else "hexagonal"
)

som_bar = st.progress(0.0)
if st.session_state["mapa som"].map is None:
    with st.empty():
        for i,som_data in enumerate(som_iter):
            som_bar.progress((i+1)/20, text="Carregando Mapa SOM" if i < 19 else "Mapa SOM")
            chart_data = som_data
            chart_data.columns = ["Municípios", "Nota", "x", "y", "Cor", "Grupo"]
            st.session_state["mapa som"].df = deepcopy(chart_data)

            if st.session_state["mapa som"].topology == "Retangular":
                c = alt.Chart(chart_data).mark_square(filled=True).encode(
                    x="x",
                    y="y",
                    size="Nota",
                    color=alt.Color("Cor:N", scale=None),
                    opacity=alt.value(1),
                    tooltip=["Grupo", "Nota", "Municípios"]
                ).interactive().configure_view(fill='black').properties(width=400, height=400)
            else:
                c = alt.Chart(chart_data).mark_point(filled=True, shape=HEX_SHAPE).encode(
                    x="x",
                    y="y",
                    size="Nota",
                    color=alt.Color("Cor:N", scale=None),
                    opacity=alt.value(1),
                    tooltip=["Grupo", "Nota", "Municípios"]
                ).interactive().configure_view(fill='black').properties(width=400, height=400)
            st.session_state["mapa som"].map = deepcopy(c)
            st.altair_chart(st.session_state["mapa som"].map, use_container_width=True)
else:
    som_bar.progress(1.0, "Mapa SOM")
    st.altair_chart(st.session_state["mapa som"].map, use_container_width=True)

st.info('Figura 1.2.1 - Mapa SOM')
st.divider()
st.markdown('Caso deseje modificar os parâmetros da criação do mapa SOM acima, clique para modificar os parâmetros.')
with st.expander("**Modificar Parâmetros do SOM**", expanded=False):
    with st.form("som_form"):
        st.markdown('Essa é uma opção avançada que acabará modificando a estruturação do mapa que foi gerado acima. Leia as instruções sobre cada parâmetro e ajuste conforme sua vontade.')
        sigma = st.slider("Sigma", min_value=1, max_value=10, value=st.session_state["sigma"], help="A largura da vizinhança inicial no mapa SOM. Controla a extensão das alterações que ocorrem durante o treinamento. Um valor alto significa que mais neurônios serão influenciados durante o treinamento inicial, enquanto um valor baixo resultará em um ajuste mais fino.")
        size = st.slider("Tamanho do mapa", min_value=5, max_value=50, value=st.session_state["size"], help="O tamanho do mapa SOM, especificado pelo número total de neurônios (unidades). Mapas maiores podem representar características complexas com maior precisão, mas também requerem mais tempo de treinamento.")
        lr = st.slider("Taxa de aprendizado", min_value=-5.0, max_value=-1.0, value=st.session_state["lr"], step=0.25, help="Taxa de aprendizado inicial. Controla a velocidade de adaptação do mapa durante o treinamento. Valores muito altos podem levar a uma convergência instável, enquanto valores muito baixos podem resultar em um treinamento lento.")
        epochs = st.slider("Épocas", min_value=100, max_value=30000, step=100, value=st.session_state["epochs"], help="Número de épocas (iterações) de treinamento. O número de vezes que o mapa será treinado em relação aos dados de entrada. Mais épocas geralmente resultam em um mapa mais bem ajustado, mas também aumentam o tempo de treinamento.")
        cluster_distance = st.slider("Distância dos agrupamentos", min_value=0.5, max_value=3.0, step=0.25, value=st.session_state["cluster_distance"], help="A distância mínima entre agrupamentos de neurônios para considerar a formação de grupos distintos. Valores mais altos podem resultar em agrupamentos mais distintos, enquanto valores mais baixos podem mesclar grupos semelhantes.")
        topology = st.radio("Topologia", options=["Retangular", "Hexagonal"], index=["Retangular", "Hexagonal"].index(st.session_state["topology"]), help="Topologia do mapa SOM para formação de vizinhanças.")
        output_influences = st.radio("Coluna de saída influencia nos resultados (experimental)", options=["Sim", "Não"], index = 0 if st.session_state["output_influences"] else 1, help="Se a coluna de saída dos dados de entrada influencia nos resultados finais. Selecione 'Sim' para permitir que a coluna de saída tenha impacto na organização do mapa, ou 'Não' para desconsiderar a coluna de saída durante o treinamento.")
        if st.form_submit_button("**Alterar parâmetros**"):
            if S2P2_update_parameters(sigma, size, lr, epochs, cluster_distance, topology, output_influences == "Sim"):
                clear_cached_data(["shap", "analise grupos", "anomalias", "relatorio individual"])
                st.session_state["mapa som"].map = None
                st.session_state["mapa som"].df = None
                st.rerun()

if st.session_state["base reader"].output_columns:
    report_page_bottom("mapa som", "pages/s2p1_descricao_arquivo.py", "pages/s2p3_heatmap.py")
else:
    report_page_bottom("mapa som", "pages/s2p1_descricao_arquivo.py", "pages/s2p9_tabela_regioes.py")
