import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from io import BytesIO
from my_utilities import generic_page_top, MODELO_DF, clear_cached_data, generate_crunched_dataframes

if not "base reader" in st.session_state.keys():
    st.switch_page("main_page.py")

if st.session_state["base reader"].finished_selection:
    st.session_state["nc_default"] = st.session_state["base reader"].name_columns
    st.session_state["ic_default"] = st.session_state["base reader"].input_columns
    st.session_state["oc_default"] = st.session_state["base reader"].output_columns
    st.session_state["base reader"].finished_selection = False

generic_page_top("**Aquisição de Dados e Parametrizações.**", 0.0)
st.markdown('Atente-se a como sua planilha está organizada! Tente deixá-la no formato do modelo padrão.')

# Criar o DataFrame
csv_buffer = BytesIO()
MODELO_DF.to_csv(csv_buffer, encoding='latin1', index=False)
csv_buffer.seek(0)

# Salvar em buffer XLSX
xlsx_buffer = BytesIO()
with pd.ExcelWriter(xlsx_buffer) as writer:
    MODELO_DF.to_excel(writer, index=False)
xlsx_buffer.seek(0)

with st.expander("**Gostaria de baixar o modelo padrão de planilha?**", expanded=False):
    col1, col2, _, _, _ = st.columns(5)
    with col1:
        st.download_button('Modelo CSV', data=csv_buffer.getvalue(), file_name='modelo_csv.csv', mime="text/csv",  help='Modelo CSV de planilha a ser enviada')
    with col2:
        st.download_button('Modelo Excel', data=xlsx_buffer.getvalue(),file_name='modelo_excel.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', help='Modelo Excel de planilha a ser enviada')

if st.button("Você já tem uma planilha selecionada, deseja selecionar uma nova?"):
    st.switch_page("main_page.py")

st.divider()
st.markdown("Visualização das primeiras 5 linhas da planilha")
st.dataframe(st.session_state["base reader"].original_database[:5], use_container_width=True)

st.divider()
tc = st.session_state["base reader"].textual_columns
nc = st.session_state["base reader"].numeric_columns
if not tc and not nc:
    st.markdown(":red[A sua planilha não possui uma coluna de texto nem uma coluna numérica, ambas são necessárias para a aplicação funcionar.]")
elif not tc:
    st.markdown(":red[A sua planilha não possui uma coluna de texto, ela é necessária para a aplicação funcionar.]")
elif not nc:
    st.markdown(":red[A sua planilha não possui uma coluna numérica, ela é necessária para a aplicação funcionar.]")
else:
    st.markdown("Caso deseje modificar a escolha de colunas padrões, clique na opção abaixo:")

    with st.expander("**Escolher colunas**", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state["base reader"].name_columns = st.multiselect("Nome", tc, default=st.session_state["nc_default"], max_selections=1, help='Selecione a coluna que será usada como o identificador principal do conjunto de dados. Esta coluna geralmente contém valores únicos, como nomes de municípios. Por padrão, é a primeira coluna da sua planilha.', placeholder="Escolha uma coluna")
        with col2:
            st.session_state["base reader"].input_columns = st.multiselect("Entradas", nc, default=st.session_state["ic_default"], help='As colunas marcadas como "Entrada" são aquelas que contêm as variáveis independentes. Estes são os dados que serão usados para analisar o valor de saída.', placeholder="Escolha uma coluna")
        with col3:
            st.session_state["base reader"].output_columns = st.multiselect("Saída", nc, default=st.session_state["oc_default"], max_selections=1, help='A coluna marcada como "Saída" contém a variável dependente ou o valor que se deseja prever ou analisar. Esta coluna representa o resultado que é influenciado pelos dados das colunas de entrada. Por padrão, deve ser a última coluna da sua planilha.', placeholder="Escolha uma coluna")
    if not st.session_state["base reader"].output_columns:
        st.markdown(":red[Você não adicionou uma variável de saída, algumas das seções da análise por agrupamentos não poderão ser geradas.]")
    
    if len(st.session_state["base reader"].input_columns) >= 2:
        _, _, _, _, _, _, _, _, col2 = st.columns(9)
        with col2:
            if st.button("Avançar", use_container_width=True):
                st.session_state["base reader"].finished_selection = True
                var_nc = st.session_state["nc_default"] != st.session_state["base reader"].name_columns
                var_ic = st.session_state["ic_default"] != st.session_state["base reader"].input_columns
                var_oc = st.session_state["oc_default"] != st.session_state["base reader"].output_columns
                if var_nc:
                    generate_crunched_dataframes()
                if any([var_nc, var_ic, var_oc]):
                    clear_cached_data()
                if st.session_state["base reader"].output_columns:
                    st.switch_page("pages/s1p1_mapa_analise_variavel.py")
                else:
                    st.switch_page("pages/s2p1_descricao_arquivo.py")
    else:
        st.markdown(":red[Você precisa selecionar ao menos duas colunas de \"Entradas\" para prosseguir com a análise.]")