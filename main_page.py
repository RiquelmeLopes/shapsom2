import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from io import BytesIO
from page_classes import BaseReader
from copy import deepcopy
from my_utilities import parse_dataframe, generic_page_top, MODELO_DF, read_csv, clear_cached_data, generate_crunched_dataframes, remove_old_tempfiles
from streamlit_javascript import st_javascript

#docker build -t streamlit-app .
#docker run -p 8501:8501 streamlit-app
#docker run --cpus="4.0" --memory="4g" -p 8501:8501 streamlit-app

clear_cached_data()
if not "base reader" in st.session_state.keys():
    st.session_state["base reader"] = BaseReader()
    remove_old_tempfiles()

generic_page_top("**Aquisição de Dados e Parametrizações.**", 0.0)
w_val = int(st_javascript('window.parent.innerWidth', key='scr'))
st.session_state["page width"] = max(w_val, st.session_state["page width"]) if "page width" in st.session_state.keys() else w_val
st.markdown('Atente-se a como sua planilha está organizada! Tente deixá-la no formato do modelo padrão.')

# Criar o DataFrame
csv_buffer = BytesIO()
MODELO_DF.to_csv(csv_buffer, index=False)
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

file = st.file_uploader("**Faça upload da sua planilha**", type=['csv', 'xlsx'], help='Caso sua planilha já esteja no mesmo formato do modelo (ou seja, com as colunas semelhantes), faça o upload dela. Caso contrário, faça o download da planilha modelo e preencha com seus dados.')
if file:
    st.divider()
    st.markdown("Visualização das primeiras 5 linhas da planilha")
    df = (read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)).map(parse_dataframe)
    df.columns = [str(c) for c in df.columns]

    descriptions = df.iloc[0]

    if all(isinstance(x, str) for x in descriptions):
        df.drop(df.index[0], inplace=True)
        df = df.reset_index(drop=True)

        # Tipos verdadeiros
        df_types = [type(df.iloc[0].dropna().iloc[x]).__name__  for x in range(df.shape[1])]

        df_dropna = deepcopy(df)
        df_dropna = df_dropna.dropna()

        # Atualizando tipos
        for i in range(df_dropna.shape[1]):
            if df_types[i] != 'str':
                df_dropna[df_dropna.columns[i]] = df_dropna.iloc[:,i].astype(df_types[i]+'64')
    else:
        descriptions = None
        df_dropna = deepcopy(df)
        df_dropna = df_dropna.dropna()

    st.session_state["base reader"].filename = file.name
    st.session_state["base reader"].original_database = deepcopy(df)
    st.session_state["base reader"].workable_database = deepcopy(df_dropna)
    st.session_state["base reader"].descriptions = descriptions
    st.session_state["base reader"].textual_columns = list(st.session_state["base reader"].workable_database.select_dtypes(include=['object']).columns)
    st.session_state["base reader"].numeric_columns = list(st.session_state["base reader"].workable_database.select_dtypes(include=['float64', 'int64','int','float']).columns)
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
                st.session_state["base reader"].name_columns = st.multiselect("Nome", tc, default=[tc[0]], max_selections=1, help='Selecione a coluna que será usada como o identificador principal do conjunto de dados. Esta coluna geralmente contém valores únicos, como nomes de municípios. Por padrão, é a primeira coluna da sua planilha.', placeholder="Escolha uma coluna")
            with col2:
                st.session_state["base reader"].input_columns = st.multiselect("Entradas", nc, default=nc[:-1], help='As colunas marcadas como "Entrada" são aquelas que contêm as variáveis independentes. Estes são os dados que serão usados para analisar o valor de saída.', placeholder="Escolha uma coluna")
            with col3:
                st.session_state["base reader"].output_columns = st.multiselect("Saída", nc, default=[nc[-1]], max_selections=1, help='A coluna marcada como "Saída" contém a variável dependente ou o valor que se deseja prever ou analisar. Esta coluna representa o resultado que é influenciado pelos dados das colunas de entrada. Por padrão, deve ser a última coluna da sua planilha.', placeholder="Escolha uma coluna")
        if not st.session_state["base reader"].output_columns:
            st.markdown(":red[Você não adicionou uma variável de saída, algumas das seções não poderão ser geradas.]")
        
        if len(st.session_state["base reader"].input_columns) >= 2:
            _, _, _, _, _, _, _, _, col2 = st.columns(9)
            with col2:
                if st.button("Avançar", use_container_width=True):
                    st.session_state["base reader"].finished_selection = True
                    generate_crunched_dataframes()
                    if st.session_state["base reader"].output_columns:
                        st.switch_page("pages/s1p1_mapa_analise_variavel.py")
                    else:
                        st.switch_page("pages/s2p1_descricao_arquivo.py")
        else:
            st.markdown(":red[Você precisa selecionar ao menos duas colunas de \"Entradas\" para prosseguir com a análise.]")