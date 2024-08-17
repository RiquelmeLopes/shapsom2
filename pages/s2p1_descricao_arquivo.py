import streamlit as st
import pandas as pd
from copy import deepcopy
from page_classes import S2P1_DescricaoArquivo
from my_utilities import report_page_top, report_page_bottom, get_numeric_column_description, PAGE_COUNT
from datetime import datetime

report_page_top("descricao arquivo", S2P1_DescricaoArquivo, "2. Análise por Agrupamentos", 5/PAGE_COUNT)

df: pd.DataFrame = deepcopy(st.session_state["base reader"].original_database)
_fname = st.session_state["base reader"].filename
_date = datetime.today().strftime("%d/%m/%Y")
_rows = len(st.session_state["base reader"].original_database)
_columns = len(st.session_state["base reader"].crunched_database_average.columns)
_na_values = "possui" if any([df[c].isna().any() for c in df.columns]) else "não possui"
descricao = f'''Este relatório foi elaborado com base nos dados presentes no arquivo "{_fname}" no dia {_date}.
             A tabela fornecida possui {_rows} linhas e {_columns} colunas. A tabela {_na_values} valores faltantes.
             A seguir, na tabela 1, apresentamos o dicionário de dados. É importante notar que colunas com texto
             ou aquelas que foram ocultadas durante a criação do mapa não foram incluídas na análise.'''

st.subheader('Seção 1 - Descrição do arquivo de entrada')
st.markdown('Essa seção tem como objetivo detalhar as especificações e requisitos dos dados necessários para o correto funcionamento do sistema.')
st.subheader('Seção 1.1 - Dicionário de Dados')
st.markdown(descricao)
st.markdown('''Um dicionário de dados é uma tabela que contém informações sobre os dados disponibilizados. As
             informações reveladas abaixo revelam o número atribuído a cada fator, sua descrição quando
             disponibilizada e seu tipo de dado.''')

oc = st.session_state["base reader"].output_columns

dict_data = []
for c in st.session_state["base reader"].name_columns:
    is_na = df[c].isna().any()
    if st.session_state["base reader"].descriptions is None:
        dict_data.append(["Nome", c, "Textual com valores faltantes" if is_na else "Textual"])
    else:
        dict_data.append(["Nome", c, st.session_state["base reader"].descriptions[c], "Textual com valores faltantes" if is_na else "Textual"])

for i,c in enumerate(st.session_state["base reader"].input_columns):
    is_na = df[c].isna().any()
    vals = list(set(sorted(df.dropna()[c].values)))
    if st.session_state["base reader"].descriptions is None:
        dict_data.append([str(i+1), c, get_numeric_column_description(vals, is_na)])
    else:
        dict_data.append([str(i+1), c,  st.session_state["base reader"].descriptions[c], get_numeric_column_description(vals, is_na)])
    
for c in st.session_state["base reader"].output_columns:
    is_na = df[c].isna().any()
    vals = list(set(sorted(df.dropna()[c].values)))
    if st.session_state["base reader"].descriptions is None:
        dict_data.append(["Saída", c, get_numeric_column_description(vals, is_na)])
    else:
        dict_data.append(["Saída", c, st.session_state["base reader"].descriptions[c], get_numeric_column_description(vals, is_na)])

if st.session_state["base reader"].descriptions is None:
    colunas = ["Fator", "Nome da coluna", "Tipo de dado"]
else:
    colunas = ["Fator", "Nome da coluna", "Descrição do dado","Tipo de dado"]

dict_df = pd.DataFrame(data=dict_data, columns=colunas)
st.session_state["descricao arquivo"].df = deepcopy(dict_df)
st.session_state["descricao arquivo"].descricao = descricao
st.dataframe(dict_df, hide_index=True, use_container_width=True)
st.info('Tabela 1.1.1 - Dicionário de Dados')

if st.session_state["base reader"].output_columns:
    report_page_bottom("descricao arquivo", "pages/s1report.py", "pages/s2p2_mapa_som.py")
else:
    report_page_bottom("descricao arquivo", "pages/s0p1_repetir_planilha.py", "pages/s2p2_mapa_som.py")