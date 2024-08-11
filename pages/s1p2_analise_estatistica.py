import streamlit as st
from copy import deepcopy
from page_classes import S1P2_AnaliseEstatistica
from my_utilities import report_page_top, report_page_bottom, PAGE_COUNT

report_page_top("analise estatistica", S1P2_AnaliseEstatistica, "1. Análise Estatística Exploratória", 2/PAGE_COUNT)
st.subheader('Seção 2 - Análise Estatística')
st.markdown('''A tabela de estatísticas fornece um resumo estatístico descritivo da variável alvo para os municípios analisados. Os valores apresentados 
            incluem a contagem de observações, média, desvio padrão, valores mínimos e máximos, bem como os percentis 25%, 50% 
            (mediana) e 75%. Estas estatísticas são úteis para entender a distribuição e a variabilidade entre os municípios.''')
column_config = {'count': 'Contagem', 'mean': 'Média', 'std': 'Desvio Padrão', 'min': 'Mínimo', '25%': '1° Quartil', '50%': 'Mediana', '75%': '3° Quartil', 'max': 'Máximo'}

with st.spinner('Gerando análise...'):
    if st.session_state["analise estatistica"].dfmc is None:
        df = deepcopy(st.session_state["base reader"].crunched_database_average)
        nc = st.session_state["base reader"].name_columns[0]
        oc = st.session_state["base reader"].output_columns[0]
        dfmc = df.groupby(nc)[oc].apply(lambda x: x.mode().iloc[0]).reset_index()
        dfmc[dfmc.columns[-1]] = dfmc[dfmc.columns[-1]].round(2)
        dfmc_copy = dfmc[dfmc.columns[-1]].describe().to_frame().T
        st.session_state["analise estatistica"].dfmc = deepcopy(dfmc_copy)   
    st.dataframe(st.session_state["analise estatistica"].dfmc, column_config=column_config, hide_index=True, use_container_width=True)            
    st.info(f'Tabela 1 - Estatísticas Descritivas da Variável Alvo')

report_page_bottom("analise estatistica", "pages/s1p1_mapa_analise_variavel.py", "pages/s1p3_grafico_dispersao.py")