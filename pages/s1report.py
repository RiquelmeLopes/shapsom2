import streamlit as st
from my_utilities import generate_report_page, PAGE_COUNT

generate_report_page(
    title="Relatório da Análise Estatística Exploratória",
    progress=4/PAGE_COUNT,
    _ids=["mapa exploratorio", "analise estatistica", "grafico dispersao"],
    _names=["Seção 1 - Mapa de Análise da Variável Alvo", 'Seção 2 - Análise Estatística', 'Seção 3 - Gráfico de Dispersão'],
    page_before="pages/s1p3_grafico_dispersao.py",
    page_after="pages/s2p1_descricao_arquivo.py"
)