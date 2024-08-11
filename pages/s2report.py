from my_utilities import generate_report_page, PAGE_COUNT

generate_report_page(
    title="Relatório da Análise por Agrupamentos",
    progress=13/PAGE_COUNT,
    _ids=["descricao arquivo", "mapa som", "heatmap", "shap", "arvore", "analise grupos", "heatmap filtro", "anomalias", "tabela regioes"],
    _names=['Seção 1.1 - Dicionário de Dados', 'Seção 1.2 - Mapa SOM', 'Seção 2 - Visão dos Dados e Gráficos de Mapas de Calor', 'Seção 3.1 - Análise de agrupamentos com SHAP', 'Seção 3.2 - Análise de agrupamentos com Árvore de Decisão', 'Seção 4 - Diferenças entre Agrupamentos', 'Seção 5 - Filtro de Triagem', 'Seção 6 - Anomalias', 'Seção 7 - Identificação de Mesorregiões e Microrregiões'],
    page_before="pages/s2p9_tabela_regioes.py",
    page_after="pages/s3report.py"
)