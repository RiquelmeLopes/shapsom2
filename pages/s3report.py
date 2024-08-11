from my_utilities import generate_individual_reports, PAGE_COUNT

generate_individual_reports(
    title="Relatório dos Municípios",
    progress=14/PAGE_COUNT,
    page_before="pages/s2report.py",
    page_after=""
)