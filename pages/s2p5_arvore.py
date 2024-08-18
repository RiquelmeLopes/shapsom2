import streamlit as st
import pandas as pd
from copy import deepcopy
from page_classes import S2P5_Arvore
from my_utilities import report_page_top, report_page_bottom, generate_random_string, PAGE_COUNT
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt
from threading import Lock
import os
globals()["lock"] = Lock()

report_page_top("arvore", S2P5_Arvore, "2. Análise por Agrupamentos", 9/PAGE_COUNT)

st.subheader('Seção 3.2 - Análise de agrupamentos com Árvore de Decisão')
st.markdown('''Esta seção divide-se em duas partes: Primeiro, uma tabela que lista as variáveis utilizadas no modelo de árvore de decisão juntamente com sua importância relativa. Em seguida, a própria imagem da árvore de decisões.''')
st.markdown('''A importância de uma variável indica quanto ela contribui para a decisão final do modelo. Valores mais altos de importância sugerem que a variável tem um impacto maior na previsão do modelo. Dessa forma, quanto maior o valor
        de sua importância na tabela, maior a importância dessa variável em geral (desconsiderando agrupamentos). Da mesma forma, quanto mais alto ela estiver posicionada na Árvore de Decisão, maior sua importância.
        Lembrando que essa Árvore de Decisão mostra a importância das variáveis num contexto mais amplo e desconsidera a análise posterior utilizando agrupamentos.''')

with st.spinner("Gerando árvore..."):
    if st.session_state["arvore"].fig is None:
        df = deepcopy(st.session_state["base reader"].crunched_database_average)
        x = df[st.session_state["base reader"].input_columns]
        y = df[st.session_state["base reader"].output_columns]
        reg = DecisionTreeRegressor(max_depth=3, random_state=42)
        reg.fit(x, y)
        feature_importances = pd.DataFrame({"Variáveis": x.columns, "Importância": reg.feature_importances_}).sort_values("Importância", ascending=False)
        st.session_state["arvore"].feature_importances = feature_importances

        st.dataframe(feature_importances, use_container_width=True)
        st.info(f"Tabela 3.2.1 - Importância das Variáveis no Modelo de Árvore de Decisão")

        rand_imgname = os.path.join("tempfiles", f"{generate_random_string(10)}.png")
        st.session_state["arvore"].img_path = rand_imgname
        with globals()["lock"]:
            fig, ax = plt.subplots(figsize=(20, 20))
            tree.plot_tree(reg, ax=ax, feature_names=x.columns, filled=True, fontsize=10)
            st.session_state["arvore"].fig = deepcopy(fig)
            plt.savefig(rand_imgname, dpi=150, bbox_inches='tight', transparent=True)
            plt.close()
    st.pyplot(st.session_state["arvore"].fig, use_container_width=True)

st.info(f"Figura 3.2.1 - Árvore de Decisão")
report_page_bottom("arvore", "pages/s2p4_shap.py", "pages/s2p6_analise_grupos.py")