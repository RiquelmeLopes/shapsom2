import pandas as pd
import streamlit as st
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re
import os
from copy import deepcopy
from difflib import SequenceMatcher
import numpy as np
from unidecode import unidecode
import statistics
import geopandas as gpd
from minisom import MiniSom
from scipy.cluster.hierarchy import linkage, fcluster
import colorsys
import matplotlib.pyplot as plt
import plotly.express as px
from xgboost import XGBRegressor
import shap
from folium import folium
from typing import Tuple
import random
import string
from threading import Lock
import time
from pypdf import PdfReader, PdfWriter

globals()["lock"] = Lock()

generate_random_string = lambda length: ''.join(random.choices(string.ascii_letters + string.digits, k=length))

PAGE_COUNT = len(os.listdir("pages")) - 2
HEX_SHAPE = "M0,-1 L0.866,-0.5 L0.866,0.5 L0,1 L-0.866,0.5 L-0.866,-0.5 Z"

MODELO_DF = pd.DataFrame(
    data=[[
        'Esse será o nome de "Rótulo" que a aplicação usará. Idealmente, deve ser nomeada como "Municípios" contendo seus nomes.',
        'Entrada de dados numéricos que será usada para os cálculos finais',
        'Entrada de dados numéricos que será usada para os cálculos finais',
        'Entrada de dados numéricos que será usada para os cálculos finais',
        'Essa será a coluna final, onde contém a variável dependente ou o valor que se deseja prever ou analisar. Esta coluna representa o resultado que é influenciado pelos dados das colunas de entrada'
    ]],
    columns=['Nome', 'Entrada 1', 'Entrada 2', 'Entrada n', 'Saída']
)

GDF: gpd.GeoDataFrame = gpd.read_file("required_files/PE_Municipios_2022.zip")
REGIONS_TABLE = pd.read_csv("required_files/Regiões-PE.csv", encoding="utf-8")[["Nome Município", "Mesorregião", "Microrregião"]]
CITIES = list(GDF["NM_MUN"].values)

# Parse all columns as floats, ints or strings
def parse_dataframe(v):
    if isinstance(v, float) or isinstance(v, int):
        return v
    elif str(v).isdigit():
        return int(v)
    elif re.match(r'^\d+[.,]\d*$', str(v)):
        return float(str(v).replace(",", "."))
    elif pd.isnull(v):
        return pd.NA
    else:
        return str(v)

# Generic header for any page
def generic_page_top(subheader: str, progress: float):
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
    st.markdown("""<style>[data-testid="collapsedControl"] {display: none}</style>""", unsafe_allow_html=True)
    st.image(Image.open(os.path.join("required_files", "tce_logo.png")), use_column_width=True)
    st.title("Sistema de Apoio a Auditorias do Tribunal de Contas do Estado 📊")
    st.progress(progress)    
    st.title(subheader)
    st.divider()

# Header for the report pages
def report_page_top(class_variable: str, class_type, subheader: str, progress: float, set_default_parameters=lambda: None, load_parameters=lambda: None):
    if not "base reader" in st.session_state.keys():
        st.switch_page("main_page.py")

    if not class_variable in st.session_state.keys():
        st.session_state[class_variable] = class_type()
        st.session_state["on_report"] = True
        set_default_parameters()

    if st.session_state[class_variable].finished_selection:
        st.session_state["on_report"] = st.session_state[class_variable].on_report
        load_parameters()
        st.session_state[class_variable].finished_selection = False

    generic_page_top(subheader, progress)

def generic_page_bottom(page_before: str, page_after: str):
    col1, _, _, _, _, _, _, _, col2 = st.columns(9)
    with col1:
        if page_before:
            if st.button("Voltar", use_container_width=True):
                with st.spinner("Carregando página..."):
                    st.switch_page(page_before)
    with col2:
        if page_after:
            if st.button("Avançar", use_container_width=True):
                with st.spinner("Carregando página..."):
                    st.switch_page(page_after)

# Functionality to advance or go back one page
def report_page_bottom(class_variable: str, page_before: str, page_after: str, update_parameters=lambda: None):
    st.session_state[class_variable].on_report = st.checkbox("Incluir no relatório", value=st.session_state["on_report"])
    update_parameters()
    col1, _, _, _, _, _, _, _, col2 = st.columns(9)
    with col1:
        if page_before:
            if st.button("Voltar", use_container_width=True):
                st.session_state[class_variable].finished_selection = True
                with st.spinner("Carregando página..."):
                    st.switch_page(page_before)
    with col2:
        if page_after:
            if st.button("Avançar", use_container_width=True):
                st.session_state[class_variable].finished_selection = True
                with st.spinner("Carregando página..."):
                    st.switch_page(page_after)

def get_numeric_column_description(vals: list, is_na: bool):
    def replace_last_comma(s):
        last_comma_index = s.rfind(',')
        return s if last_comma_index == -1 else s[:last_comma_index] + ' e ' + s[last_comma_index + 1:]

    val_string = replace_last_comma(", ".join([str(int(v)) if v == 0 or v == 1 else str(v) for v in vals]))
    if len(vals) == 2 and 1 in vals and 0 in vals:
        return f"Binário (0 ou 1) com valores faltantes" if is_na else f"Binário (0 ou 1)"
    elif len(vals) <= 4:
        return f"Numérico ({val_string}) com valores faltantes" if is_na else f"Numérico ({val_string})"
    elif min(vals) >= 0 and max(vals) <= 1:
        return f"Numérico (entre 0 e 1) com valores faltantes" if is_na else f"Numérico (entre 0 e 1)"
    else:
        return f"Numérico com valores faltantes" if is_na else f"Numérico"

# Reads a CSV file avoiding bad formatting by choosing the one with less errors
def read_csv(file) -> pd.DataFrame:
    count_parsing_errors = lambda s : max(0, len(s) - len(re.sub(r'[^\w\s\.,!?:;]', '', s)))
    csv_dfs = []
    for encoding in ['utf-8', 'latin-1']:
        try:
            _df = pd.read_csv(deepcopy(file), sep=",", encoding=encoding, encoding_errors="replace")
            _errors = sum(sum([count_parsing_errors(str(v)) for v in _df[c].values]) for c in _df.columns)
            csv_dfs.append((_df, _errors))
        except:
            pass
    return min(csv_dfs, key=lambda a : a[1])[0]

# Clears cached values from the report pages
def clear_cached_data(sections=["mapa exploratorio", "analise estatistica", "grafico dispersao", "descricao arquivo", "mapa som", "heatmap", "shap", "arvore", "analise grupos", "heatmap filtro", "anomalias", "tabela regioes", "relatorio individual"]):
    for k in sections:
        if k in st.session_state.keys():
            del st.session_state[k]

# Generates the workable dataframes for average and standard deviation
def generate_crunched_dataframes():
    with st.spinner("Carregando informações..."):
        tc = [st.session_state["base reader"].name_columns[0]]
        nc = st.session_state["base reader"].numeric_columns
        mun_names = list(REGIONS_TABLE["Nome Município"].values)

        crunched_df: pd.DataFrame = deepcopy(st.session_state["base reader"].workable_database)[tc + nc]
        crunched_df[tc[0]] = correct_city_names(crunched_df[tc[0]].values)
        crunched_df[nc] = crunched_df[nc].clip(lower=0, upper=1)

        municipio_dfs = crunched_df.groupby(tc[0])
        list_of_dfs = [group_df for _, group_df in municipio_dfs]
        new_df_avg = []
        new_df_std = []

        for l in list_of_dfs:
            flatlist = lambda a : np.array(a).flatten().tolist()
            calc_std = lambda a : statistics.stdev(a) if len(a) > 1 else 0
            mun_txt = [l[tc[0]].values[0]]
            mun_avg = [float(np.average(flatlist(l[c].values))) for c in nc]
            mun_std = [float(calc_std(flatlist(l[c].values))) for c in nc]
            new_df_avg.append(mun_txt + mun_avg)
            new_df_std.append(mun_txt + mun_std)
        
        dfavg = pd.DataFrame(new_df_avg, columns=crunched_df.columns)
        dfavg = dfavg.sort_values(by=tc, key=lambda col: col.map(unidecode)).reset_index(drop=True)
        st.session_state["base reader"].crunched_database_average = deepcopy(dfavg)
        dfstd = pd.DataFrame(new_df_std, columns=crunched_df.columns)
        dfstd = dfstd.sort_values(by=tc, key=lambda col: col.map(unidecode)).reset_index(drop=True)
        st.session_state["base reader"].crunched_database_stdev = deepcopy(dfstd)

def correct_city_names(vals: 'list[str]'):
    solutions_dict = {}
    new_vals = []
    for v in vals:
        if not v in solutions_dict.keys():
            similarities = [SequenceMatcher(None, unidecode(v).lower(), unidecode(c).lower()).ratio() for c in CITIES]
            solutions_dict[v] = CITIES[int(np.argmax(similarities))]
        new_vals.append(solutions_dict[v])
    return new_vals

def make_map(df: pd.DataFrame, name_column: str, output_column: str, color: str="") -> Tuple[folium.Map, str]:
    gdf = deepcopy(GDF)
    gdf = gdf[~gdf.isin(['Fernando de Noronha']).any(axis=1)]
    intersect_gdf = gdf.merge(df[[output_column, name_column]], left_on='NM_MUN', right_on=name_column).dropna(subset=[output_column, name_column])
    intersect_gdf[output_column] = intersect_gdf[output_column].round(2)
    col_values = [intersect_gdf.loc[intersect_gdf['NM_MUN'] == mun, output_column].tolist()[0] if mun in intersect_gdf['NM_MUN'].values else -1 for mun in gdf['NM_MUN'].values]
    gdf[output_column] = col_values

    def style_function_map(feature):
        value = feature['properties'][output_column]
        if value < 0:
            return {'fillColor': 'gray', 'color': 'gray', 'fillOpacity': 0.15, 'weight': 0.5, 'clickable': False}
        else:
            if color:
                return {'fillColor': color, 'color': 'gray', 'fillOpacity': 0.9, 'weight': 0.5}
            else:
                return {'color': 'gray', 'fillOpacity': 0.9, 'weight': 0.5}
    
    def style_function_image(row):
        value = row[output_column]
        if value < 0:
            return {'color': 'gray', 'edgecolor': 'gray', 'alpha': 0.15}
        else:
            if color:
                return {'color': color, 'edgecolor': 'black', 'alpha': 0.9}
            else:
                return {'color': plt.cm.viridis(value), 'edgecolor': 'black', 'alpha': 0.9}

    def combine_images(background: Image.Image, foreground: Image.Image) -> Image:
        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size

        new_fg_width = int(bg_width * 0.99)
        new_fg_height = int((fg_height / fg_width) * new_fg_width)
        
        foreground_resized = foreground.resize((new_fg_width, new_fg_height))
        
        fg_x = (bg_width - new_fg_width) // 2
        fg_y = (bg_height - new_fg_height) // 2
        
        combined_image = Image.new("RGBA", (bg_width, bg_height))
        combined_image.paste(background, (0, 0))
        combined_image.paste(foreground_resized, (fg_x, fg_y), foreground_resized)
        return combined_image

    rand_imgname = os.path.join("tempfiles", f"{generate_random_string(10)}.png")
    map = gdf.explore(column=output_column, vmin=0, vmax=1, fitbounds="locations", map_kwrds={'scrollWheelZoom': 4}, style_kwds={'style_function': style_function_map}, use_container_width=True, categorical=False, legend=len(color) == 0)
    gdf['color'] = gdf.apply(lambda row: style_function_image(row)['color'], axis=1)
    gdf['edgecolor'] = gdf.apply(lambda row: style_function_image(row)['edgecolor'], axis=1)
    gdf['alpha'] = gdf.apply(lambda row: style_function_image(row)['alpha'], axis=1)

    with globals()["lock"]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        gdf.plot(ax=ax, edgecolor=gdf['edgecolor'], alpha=gdf['alpha'], color=gdf['color'])
        ax.set_axis_off()
        plt.savefig(rand_imgname, dpi=150, bbox_inches='tight', transparent=True)
        foreground = Image.open(rand_imgname)
        background = Image.open(os.path.join("required_files", "mapa_pe.png"))
        img = combine_images(background, foreground)
        img.save(rand_imgname)
        
    return map, rand_imgname

def display_heatmaps(filtro_min=0.0, filtro_max=1.0, numero_secao=2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def generate_heatmap(df: pd.DataFrame, name_column: str, variable_columns: 'list[str]', cmap):
        fig_height = int((st.session_state["page width"] / 2) * len(df) / len(df.columns))
        col_height = len(max(variable_columns, key=len)) * 4
        fig = px.imshow(df[variable_columns], color_continuous_scale=cmap, aspect='auto', zmin=0, zmax=1, labels=dict(x="Fatores", y="Municipios", color="Valor"), y=df[name_column])
        fig.update_layout(coloraxis_colorbar=dict(orientation='v', ticks='outside', ticklen=1, tickwidth=1), height=fig_height + col_height, width=st.session_state["page width"] // 2)
        fig.update_traces(hovertemplate='Municipio: %{y}<br>Variável: %{x}<br>Valor: %{z}')
        return fig
    
    col1, col2 = st.columns(2)
    name_col = st.session_state["base reader"].name_columns[0]
    num_cols = st.session_state["base reader"].input_columns + st.session_state["base reader"].output_columns
    out_col = st.session_state["base reader"].output_columns[0]

    with st.spinner('Gerando mapa...'):
        crunched_average = deepcopy(st.session_state["base reader"].crunched_database_average)
        crunched_average = crunched_average[(crunched_average[out_col] >= filtro_min) & (crunched_average[out_col] <= filtro_max)]
        municipios_filtrados = list(crunched_average[name_col].values)

        with col1:
            st.dataframe(crunched_average)
            st.info(f"**Tabela {numero_secao}.1 - Média**")
            heatmap1 = generate_heatmap(crunched_average, name_col, num_cols, 'YlOrRd')
            st.plotly_chart(heatmap1, use_container_width=True)
            st.info(f'Gráfico {numero_secao}.1 - Mapa de Calor (Heatmap) da Média dos Dados dos Municípios')

        crunched_stdev = deepcopy(st.session_state["base reader"].crunched_database_stdev)
        crunched_stdev = crunched_stdev[crunched_stdev[name_col].isin(municipios_filtrados)]

        with col2:
            st.dataframe(crunched_stdev)
            st.info(f"**Tabela {numero_secao}.2 - Desvio Padrão**")
            heatmap2 = generate_heatmap(crunched_stdev, name_col, num_cols, 'gray')
            st.plotly_chart(heatmap2, use_container_width=True)
            st.info(f'Gráfico {numero_secao}.2 - Mapa de Calor (Heatmap) do Desvião Padrão dos Dados dos Municípios')
    
    return crunched_average, crunched_stdev

### SOM
def hsv_to_hex(hsv) -> str:
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    r,g,b = int(r * 255), int(g * 255), int(b * 255)
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    return hex_color

def create_map(df: pd.DataFrame, label_column: str, variable_columns: 'list[str]', output_column: str, size: int = 50, lr: float = 1e-1, epochs: int = 1000, sigma = 2, topology = "hexagonal", cluster_distance: float = 2, interval_epochs: int=100, output_influences=True):
    def cluster_coordinates(coordenadas: 'list[tuple]', distancia_maxima: float) -> 'list[list[tuple]]':
        mat = linkage(coordenadas, method='single', metric='chebyshev')
        cluster_ids = fcluster(mat, distancia_maxima, criterion='distance')
        elements_with_id = lambda id : np.array(np.where(cluster_ids == id), dtype=int).flatten().tolist()
        clusters = [[coordenadas[i] for i in elements_with_id(id)] for id in set(cluster_ids)]
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters
    
    def get_som_data(som: MiniSom, labels, x, y, cluster_distance) -> pd.DataFrame:
        units = som.labels_map(x, labels)
        is_hex = som.topology == "hexagonal"
        units = {((_x + 0.5) if (is_hex and _y % 2 == 1) else (_x), _y) : units[_x,_y] for _x,_y in units.keys()}
        clusters = cluster_coordinates(list(units.keys()), cluster_distance)
        
        cluster_amount = len(clusters)
        hsv_boundaries = ((np.arange(cluster_amount+1) / (cluster_amount+1)) * 360).flatten().tolist()
        hsv_region_width = (360 / (cluster_amount+1)) * 0.25
        hsv_centers = np.array([(hsv_boundaries[i] + hsv_boundaries[i+1]) / 2 for i in range(cluster_amount)]).flatten().tolist()
        hsv_regions = np.array([(c-(hsv_region_width/2), c+(hsv_region_width/2)) for c in hsv_centers]).tolist()
        som_data_list = []

        for i, coords in enumerate(clusters):
            coord_dict = {}
            cell_scores = []
            for c in filter(lambda _c : units[_c], coords):
                cell_labels = list(units[c])
                score_data = np.array([y.loc[y['label'] == u, y.columns[-1]].values[0] for u in cell_labels]).flatten()
                variables = [x[list(labels).index(v)] for v in units[c]]
                cell_score = np.average(score_data)
                cell_scores.append(cell_score)
                score_dict = {}
                for score_type in y.columns[1:]:
                    data = [y.loc[y['label'] == u, y.columns[-1]].values[0] for u in cell_labels]
                    score_dict[score_type] = data
                coord_dict[c] = {"labels": cell_labels, "score": score_dict, "scores": score_data, "variables": variables, "cell_score": cell_score, "cluster": i}
        
            min_hsv, max_hsv = hsv_regions[i]
            central_color = hsv_to_hex([(min_hsv+max_hsv / 2), 1, 1])
            
            for c in coord_dict.keys():
                for municipio in coord_dict[c]["labels"]:
                    _labels = municipio
                    _score = coord_dict[c]["cell_score"]
                    _x = c[0]
                    _y = c[1]
                    som_data_list.append([_labels, round(_score, 2), _x, _y, central_color, i+1])

            score_dict = {}
            for c in coord_dict.keys():
                for score_type in coord_dict[c]["score"].keys():
                    if not score_type in score_dict.keys():
                        score_dict[score_type] = []
                    score_dict[score_type] += coord_dict[c]["score"][score_type]

            c_score = {}
            for k in score_dict.keys():
                c_score[k] = np.average(score_dict[k])

            for c in coord_dict.keys():
                coord_dict[c]["cluster_scores"] = c_score

        som_data = pd.DataFrame(som_data_list, columns=["labels", "score", "x", "y", "color", "cluster"])
        return som_data

    labels = df[label_column]

    x = np.array(df[variable_columns].select_dtypes(include='number').values)
    if output_influences and output_column:
        val_multipliers = df[output_column].values
        new_x = []
        for r,m in zip(x, val_multipliers):
            new_r = r if np.sum(r) != 0 else np.ones_like(r)
            new_r /= np.sum(new_r)
            new_x.append(new_r * m)
        x = np.array(new_x)
    
    if output_column:
        y = pd.concat([pd.DataFrame({"label": labels}), df[output_column]], axis=1)
    else:
        y = pd.concat([pd.DataFrame({"label": labels}), pd.DataFrame({"Média": np.average(x, axis=1).tolist()})], axis=1)
    som = MiniSom(
        x=size,
        y=size,
        input_len=len(x[0]),
        sigma=sigma,
        topology=topology,
        learning_rate=lr,
        neighborhood_function="gaussian",
        activation_distance="euclidean"
    )
    
    som.pca_weights_init(x)
    for _ in range(epochs // interval_epochs):
        som.train(x, interval_epochs, verbose=False)
        som_data = get_som_data(som, labels, x, y, cluster_distance)
        yield som_data

### SHAP
def calculate_shap(df: pd.DataFrame, input_columns: 'list[str]', output_columns: 'list[str]', return_average=True) -> np.ndarray:
    x = df[input_columns]
    y = df[output_columns]
    model = XGBRegressor(objective='reg:squarederror', random_state=34)
    model.fit(x, y)
    explainer = shap.Explainer(model, x)
    shap_explanations = [shap.Explanation(values=v, base_values=explainer.expected_value, feature_names=input_columns) for v in explainer(x)]
    
    if return_average:
        _result = np.average(np.array([np.array(exp.values).flatten() for exp in shap_explanations]), axis=0)
        if np.sum(_result) != 0:
            _result /= np.sum(_result)
        _result *= np.average(y[output_columns[0]].values)
    else:
        _result = []
        for _x, _y in zip([np.array(exp.values).flatten() for exp in shap_explanations], y[output_columns[0]].values.tolist()):
            if np.sum(_x) != 0:
                _result.append((_x / np.sum(_x)) * _y)
            else:
                _result.append(_x * _y)
        _result = np.array(_result)
    return _result

def remove_old_tempfiles():
    with globals()["lock"]:
        directory = "tempfiles"
        current_time = time.time()

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                file_age = current_time - file_mtime
                
                if file_age > 24 * 60 * 60:
                    try:
                        os.remove(file_path)
                    except:
                        pass

def merge_pdfs(pdf_list: 'list[str]', output_path: str):
    pdf_writer = PdfWriter()

    for pdf in filter(lambda a : os.path.exists(a), pdf_list):
        pdf_reader = PdfReader(pdf)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(output_path, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)

# Página de relatório genérica
def generate_report_page(title: str, progress: float, _ids: 'list[str]', _names: 'list[str]', page_before: str, page_after: str):
    generic_page_top(title, progress)
    _id_name_pairs = [(_id, _name) for _id, _name in zip(_ids, _names) if _id in st.session_state.keys()]
    selectable_sections = [_id for _id, _ in _id_name_pairs]
    section_names = [_name for _, _name in _id_name_pairs]
    include_on_report = []

    with st.form("report_form"):
        st.markdown('Selecione as seções que deseja incluir no relatório')
        for section_id, section_name in zip(selectable_sections, section_names):
            if section_id in st.session_state.keys():
                include_on_report.append(st.checkbox(section_name, value=st.session_state[section_id].on_report))

        if st.form_submit_button("**Gerar relatório**"):
            # Atualiza a barra e o valor das seções
            _id_name_selected = []
            for _id, _name, _on_report in zip(selectable_sections, section_names, include_on_report):
                st.session_state[_id].on_report = _on_report
                if _on_report:
                    _id_name_selected.append((_id, _name))
            
            # Gera o relatório
            report_bar = st.progress(0.0)
            total_sections = include_on_report.count(True)
            pdf_names = []
            for i, (section_id, section_name) in enumerate(_id_name_selected):
                rand_pdf = os.path.join("tempfiles//", f"{generate_random_string(10)}.pdf")
                report_bar.progress(i / total_sections, f"Adicionando {section_name}...")
                with globals()["lock"]:
                    st.session_state[section_id].write_page(rand_pdf)
                    time.sleep(1) # É só pra simular um tempo de espera, tira isso depois
                pdf_names.append(rand_pdf)
            
            with globals()["lock"]:
                output_file = f"tempfiles//{generate_random_string(10)}.pdf"
                pdf_names = ['required_files//capa.pdf'] + pdf_names
                merge_pdfs(pdf_names, output_file)
                for f in filter(lambda a : os.path.exists(a) and a != 'required_files//capa.pdf', pdf_names):
                    os.remove(f)
            report_bar.progress(1.0, f"Relatório concluído.")

    try:
        if output_file and os.path.exists(output_file):        
            with open(output_file, "rb") as pdf_file:
                st.download_button(
                    label="Baixar relatório",
                    data=pdf_file,
                    file_name='Relatorio_Grupos_Final.pdf',
                    mime='application/pdf'
                )
    except:
        pass

    generic_page_bottom(page_before, page_after)

def generate_individual_reports(title: str, progress: float, page_before: str, page_after: str):
    generic_page_top(title, progress)
    with st.form("individual_form"):
        st.subheader("Selecione os Municípios")
        tc = st.session_state["base reader"].name_columns[0]
        labels = deepcopy(st.session_state["base reader"].crunched_database_average)[tc].values
        list_selected_labels = st.multiselect("Municípios", labels, help='Selecione os municípios para gerar os relatórios individuais de cada um deles')
        use_mark_all = st.checkbox("Selecionar Todos", help="Selecione para gerar o relatório de todos os municípios")
        if st.form_submit_button('**Gerar relatório**'):
            if use_mark_all:
                list_selected_labels = labels
            list_selected_labels.sort()
            report_bar = st.progress(0.0)
            pdf_names = []
            for i, municipio in enumerate(list_selected_labels):
                rand_pdf = os.path.join("tempfiles", f"{generate_random_string(10)}.pdf")
                report_bar.progress(i / len(list_selected_labels), f"Gerando relatório de {municipio}...")
                with globals()["lock"]:
                    st.session_state["relatorio individual"].write_page(municipio, rand_pdf)
                    time.sleep(0.05)
                pdf_names.append(rand_pdf)
            
            with globals()["lock"]:
                output_file = f"tempfiles//{generate_random_string(10)}.pdf"
                pdf_names = ['required_files//capa.pdf'] + pdf_names
                merge_pdfs(pdf_names, output_file)
                for f in filter(lambda a : os.path.exists(a) and a != 'required_files//capa.pdf', pdf_names):
                    os.remove(f)
            report_bar.progress(1.0, f"Relatórios concluídos.")
    try:
        if output_file and os.path.exists(output_file):        
            with open(output_file, "rb") as pdf_file:
                st.download_button(
                    label="Baixar relatório",
                    data=pdf_file,
                    file_name='Relatorios_Individuais_Final.pdf',
                    mime='application/pdf'
                )
    except:
        pass
    generic_page_bottom(page_before, page_after)