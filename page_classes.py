import pandas as pd
from copy import deepcopy
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer
from reportlab import platypus
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
import os

class BaseReader():
    def __init__(self):
        self.filename = ""
        self.original_database = None
        self.workable_database = None

        self.textual_columns = []
        self.numeric_columns = []

        self.finished_selection = False
        self.name_columns = []
        self.input_columns = []
        self.output_columns = []

        self.crunched_database_average = None
        self.crunched_database_stdev = None

class S1P1_MapaExploratorio():
    def __init__(self):
        self.map = None
        self.img_path = ""
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="mapa_exploratorio.pdf"):
        # Essa é a imagem do mapa
        mapa_img = Image.open(self.img_path)
        print("#"*80)
        print("Escrevendo", name)
        print(mapa_img.size)
        print("#"*80)
        
class S1P2_AnaliseEstatistica():
    def __init__(self):
        self.dfmc = None
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="analise_estatistica.pdf"):
        # Esse é o dataframe com as estatísticas
        df: pd.DataFrame = deepcopy(self.dfmc)
        print("#"*80)
        print("Escrevendo", name)
        print(df)
        print("#"*80)

class S1P3_GraficoDispersao():
    def __init__(self):
        self.map = None
        self.img_path = ""
        self.on_report = False
        self.finished_selection = False

        self.variavel_dispersao = ""
    
    def write_page(self, name: str="grafico_dispersao.pdf"):
        # Essa é a imagem da dispersão
        dispersao_img = Image.open(self.img_path)
        print("#"*80)
        print("Escrevendo", name)
        print(dispersao_img.size)
        print("#"*80)

class S2P1_DescricaoArquivo():
    def __init__(self):
        self.df = None
        self.descricao = ""
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="descricao_arquivo.pdf"):
        # Esse é o dataframe utilizado com as informações de cada coluna
        df: pd.DataFrame = deepcopy(self.df)
        # Essa é a descrição que fala quando o arquivo foi gerado, quantas colunas tem, blablabla
        descricao: str = self.descricao
        print("#"*80)
        print("Escrevendo", name)
        print(descricao)
        print(df)
        print("#"*80)

class S2P2_MapaSom():
    def __init__(self):
        self.map = None
        self.df = None
        self.on_report = False
        self.finished_selection = False

        self.sigma = 0
        self.size = 0
        self.lr = 0
        self.epochs = 0
        self.cluster_distance = 0
        self.topology = ""
        self.output_influences = False
    
    def write_page(self, name: str="mapa_som.pdf"):
        # É só puxar os parâmetros em cima pra compor o relatório
        sigma: int = self.sigma
        size: int = self.size
        #...
        print("#"*80)
        print("Escrevendo", name)
        print(self.sigma, self.size, self.lr, self.epochs, self.cluster_distance, self.topology, self.output_influences)
        print("#"*80)

class S2P3_Heatmap():
    def __init__(self):
        self.avg_df = None
        self.std_df = None
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="heatmap.pdf"):
        # Esses são os dataframes com média e desvio padrão, tem que transformar em imagem e partir se for muito grande
        avg: pd.DataFrame = deepcopy(self.avg_df)
        std: pd.DataFrame = deepcopy(self.std_df)
        print("#"*80)
        print("Escrevendo", name)
        print(avg)
        print(std)
        print("#"*80)

class S2P4_Shap():
    def __init__(self):
        self.df = None
        self.on_report = False
        self.finished_selection = False

    def write_page(self, name: str="shap.pdf"):
        # Esse é o dataframe com o shap das colunas nos grupos
        df: pd.DataFrame = deepcopy(self.df)
        print("#"*80)
        print("Escrevendo", name)
        print(df)
        print("#"*80)

class S2P5_Arvore():
    def __init__(self):
        self.img_path = ""
        self.fig = None
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="arvore.pdf"):
        # A imagem da árvore
        img = Image.open(self.img_path)
        print("#"*80)
        print("Escrevendo", name)
        print(img.size)
        print("#"*80)

class S2P6_AnaliseGrupos():
    def __init__(self):
        self.output_averages = []
        self.municipio_dfs = []
        self.shap_dfs = []
        self.maps = []
        self.image_paths = []

        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="analise_grupos.pdf"):
        # Setup document
        doc = SimpleDocTemplate(name, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        # For each group, generate a page
        for i, (output_average, municipio_df, shap_df, image_path) in enumerate(list(zip(self.output_averages, self.municipio_dfs, self.shap_dfs, self.image_paths))):
            # Title
            title = f"Grupo {i+1}"
            elements.append(Paragraph(title, styles['Title']))

            # Output averages section
            if isinstance(output_average, dict):
                data = [['Feature', 'Value']] + list(output_average.items())
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
            elif isinstance(output_average, (float, int)):
                elements.append(Paragraph(f"Average value: {output_average}", styles['Normal']))
            else:
                elements.append(Paragraph("Output averages not available", styles['Normal']))

            elements.append(Spacer(1, 12))

            # Municipio dataframe table
            if not municipio_df.empty:
                data = [municipio_df.columns.tolist()] + municipio_df.values.tolist()
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))

            # Shap dataframe table
            if not shap_df.empty:
                data = [shap_df.columns.tolist()] + shap_df.values.tolist()
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))

            # Add image
            if os.path.exists(image_path):
                img = platypus.Image(image_path)
                img.drawHeight = 200
                img.drawWidth = 200
                elements.append(img)

            # Page break between groups
            elements.append(Spacer(1, 48))

        # Build PDF
        doc.build(elements)
            
class S2P7_HeatmapFiltro():
    def __init__(self):
        self.avg_df = None
        self.std_df = None
        self.filtro_min = 0
        self.filtro_max = 100
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="heatmap_filtro.pdf"):
        # Esses são os dataframes com média e desvio padrão, tem que transformar em imagem e partir se for muito grande
        # Obs: Os dois já estão filtrados para facilitar, é só fazer a imagem aqui também
        avg: pd.DataFrame = deepcopy(self.avg_df)
        std: pd.DataFrame = deepcopy(self.std_df)
        print("#"*80)
        print("Escrevendo", name)
        print(self.filtro_min, self.filtro_max)
        print(avg)
        print(std)
        print("#"*80)

class S2P8_Anomalias():
    def __init__(self):
        self.porcentagem = 10
        self.df = None
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="anomalias.pdf"):
        # Esse é o dataframe que você vai usar. Essa parte já calcula quais são as anomalias
        anomalias_df: pd.DataFrame = deepcopy(self.df)
        anomalias_df = anomalias_df[:int(len(anomalias_df) * self.porcentagem / 100)]
        print("#"*80)
        print("Escrevendo", name)
        print(self.porcentagem)
        print(anomalias_df)
        print("#"*80)

class S2P9_TabelaRegioes():
    def __init__(self):
        self.df = None
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="tabela_regioes.pdf"):
        # Esse é o dataframe que você vai usar
        tabela_regioes: pd.DataFrame = deepcopy(self.df)
        print("#"*80)
        print("Escrevendo", name)
        print(tabela_regioes)
        print("#"*80)

class S3P1_RelatorioIndividual():
    def __init__(self):
        self.municipios = {}
    
    def write_page(self, municipio: str, name: str="relatorio_individual.pdf"):
        if municipio in self.municipios.keys():
            data = self.municipios[municipio]
            # Essa é a estrutura de dados que você tem acesso
            #data = {
            #    "dados": df,
            #    "grupo": _grupo,
            #    "nota_media_grupo": _avg_grupo,
            #    "nota_individual": _nota_municipio,
            #    "output": output_cols[0],
            #    "vizinhos": _closest_ones
            #}
            print("#"*80)
            print("Escrevendo", name)
            print(data)
            print("#"*80)