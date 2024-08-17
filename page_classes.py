import pandas as pd
from copy import deepcopy
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image as ReportlabImage
from reportlab import platypus
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
import os
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

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
        doc = SimpleDocTemplate(name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []
        styles = getSampleStyleSheet()

        if os.path.exists('required_files//cabecalho.jpeg'):
            elements.append(ReportlabImage('required_files//cabecalho.jpeg', width=500, height=50))
            elements.append(Spacer(1, 18))

        # Adiciona a imagem do mapa ao PDF
        if os.path.exists(self.img_path):
            mapa_img = Image.open(self.img_path)
            elements.append(Paragraph("Mapa Exploratório", styles['Title']))
            elements.append(ReportlabImage(self.img_path, width=400, height=300))
            elements.append(Spacer(1, 12))

        # Gera o PDF
        doc.build(elements)
        
class S1P2_AnaliseEstatistica():
    def __init__(self):
        self.dfmc = None
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="analise_estatistica.pdf"):
        df: pd.DataFrame = deepcopy(self.dfmc)
        doc = SimpleDocTemplate(name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []
        styles = getSampleStyleSheet()
        if os.path.exists('required_files//cabecalho.jpeg'):
            elements.append(ReportlabImage('required_files//cabecalho.jpeg', width=500, height=50))
            elements.append(Spacer(1, 18))

        elements.append(Paragraph("Análise Estatística", styles['Title']))
        if isinstance(df, pd.DataFrame) and not df.empty:
            data = [df.columns.tolist()] + df.values.tolist()
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

        # Gera o PDF
        doc.build(elements)

class S1P3_GraficoDispersao():
    def __init__(self):
        self.map = None
        self.img_path = ""
        self.on_report = False
        self.finished_selection = False

        self.variavel_dispersao = ""
    
    def write_page(self, name: str="grafico_dispersao.pdf"):
        doc = SimpleDocTemplate(name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []
        styles = getSampleStyleSheet()

        if os.path.exists('required_files//cabecalho.jpeg'):
            elements.append(ReportlabImage('required_files//cabecalho.jpeg', width=500, height=50))
            elements.append(Spacer(1, 18))

        # Adiciona o gráfico de dispersão ao PDF
        if os.path.exists(self.img_path):
            dispersao_img = Image.open(self.img_path)
            elements.append(Paragraph(f"Gráfico de Dispersão - {self.variavel_dispersao}", styles['Title']))
            elements.append(ReportlabImage(self.img_path, width=400, height=300))
            elements.append(Spacer(1, 12))

        # Gera o PDF
        doc.build(elements)

class S2P1_DescricaoArquivo():
    def __init__(self):
        self.df = None
        self.descricao = ""
        self.on_report = False
        self.finished_selection = False
        self.has_descriptions = False
    
    def write_page(self, name: str="descricao_arquivo.pdf"):
        # Esse é o dataframe utilizado com as informações de cada coluna
        df: pd.DataFrame = deepcopy(self.df).to_numpy()
        # Essa é a descrição que fala quando o arquivo foi gerado, quantas colunas tem, blablabla
        descricao: str = self.descricao

        decricao2 = '''Um dicionário de dados é uma tabela que contém informações sobre os dados disponibilizados. As
             informações reveladas abaixo revelam o número atribuído a cada fator, sua descrição quando
             disponibilizada e seu tipo de dado.'''
        
        self.has_descriptions = df.shape[1] == 4

        def insert_newlines(text, every=40):
            lines = []
            while len(text) > every:
                split_index = text.rfind(' ', 0, every)
                if split_index == -1:
                    split_index = every
                lines.append(text[:split_index].strip())
                text = text[split_index:].strip()
            lines.append(text)
            return '\n'.join(lines)

        def dividirlinhas(data, every):
            if(len(data)>every):
                data = insert_newlines(data, every=every)
            return data

        if(not self.has_descriptions):
            for i in range(df.shape[0]):
                df[i][1]=dividirlinhas(df[i][1],65)
            for i in range(df.shape[0]):
                df[i][2]=dividirlinhas(df[i][2],25)
        else:
            for i in range(df.shape[0]):
                df[i][1]=dividirlinhas(df[i][1],30)
            for i in range(df.shape[0]):
                df[i][2]=dividirlinhas(df[i][2],30)
            for i in range(df.shape[0]):
                df[i][3]=dividirlinhas(df[i][3],25)
        df = df.tolist()

        # Pdf
        page_w, page_h = letter
        c = canvas.Canvas(name)
        c, h = self.gerarSecao(c,'t','1. Descrição do arquivo de entrada',65)
        c, h = self.gerarSecao(c,'p','Essa seção tem como objetivo detalhar as especificações e requisitos dos dados necessários para o correto funcionamento do sistema.',65)
        c, h = self.gerarSecao(c,'s','1.1 Dados de Treinamento',h)
        c, h = self.gerarSecao(c,'p',descricao,h)
        c, h = self.gerarSecao(c,'p',decricao2,h-28)
        c, h = self.gerarSecaoTabela(c,h,df)
        c, h = self.gerarLegenda(c,'Tabela 1.1 - Dicionário de Dados', h+5)
        c.drawImage('required_files/cabecalho.jpeg', inch-8, page_h-50,page_w-inch-52,50)
        c.saveState()
        c.showPage()
        c.save()

    def gerarTabela(self, data):
        if(self.has_descriptions):
            data = [['Fator','Nome da coluna','Descrição do dado','Tipo de dado']]+data
        else:
            data = [['Fator','Nome da coluna','Tipo de dado']]+data

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 'WORD'),
        ])

        if(self.has_descriptions):
            col_widths = [40,165,165,100]
        else:
            col_widths = [40,330,100]

        table = Table(data, colWidths=col_widths)
        table.setStyle(style)
        return table
    
    def gerarTabelaPdf(self, c,data,h,start):
        page_w, page_h = letter
        if(len(data)>start):
            data2 = []
            end = 0
            for i in range(len(data)-start+1):
                table = self.gerarTabela(data2)
                w_paragrafo, h_paragrafo = table.wrapOn(c, 0, 0)
                if(page_h - h- h_paragrafo< inch):
                    end = i
                    break

                if(i<len(data)-start):
                    data2+= [data[i+start]]
            table.drawOn(c, inch, page_h - h- h_paragrafo)
            return c, h_paragrafo+h, start+end
        else:
            return c, h, start
        
    def quebraPagina(self, c, h, tamanho):
        page_w, page_h = letter
        if(h>tamanho):
            c.drawImage('required_files/cabecalho.jpeg', inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            h=65
        return c, h
    
    def gerarSecaoTabela(self, c,h,dados):
        start = 0
        start2 = 0
        while(True):
            c, h, start = self.gerarTabelaPdf(c,dados,h,start)
            if(start==start2):
                break
            else:
                c, h = self.quebraPagina(c, h, 0)
                start2=start
        return c, h

    def gerarSecao(self, c,tipo,paragrafo,h):
        page_w, page_h = letter
        if(tipo=='p'):
            style_paragrafo = ParagraphStyle("paragrafo", fontName="Helvetica", fontSize=12, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='t'):
            style_paragrafo = ParagraphStyle("titulo", fontName="Helvetica-Bold", fontSize=16, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='s'):
            style_paragrafo = ParagraphStyle("subtitulo", fontName="Helvetica-Bold", fontSize=14, alignment=4, leading=18, encoding="utf-8")
        message_paragrafo = Paragraph(paragrafo, style_paragrafo)
        w_paragrafo, h_paragrafo = message_paragrafo.wrap(page_w -2*inch, page_h)
        message_paragrafo.drawOn(c, inch, page_h - h- h_paragrafo)
        return c, h+h_paragrafo+30
    
    def gerarLegenda(self, c,paragrafo,h):
        page_w, page_h = letter
        style_paragrafo = ParagraphStyle("paragrafo", fontName="Helvetica-Oblique", fontSize=10, alignment=4, leading=18, encoding="utf-8", textColor = 'blue')
        message_paragrafo = Paragraph(paragrafo, style_paragrafo)
        w_paragrafo, h_paragrafo = message_paragrafo.wrap(page_w -2*inch, page_h)
        message_paragrafo.drawOn(c, inch, page_h - h- h_paragrafo)
        return c, h+h_paragrafo+20

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
        lr: float = self.lr
        epochs: int = self.epochs
        cluster_distance: float = self.cluster_distance
        topology: str = self.topology

        

        textoSOM = '''Um Mapa Auto-Organizável (SOM) é uma técnica de aprendizado não supervisionado usada para visualizar e organizar dados complexos em uma representação bidimensional. Os principais parâmetros que definem um mapa SOM incluem:
                    \n●	Topologia: Define como as células do mapa influenciam suas vizinhas em um arranjo geométrico.
                    \n●	Distância de cluster: Determina como as unidades são agrupadas com base na similaridade dos dados.
                    \n●	Épocas: Representam o número de vezes que o modelo passa pelos dados durante o treinamento.
                    \n●	Tamanho do mapa: Define o número total de unidades no mapa.
                    \n●	Sigma: O raio de influência de cada unidade durante o treinamento.
                    \n●	Taxa de aprendizado: Controla a magnitude das atualizações dos pesos das unidades durante o treinamento.
                    '''
        
        page_w, page_h = letter
        c = canvas.Canvas(name)
        c, h = self.gerarSecao(c,'s','1.2 Parâmetros de Treinamento',65)
        c, h = self.gerarSecao(c,'p', textoSOM.split('\n')[0], h)
        c, h = self.gerarSecao(c,'p', textoSOM.split('\n')[2], h-20)
        c, h = self.gerarSecao(c,'p', textoSOM.split('\n')[4], h-20)
        c, h = self.gerarSecao(c,'p', textoSOM.split('\n')[6], h-20)
        c, h = self.gerarSecao(c,'p', textoSOM.split('\n')[8], h-20)
        c, h = self.gerarSecao(c,'p', textoSOM.split('\n')[10], h-20)
        c, h = self.gerarSecao(c,'p', textoSOM.split('\n')[12], h-20)
        c, h = self.gerarSecao(c,'p','Nesta seção, apresentamos os hiperparâmetros utilizados para configurar o algoritmo. Os dados mencionados no parágrafo anterior foram aplicados a um algoritmo de Mapas Auto-Organizáveis (Mapas SOM), utilizando os seguintes parâmetros:',h)
        c, h = self.gerarSecao(c,'p','• Topologia: '+str(topology),h-28)
        c, h = self.gerarSecao(c,'p','• Distância de cluster: '+str(cluster_distance),h-28)
        c, h = self.gerarSecao(c,'p','• Épocas: '+str(epochs),h-28)
        c, h = self.gerarSecao(c,'p','• Tamanho do mapa: '+str(size),h-28)
        c, h = self.gerarSecao(c,'p','• Sigma: '+str(sigma),h-28)
        c, h = self.gerarSecao(c,'p','• Taxa de aprendizado: '+str(lr),h-28)
        h = h+30
        c.drawImage('required_files/cabecalho.jpeg', inch-8, page_h-50,page_w-inch-52,50)
        c.saveState()
        c.showPage()
        c.save()


        print("#"*80)
        print("Escrevendo", name)
        print(self.sigma, self.size, self.lr, self.epochs, self.cluster_distance, self.topology, self.output_influences)
        print("#"*80)

    def gerarSecao(self, c,tipo,paragrafo,h):
        page_w, page_h = letter
        if(tipo=='p'):
            style_paragrafo = ParagraphStyle("paragrafo", fontName="Helvetica", fontSize=12, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='t'):
            style_paragrafo = ParagraphStyle("titulo", fontName="Helvetica-Bold", fontSize=16, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='s'):
            style_paragrafo = ParagraphStyle("subtitulo", fontName="Helvetica-Bold", fontSize=14, alignment=4, leading=18, encoding="utf-8")
        message_paragrafo = Paragraph(paragrafo, style_paragrafo)
        w_paragrafo, h_paragrafo = message_paragrafo.wrap(page_w -2*inch, page_h)
        message_paragrafo.drawOn(c, inch, page_h - h- h_paragrafo)
        return c, h+h_paragrafo+30

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

        if avg is not None and std is not None:
            avg_heatmap_path = os.path.join('tempfiles', "heatmap_avg.png")
            std_heatmap_path = os.path.join('tempfiles', "heatmap_std.png")
            combined_image_path = os.path.join('tempfiles', "combined_heatmap.png")

            # Gera os heatmaps e salva na pasta tempfiles
            self.generate_heatmap(avg, avg_heatmap_path, "Média", 'YlOrRd')
            self.generate_heatmap(std, std_heatmap_path, "Desvio Padrão", 'gray')
            
            self.combine_images_horizontally(avg_heatmap_path, std_heatmap_path, combined_image_path)
            self.generate_pdf(name, combined_image_path)

            # Exclui as imagens após gerar o PDF
            if os.path.exists(avg_heatmap_path):
                os.remove(avg_heatmap_path)
            if os.path.exists(std_heatmap_path):
                os.remove(std_heatmap_path)

    def generate_heatmap(self, df, filename, title, cmap):
        # Garante que apenas colunas numéricas sejam usadas no heatmap
        df_numeric = df.select_dtypes(include=[float, int])

        # Configuração da figura
        fig, ax = plt.subplots(figsize=(16, 12))

        # Criação do heatmap diretamente com matplotlib
        cax = ax.matshow(df_numeric, cmap=cmap)

        # Configurações do gráfico
        fig.colorbar(cax)
        ax.set_title(title, pad=20)
        
        num_rows, num_cols = df_numeric.shape
        ax.set_xticks(range(num_cols))
        ax.set_yticks(range(num_rows))
        ax.set_xticklabels(range(1, num_cols + 1)) 
        ax.set_yticklabels(range(1, num_rows + 1))  

        # Ajusta o layout e salva a figura
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def combine_images_horizontally(self, img1_path, img2_path, output_path):
        # Abre as duas imagens
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Calcula a altura máxima e a largura combinada
        max_height = max(img1.height, img2.height)
        total_width = img1.width + img2.width

        # Cria uma nova imagem com o tamanho combinado
        combined_img = Image.new("RGB", (total_width, max_height))

        # Cola as imagens lado a lado
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))

        # Salva a imagem combinada
        combined_img.save(output_path)

    def generate_pdf(self, pdf_name, combined_image_path):
        # Define margens mínimas para maximizar o espaço útil
        doc = SimpleDocTemplate(pdf_name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []

        # Adiciona o cabeçalho no topo absoluto da página
        header_path = os.path.join("required_files", "cabecalho.jpeg")
        if os.path.exists(header_path):
            elements.append(ReportlabImage(header_path, width=500, height=100))  # Tamanho ajustado em pixels
            elements.append(Spacer(1, 12))

        # Adiciona a imagem combinada ao PDF
        combined_image = ReportlabImage(combined_image_path, width=500, height=700)  # Define o tamanho em pixels
        elements.append(combined_image)

        # Gera o PDF
        doc.build(elements)

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
        doc = SimpleDocTemplate(name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []
        styles = getSampleStyleSheet()

        # For each group, generate a page
        for i, (output_average, municipio_df, shap_df, image_path) in enumerate(list(zip(self.output_averages, self.municipio_dfs, self.shap_dfs, self.image_paths))):
            # Title
            title = f"Grupo {i+1}"
            if os.path.exists('required_files//cabecalho.jpeg'):
                elements.append(ReportlabImage('required_files//cabecalho.jpeg', width=500, height=50))
                elements.append(Spacer(1, 18))

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
        avg: pd.DataFrame = deepcopy(self.avg_df)
        std: pd.DataFrame = deepcopy(self.std_df)

        if avg is not None and std is not None:

            std_heatmap_path = os.path.join('tempfiles', "heatmap_filtro_std.png")
            avg_heatmap_path = os.path.join('tempfiles', "heatmap_filtro_avg.png")

            # Gera os heatmaps e salva na pasta tempfiles
            self.generate_heatmap(avg, avg_heatmap_path, "Média Filtrada", 'YlOrRd')
            self.generate_heatmap(std, std_heatmap_path, "Desvio Padrão Filtrado",'gray')
            combined_image_path = os.path.join('tempfiles', "combined_heatmap_filtro.png")

            # Gera o PDF com os DataFrames e heatmaps
            self.combine_images_horizontally(avg_heatmap_path, std_heatmap_path, combined_image_path)
            self.generate_pdf(name, combined_image_path)

            # Exclui as imagens após gerar o PDF
            if os.path.exists(avg_heatmap_path):
                os.remove(avg_heatmap_path)
            if os.path.exists(std_heatmap_path):
                os.remove(std_heatmap_path)

    def generate_heatmap(self, df, filename, title, cmap):
        # Garante que apenas colunas numéricas sejam usadas no heatmap
        df_numeric = df.select_dtypes(include=[float, int])

        # Configuração da figura
        fig, ax = plt.subplots(figsize=(16, 12))

        # Criação do heatmap diretamente com matplotlib
        cax = ax.matshow(df_numeric, cmap=cmap)        

        # Configurações do gráfico
        fig.colorbar(cax)
        ax.set_title(title, pad=20)

        num_rows, num_cols = df_numeric.shape
        ax.set_xticks(range(num_cols))
        ax.set_yticks(range(num_rows))
        ax.set_xticklabels(range(1, num_cols + 1)) 
        ax.set_yticklabels(range(1, num_rows + 1)) 

        # Ajusta o layout e salva a figura
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def combine_images_horizontally(self, img1_path, img2_path, output_path):
        # Abre as duas imagens
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Calcula a altura máxima e a largura combinada
        max_height = max(img1.height, img2.height)
        total_width = img1.width + img2.width

        # Cria uma nova imagem com o tamanho combinado
        combined_img = Image.new("RGB", (total_width, max_height))

        # Cola as imagens lado a lado
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))

        # Salva a imagem combinada
        combined_img.save(output_path)

    def generate_pdf(self, pdf_name, combined_image_path):
        # Define margens mínimas para maximizar o espaço útil
        doc = SimpleDocTemplate(pdf_name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []

        # Adiciona o cabeçalho no topo absoluto da página
        header_path = os.path.join("required_files", "cabecalho.jpeg")
        if os.path.exists(header_path):
            elements.append(ReportlabImage(header_path, width=500, height=100))  # Tamanho ajustado em pixels
            elements.append(Spacer(1, 12))

        # Adiciona a imagem combinada ao PDF
        combined_image = ReportlabImage(combined_image_path, width=500, height=700)  # Define o tamanho em pixels
        elements.append(combined_image)

        # Gera o PDF
        doc.build(elements)


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
        tabela_regioes.insert(0, 'Índice', tabela_regioes.index)

        # Constroi e gera o PDF
        page_w, page_h = letter
        texto1 = 'Essa seção traz uma tabela com todos os municípios de Pernambuco, identificando suas mesorregiões e microrregiões e dando um índice para elas, que é o índice utilizado nos Mapas de Calor.'
        c = canvas.Canvas(name)
        c, h = self.gerarSecao(c,'t','Seção 7 - Identificação de Meso e Microrregiões',65)
        c, h = self.gerarSecao(c,'p',texto1,h)
        c, h = self.gerarSecaoTabela(c,h,tabela_regioes.to_numpy())
        c, h = self.gerarLegenda(c,'Tabela 7 - Municípios e Suas Mesorregiões e Microrregiões', h+5)
        h = h+30
        c.drawImage('required_files/cabecalho.jpeg', inch-8, page_h-50,page_w-inch-52,50)
        c.saveState()
        c.showPage()
        c.save()

        # Remove coluna de índice do DataFrame
        tabela_regioes = tabela_regioes.drop(['Índice'], axis=1)

        print("#"*80)
        print("Escrevendo", name)
        print(tabela_regioes)
        print("#"*80)

    def gerarSecao(self,c,tipo,paragrafo,h):
        page_w, page_h = letter
        if(tipo=='p'):
            style_paragrafo = ParagraphStyle("paragrafo", fontName="Helvetica", fontSize=12, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='t'):
            style_paragrafo = ParagraphStyle("titulo", fontName="Helvetica-Bold", fontSize=16, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='s'):
            style_paragrafo = ParagraphStyle("subtitulo", fontName="Helvetica-Bold", fontSize=14, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='c'):
            style_paragrafo = ParagraphStyle("caption", fontName="Helvetica-Bold",backColor='#d3d3d3' , textColor='black', fontSize=18, alignment=TA_CENTER, leading=25, borderColor='gray', borderWidth=2, borderPadding=5, encoding="utf-8")
        message_paragrafo = Paragraph(paragrafo, style_paragrafo)
        w_paragrafo, h_paragrafo = message_paragrafo.wrap(page_w -2*inch, page_h)
        message_paragrafo.drawOn(c, inch, page_h - h- h_paragrafo)
        return (c, h+h_paragrafo+30) if tipo != 'c' else (c, h+h_paragrafo+12)
    
    def gerarLegenda(self,c,paragrafo,h):
        page_w, page_h = letter
        style_paragrafo = ParagraphStyle("paragrafo", fontName="Helvetica-Oblique", fontSize=10, alignment=4, leading=18, encoding="utf-8", textColor = 'blue')
        message_paragrafo = Paragraph(paragrafo, style_paragrafo)
        w_paragrafo, h_paragrafo = message_paragrafo.wrap(page_w -2*inch, page_h)
        message_paragrafo.drawOn(c, inch, page_h - h- h_paragrafo)
        return c, h+h_paragrafo+20

    def gerarTabela(self,data):
        data = [['Índice','Nome Município','Mesorregião', 'Microrregião']]+data

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 'WORD'),
        ])

        col_widths = [40,140,140,150]

        table = Table(data, colWidths=col_widths)
        table.setStyle(style)
        return table

    def gerarTabelaPdf(self,c,data,h,start):
        page_w, page_h = letter
        if(len(data)>start):
            data2 = []
            end = 0
            for i in range(len(data)-start+1):
                table = self.gerarTabela(data2)
                w_paragrafo, h_paragrafo = table.wrapOn(c, 0, 0)
                if(page_h - h- h_paragrafo< inch):
                    end = i
                    break

                if(i<len(data)-start):
                    data2+= [data[i+start]]
            table.drawOn(c, inch, page_h - h- h_paragrafo)
            return c, h_paragrafo+h, start+end
        else:
            return c, h, start

    def quebraPagina(self,c, h, tamanho):
        page_w, page_h = letter
        if(h>tamanho):
            c.drawImage('required_files/cabecalho.jpeg', inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            h=65
        return c, h

    def gerarSecaoTabela(self,c,h,dados):
        start = 0
        start2 = 0
        while(True):
            c, h, start = self.gerarTabelaPdf(c,dados,h,start)
            if(start==start2):
                break
            else:
                c, h = self.quebraPagina(c, h, 0)
                start2=start
        return c, h

class S3P1_RelatorioIndividual():
    def __init__(self):
        self.municipios = {}
    
    def write_page(self, municipio: str, name: str="relatorio_individual.pdf"):
        if municipio in self.municipios.keys():
            data = self.municipios[municipio]

            shap_columns  = data['dados']['Fator'].tolist()
            shape_results = pd.DataFrame(data={'data': data['dados']['Valor'].tolist(), 'values': data['dados']['Influência'].tolist()})

            numero_atributos = 3
            values = shape_results['values']
            maiores_valores = [v for v in sorted(values, reverse=True)[:numero_atributos] if v > 0]
            menores_valores = [v for v in sorted(values)[:numero_atributos] if v < 0]
            indices_maiores_valores = {}
            indices_menores_valores = {}
            for id, v in enumerate(values):
                if v in maiores_valores and id not in indices_maiores_valores:
                    indices_maiores_valores[id] = v
                elif v in menores_valores and id not in indices_menores_valores:
                    indices_menores_valores[id] = v
            indices_maiores_valores = dict(sorted(indices_maiores_valores.items(), key=lambda item: item[1]))
            indices_menores_valores = dict(sorted(indices_menores_valores.items(), key=lambda item: item[1], reverse=True))

            # Gera os DF das tabelas
            df1, df2, df3, df4 = self.gerarDataFrames(shap_columns, shape_results, indices_maiores_valores, indices_menores_valores, data)

            # Ajusta algumas colunas do DF (arg2), definindo um número máximo de caracteres (arg3) em uma linha 
            df1 = self.ajustarDataFrames(df1, ['Fatores'], 38)
            df1 = self.ajustarDataFrames(df1, ['Valor'], 6)
            df2 = self.ajustarDataFrames(df2, ['Positivamente', 'Negativamente'], 24)

            # Constroi e gera o PDF
            page_w, page_h = letter
            c = canvas.Canvas(name)

            c, h = self.gerarSecao(c,'t1',municipio,65)
            c, h = self.gerarSecao(c,'c',f"Resultado da Influência dos Fatores no(a) {data['output']}",h+30)
            c, h = self.gerarSecaoTabela(c,h,df1.to_numpy(), df1.columns.tolist(),1)
            c, h = self.gerarLegenda(c,f"Tabela 1 - Impacto dos Fatores na Taxa de {data['output']}", h+5)
            posicao, h_last = 61 + df1.shape[0] * 14, h # Calcula +/- a posição da tabela no pdf
            c, h = self.quebraPagina(c, posicao, h, 98) # Tenta evitar que a tabela 2 seja quebrada ao meio
            pagina_quebrada = True if h < h_last else False

            c, h = self.gerarSecao(c,'c',"Fatores que Mais Influenciaram",h+30)
            c, h = self.gerarSecaoTabela(c,h,df2.to_numpy(), df2.columns.tolist(),2)
            c, h = self.gerarLegenda(c,"Tabela 2 - Principais Fatores de Influência", h+5)
            posicao, h_last = (posicao + 14 + df2.shape[0] * 14, h) if not pagina_quebrada else (34 + df2.shape[0] * 14, h)
            c, h = self.quebraPagina(c, posicao, h, 165) # Tenta evitar que a tabela 3 seja quebrada ao meio
            pagina_quebrada = True if h < h_last else False

            c, h = self.gerarSecao(c,'c',f"{data['output']}",h+30)
            c, h = self.gerarSecaoTabela(c,h,df3.to_numpy(), df3.columns.tolist(),3)
            c, h = self.gerarLegenda(c,f"Tabela 3 - Comparação do(a) {data['output']} entre o Município e o seu Grupo", h+5)
            posicao, h_last = (posicao + 14 + df3.shape[0] * 14, h) if not pagina_quebrada else (34 + df3.shape[0] * 14, h)
            c, h = self.quebraPagina(c, posicao, h, 103) # Tenta evitar que a tabela 4 seja quebrada ao meio

            c, h = self.gerarSecao(c,'c',"Vizinhos Mais Próximos",h+30)
            c, h = self.gerarSecaoTabela(c,h,df4.to_numpy(), df4.columns.tolist(),4)
            c, h = self.gerarSecao(c,'c1',"OBS: A PROXIMIDADE ENVOLVE O CONJUNTO TOTAL DOS FATORES E SUAS SEMELHANÇAS, AO INVÉS DE QUESTÕES GEOGRÁFICAS.",h+8)
            c, h = self.gerarLegenda(c,f"Tabela 4 - Municípios Mais Semelhantes a {municipio}", h+5)
            h = h+30

            c.drawImage('required_files/cabecalho.jpeg', inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            c.save()

            print(municipio)
            print("#"*80)
            print("Escrevendo", name)
            print(data)
            print("#"*80)

    def gerarSecao(self,c,tipo,paragrafo,h):
        page_w, page_h = letter
        if(tipo=='p'):
            style_paragrafo = ParagraphStyle("paragrafo", fontName="Helvetica-Bold", fontSize=12, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='t'):
            style_paragrafo = ParagraphStyle("titulo", fontName="Helvetica-Bold", fontSize=16, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='t1'):
            style_paragrafo = ParagraphStyle("titulo1", fontName="Times-Bold", fontSize=35, alignment=TA_CENTER, leading=18, encoding="utf-8")
        elif(tipo=='s'):
            style_paragrafo = ParagraphStyle("subtitulo", fontName="Helvetica-Bold", fontSize=14, alignment=4, leading=18, encoding="utf-8")
        elif(tipo=='c'):
            style_paragrafo = ParagraphStyle("caption-up", fontName="Times-Bold" , leftIndent=5, rightIndent=1, textColor='#8B4513', fontSize=21, alignment=TA_CENTER, leading=25, borderColor='gray', borderWidth=2, borderPadding=5, encoding="utf-8")
        elif(tipo=='c1'):
            style_paragrafo = ParagraphStyle("caption-down", fontName="Times-Bold" , leftIndent=5, rightIndent=1, textColor='black', fontSize=12, alignment=TA_CENTER, leading=25, borderColor='gray', borderWidth=2, borderPadding=5, encoding="utf-8")

        message_paragrafo = Paragraph(paragrafo, style_paragrafo)
        w_paragrafo, h_paragrafo = message_paragrafo.wrap(page_w -2*inch, page_h)
        message_paragrafo.drawOn(c, inch, page_h - h- h_paragrafo)
        return (c, h+h_paragrafo+30) if tipo != 'c' and tipo != 'c1' else (c, h+h_paragrafo+8)

    def gerarLegenda(self,c,paragrafo,h):
        page_w, page_h = letter
        style_paragrafo = ParagraphStyle("paragrafo", fontName="Helvetica-Oblique", fontSize=13, alignment=4, leading=18, encoding="utf-8", textColor = 'blue')
        message_paragrafo = Paragraph(paragrafo, style_paragrafo)
        w_paragrafo, h_paragrafo = message_paragrafo.wrap(page_w -2*inch, page_h)
        message_paragrafo.drawOn(c, inch, page_h - h- h_paragrafo)
        return c, h+h_paragrafo+20

    def gerarTabela(self,data, columns, j):
        data = [columns]+data

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 16),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 'WORD'),
        ])

        if  (j==1): col_widths = [52,297,50,75]
        elif(j==2): col_widths = [189,48,189,48]
        elif(j==3): col_widths = [237,237]
        elif(j==4): col_widths = [474]

        table = Table(data, colWidths=col_widths)
        table.setStyle(style)
        return table

    def gerarTabelaPdf(self,c,data,h,start,columns,j):
        page_w, page_h = letter
        if(len(data)>start):
            data2 = []
            end = 0
            for i in range(len(data)-start+1):
                table = self.gerarTabela(data2,columns,j)
                w_paragrafo, h_paragrafo = table.wrapOn(c, 0, 0)
                if(page_h - h- h_paragrafo< inch):
                    end = i
                    break

                if(i<len(data)-start):
                    data2+= [data[i+start]]
            table.drawOn(c, inch, page_h - h- h_paragrafo)
            return c, h_paragrafo+h, start+end
        else:
            return c, h, start

    def quebraPagina(self,c, posicao, h, tamanho):
        page_w, page_h = letter
        if(posicao % 200 > tamanho):
            c.drawImage('required_files/cabecalho.jpeg', inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            h=65
        return c, h

    def gerarSecaoTabela(self,c,h,dados,columns,j):
        start = 0
        start2 = 0
        while(True):
            c, h, start = self.gerarTabelaPdf(c,dados,h,start,columns,j)
            if(start==start2):
                break
            else:
                c, h = self.quebraPagina(c, h, h, 0)
                start2=start
        return c, h

    def insert_newlines(self,text, every=40):
        lines = []
        while len(text) > every:
            split_index = text.rfind(' ', 0, every)
            if split_index == -1:
                split_index = every
            lines.append(text[:split_index].strip())
            text = text[split_index:].strip()
        lines.append(text)
        return '\n\n'.join(lines)

    def dividirlinhas(self,data, every):
        if(len(data)>every):
            data = self.insert_newlines(data, every=every)
        return data

    def ajustarDataFrames(self,df, colunas, numCar):
        for coluna in colunas:
            df[coluna] = [self.dividirlinhas(str(linha), numCar) for linha in df[coluna]]
        return df

    def gerarDataFrames(self, shap_columns, shape_results, indices_maiores_valores, indices_menores_valores, data):
        df1 = pd.DataFrame(data={'Índice'       : range(1, len(shap_columns) + 1),
                                 'Fatores'      : shap_columns,
                                 'Valor'        : [f"{x:.2f}" for x in shape_results['data']],
                                 'Influência'   : [f"{x:.3f}" for x in shape_results['values']]})
        
        df2 = pd.DataFrame(data={'Positivamente': list(reversed([shap_columns[chave] for chave, valor in indices_maiores_valores.items()])) + ['---'] * (3 - len(indices_maiores_valores)),
                                 '+'            : list(reversed([f"{valor:.3f}" for chave, valor in indices_maiores_valores.items()])) + ['---'] * (3 - len(indices_maiores_valores)),
                                 'Negativamente': list(reversed([shap_columns[chave] for chave, valor in indices_menores_valores.items()])) + ['---'] * (3 - len(indices_menores_valores)),
                                 '-'            : list(reversed([f"{valor:.3f}" for chave, valor in indices_menores_valores.items()])) + ['---'] * (3 - len(indices_menores_valores))})
        
        df3 = pd.DataFrame(data={f"Grupo {data['grupo']}": [f"{data['nota_media_grupo']*100:.2f} %"],
                                  'Município'                : [f"{data['nota_individual']*100:.2f} %"]})

        df4 = pd.DataFrame(data={'Nome': [label for label in data['vizinhos']] if len(data['vizinhos']) > 0 else ['---']})
        
        return df1, df2, df3, df4
    