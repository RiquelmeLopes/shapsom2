import pandas as pd
import streamlit as st
from copy import deepcopy
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image as ReportlabImage
from reportlab import platypus
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.platypus import PageBreak
import os
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter

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

        if os.path.exists(os.path.join('required_files', 'cabecalho.jpeg')):
            elements.append(ReportlabImage(os.path.join('required_files', 'cabecalho.jpeg'), width=500, height=50))
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
        if os.path.exists(os.path.join('required_files', 'cabecalho.jpeg')):
            elements.append(ReportlabImage(os.path.join('required_files', 'cabecalho.jpeg'), width=500, height=50))
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

        if os.path.exists(os.path.join('required_files', 'cabecalho.jpeg')):
            elements.append(ReportlabImage(os.path.join('required_files', 'cabecalho.jpeg'), width=500, height=50))
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
            # Filtra apenas os valores numéricos
            avg = avg.select_dtypes(include=[float, int])
            std = std.select_dtypes(include=[float, int])

            # Diretório para salvar os heatmaps
            
            avg_heatmap_paths = self.generate_heatmap_fragments(avg, "Média", "YlOrRd", 'tempfiles')
            std_heatmap_paths = self.generate_heatmap_fragments(std, "Desvio Padrão", "gray", 'tempfiles')

            combined_image_paths = self.combine_images(avg_heatmap_paths, std_heatmap_paths, 'tempfiles')

            # Gera o PDF com as imagens combinadas e textos
            self.generate_pdf(name, combined_image_paths)

    def generate_heatmap_fragments(self, df, title, cmap, output_dir):
        heatmap_paths = []
        row_blocks = [df.iloc[i:i + 63, :] for i in range(0, len(df), 63)]

        for index, block in enumerate(row_blocks):
            height = 4 * (len(block) / 63)  # Ajusta a altura de acordo com as linhas (base 4in para 63 linhas)
            max_cols = 30
            width = 3 * min(len(block.columns), max_cols) / max_cols  # Ajusta a largura com base nas colunas

            fig, ax = plt.subplots(figsize=(width, height))  # Configura a largura e altura dinamicamente
            cax = ax.matshow(block, cmap=cmap)

            # Adiciona a barra de cores em cima
            cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1)
            cbar.ax.tick_params(labelsize=8)

            # Adiciona o título acima da barra de cores
            ax.set_title(f"{title} - Parte {index + 1}", pad=20, fontsize=5)

            # Remove as legendas dos eixos
            ax.set_xticks(range(len(block.columns)))
            ax.set_xticklabels(range(1, len(block.columns) + 1), fontsize=3)
            ax.set_yticks(range(len(block)))
            ax.set_yticklabels(range(1, len(block) + 1), fontsize=3)

            # Salva o fragmento do heatmap
            heatmap_path = os.path.join(output_dir, f"heatmap_{title.replace(' ', '_').lower()}_{index + 1}.png")
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300)
            plt.close()

            heatmap_paths.append(heatmap_path)

        return heatmap_paths

    def combine_images(self, avg_heatmap_paths, std_heatmap_paths, output_dir):
        combined_image_paths = []

        for i in range(len(avg_heatmap_paths)):
            # Abre as duas imagens
            avg_img = Image.open(avg_heatmap_paths[i])
            std_img = Image.open(std_heatmap_paths[i])

            # Calcula a altura máxima e a largura combinada
            max_height = max(avg_img.height, std_img.height)
            total_width = avg_img.width + std_img.width

            # Cria uma nova imagem com o tamanho combinado
            combined_img = Image.new("RGB", (total_width, max_height))

            # Cola as imagens lado a lado
            combined_img.paste(avg_img, (0, 0))
            combined_img.paste(std_img, (avg_img.width, 0))

            # Salva a imagem combinada
            combined_image_path = os.path.join(output_dir, f"combined_heatmap_{i + 1}.png")
            combined_img.save(combined_image_path)

            combined_image_paths.append(combined_image_path)

        return combined_image_paths

    def generate_pdf(self, pdf_name, combined_image_paths: list):
        doc = SimpleDocTemplate(pdf_name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []

        # Adiciona o cabeçalho no topo absoluto da página
        header_path = os.path.join("required_files", "cabecalho.jpeg")
        if os.path.exists(header_path):
            elements.append(ReportlabImage(header_path, width=500, height=100))
            elements.append(Spacer(1, 12))

        # Estilos para o texto
        styles = getSampleStyleSheet()
        style_title = ParagraphStyle(
            'Title',
            fontSize=14,
            leading=16,
            fontName="Helvetica-Bold",
            spaceAfter=14
        )
        style_normal = styles["Normal"]

        style_caption = ParagraphStyle(
            'Caption',
            fontSize=10,
            fontName="Helvetica-Oblique",
            textColor=colors.blue
        )

        explanation_text1 = """
        Esta seção traz uma análise visual da base de dados, fornecendo mapas de calor para
        a média (Gráfico 1) e desvio padrão (Gráfico 2) dos fatores disponibilizados para cada
        um dos municípios.\n"""
        explanation_text2 = """
        Mapa de Calor, também conhecido como Heatmap, é uma visualização gráfica que usa
        cores para representar a intensidade dos valores em uma matriz de dados. Cada célula
        da matriz é colorida de acordo com seu valor, facilitando a identificação de padrões,
        tendências e anomalias nos dados.\n
        """
        explanation_text3 = """
        <b>Média:</b>\n
        - É a soma de todos os valores de um conjunto dividida pelo número de valores.
        Representa o valor médio.\n
        <b>Desvio padrão:</b>\n
        - Mede a dispersão dos valores em relação à média. Mostra o quanto os valores
        variam da média.\n
        """
        explanation_text4 = """
        <b>Importante:</b>\n
        Nos gráficos referentes aos Mapas de Calor:
        As linhas representam os municípios, que estão em ordem alfabética;
        As colunas representam os fatores selecionados pelo usuário na base de dados;
        """

        elements.append(Paragraph("Seção 2 - Visão dos Dados e Gráficos de Mapas de Calor", style_title))
        elements.append(Paragraph(explanation_text1, style_normal))
        elements.append(Paragraph(explanation_text2, style_normal))
        elements.append(Paragraph(explanation_text3, style_normal))
        elements.append(Paragraph(explanation_text4, style_normal))
        elements.append(Spacer(1, 12))

        def pairwise(lst: list) -> list:
            pairs = [(lst[i], lst[i + 1]) for i in range(0, len(lst) - 1, 2)]
            if len(lst) % 2 != 0:
                pairs.append((lst[-1], None))
            return pairs

        if combined_image_paths:
            # Adiciona a primeira imagem e quebra a página
            elements.append(ReportlabImage(combined_image_paths[0], width=500, height=300))
            elements.append(Paragraph(f"<i>Figura 2.1 - Heatmap média e desvio padrão parte 1</i>", style_caption))
            elements.append(PageBreak())

            for idx, (i1, i2) in enumerate(pairwise(combined_image_paths[1:])):
                # Adiciona o restante das imagens duas por folha
                fig_idx = (idx+1)*2
                if i1:
                    elements.append(ReportlabImage(i1, width=500, height=250))
                    elements.append(Paragraph(f"<i>Figura 2.{fig_idx} - Heatmap média e desvio padrão parte {fig_idx}</i>", style_caption))
                    elements.append(Spacer(1, 12))
                if i2:
                    elements.append(ReportlabImage(i2, width=500, height=250))
                    elements.append(Paragraph(f"<i>Figura 2.{fig_idx+1} - Heatmap média e desvio padrão parte {fig_idx+1}</i>", style_caption))
                    elements.append(PageBreak())

        # Gera o PDF
        doc.build(elements)

        # Exclui as imagens combinadas
        for path in combined_image_paths:
            if os.path.exists(path):
                os.remove(path)

class S2P4_Shap():
    def __init__(self):
        self.df = None
        self.on_report = False
        self.finished_selection = False

    def write_page(self, name: str="shap.pdf"):
        doc = SimpleDocTemplate(name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []
        styles = getSampleStyleSheet()
        #title_style = ParagraphStyle(name='CenterTitle', alignment=TA_CENTER, fontSize=14, spaceAfter=20)
        title_style = ParagraphStyle(name='CenterTitle', fontName="Helvetica-Bold", fontSize=16, alignment=4, leading=16, encoding="utf-8")
        paragraph_style = ParagraphStyle(name='CenterParagraph',fontName="Helvetica", fontSize=12, alignment=4, leading=16, encoding="utf-8", leftIndent=0, rightIndent=0)
        table_caption_style = ParagraphStyle(name="Subtitulo", fontName="Helvetica-Oblique", fontSize=10, alignment=4, leading=18, encoding="utf-8", textColor = 'blue')

        # Add the initial text
        intro_text = '''
        Nesta seção, apresentamos os grupos identificados e as variáveis que mais influenciaram na formação desses grupos. Um "agrupamento" reúne dados que são mais semelhantes em termos de suas características globais. Esses grupos são utilizados na aplicação de IA através de bases de dados (tabelas) fornecidas pela área usuária para o processamento com Redes Neurais Artificiais. "Agrupamento" é o processo de reunir, por exemplo, municípios, com base em suas semelhanças, visando realizar triagens para guiar auditorias.
        '''
        elements.append(Paragraph("Seção 3.1 - Análise de agrupamentos com SHAP", title_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(intro_text, paragraph_style))
        elements.append(Spacer(1, 12))

        shap_df = deepcopy(st.session_state["shap"].df)

        # Largura total da tabela
        total_width = 480
        first_col_width = total_width * 0.50
        remaining_width = total_width - first_col_width


        # Divide os dados em subgrupos de no máximo 7 colunas
        columns = shap_df.columns.tolist()
        num_cols = len(columns)

        i = 0
        for start_col in range(1, num_cols, 6):
            i += 1
            end_col = min(start_col + 6, num_cols)
            sub_columns = [columns[0]] + columns[start_col:end_col]
            
            sub_table_data = [[str(row[0])] + [str(val) for val in row[start_col:end_col]] for row in shap_df.values]
            sub_table_data.insert(0, sub_columns)
            
            col_widths = [first_col_width] + [remaining_width / (len(sub_columns) - 1)] * (len(sub_columns) - 1)

            for row_idx in range(1, len(sub_table_data)):
                if (len(sub_table_data[row_idx][0]) > 45):
                    sub_table_data[row_idx][0] = sub_table_data[row_idx][0][:45] +'...'

                for col_idx in range(1, len(sub_table_data[row_idx])):
                    sub_table_data[row_idx][col_idx] = f"{float(sub_table_data[row_idx][col_idx]):.3f}"


            table = Table(sub_table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('WORDWRAP', (0, 0), (-1, -1), 'WORD'),
            ]))
            
            for row_idx in range(1, len(sub_table_data)):
                for col_idx in range(1, len(sub_table_data[row_idx])):
                    value = float(sub_table_data[row_idx][col_idx])
                    if value > 0:
                        table.setStyle(TableStyle([('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), colors.blue)]))
                    elif value < 0:
                        table.setStyle(TableStyle([('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), colors.red)]))
            
            elements.append(table)
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"Tabela 3.{i} - Resultados do SHAP", table_caption_style))
            elements.append(Spacer(1, 24))

        # Build PDF
        doc.build(elements)

class S2P5_Arvore():
    def __init__(self):
        self.img_path = ""
        self.fig = None
        self.on_report = False
        self.finished_selection = False
    
    def write_page(self, name: str="arvore.pdf"):
        doc = SimpleDocTemplate(name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []
        styles = getSampleStyleSheet()

        # Estilos personalizados
        title_style = ParagraphStyle(name='CenterTitle', fontName="Helvetica-Bold", fontSize=16, alignment=4, leading=16, encoding="utf-8")
        paragraph_style = ParagraphStyle(name='CenterParagraph', fontName="Helvetica", fontSize=12, alignment=4, leading=16, leftIndent=0, rightIndent=0, encoding="utf-8")
        table_caption_style = ParagraphStyle(name="Subtitulo", fontName="Helvetica-Oblique", fontSize=10, alignment=4, leading=16, encoding="utf-8", textColor='blue')

        # Adiciona o título e introdução
        intro_text = '''
        Nesta seção, apresentamos a análise de agrupamentos utilizando um modelo de árvore de decisão. A importância de uma variável indica quanto ela contribui para a decisão final do modelo. Valores mais altos de importância sugerem que a variável tem um impacto maior na previsão do modelo.
        '''
        elements.append(Paragraph("Seção 3.2 - Análise de agrupamentos com Árvore de Decisão", title_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(intro_text, paragraph_style))
        elements.append(Spacer(1, 12))

        # Recupera o DataFrame com a importância das variáveis
        feature_importances = deepcopy(st.session_state["arvore"].feature_importances)
        
        data = [feature_importances.columns.tolist()] + feature_importances.values.tolist()
        # largura total da tabela
        total_width = 480

        # Define a largura da primeira coluna como 60% do total
        first_col_width = total_width * 0.80

        # Calcula a largura das colunas restantes
        num_cols = len(data[0])
        remaining_width = total_width - first_col_width
        remaining_col_width = remaining_width / (num_cols - 1)  # Distribui o restante igualmente

        # Define as larguras das colunas
        col_widths = [first_col_width] + [remaining_col_width] * (num_cols - 1)

        for row_idx in range(1, len(data)):
            if (len(data[row_idx][0]) > 80):
                data[row_idx][0] = data[row_idx][0][:80] +'...'

            for col_idx in range(1, len(data[row_idx])):
                data[row_idx][col_idx] = float(f"{data[row_idx][col_idx]:.5f}")

        
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 'WORD'),
        ]))

        # Deixar azul, se for positivo (sempre é positivo)
        # for row_idx in range(1, len(data)):
        #     for col_idx in range(1, len(data[row_idx])):
        #         value = data[row_idx][col_idx]
        #         if value > 0:
        #             table.setStyle(TableStyle([('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), colors.blue)]))
        #         elif value < 0:
        #             table.setStyle(TableStyle([('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), colors.red)]))
        
        elements.append(table)
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Tabela 3.2.1 - Importância das Variáveis no Modelo de Árvore de Decisão", table_caption_style))
        elements.append(Spacer(1, 24))
        
        # Adiciona a imagem da árvore de decisão
        if st.session_state["arvore"].img_path:
            img_path = st.session_state["arvore"].img_path
            elements.append(ReportlabImage(img_path, width=480, height=480))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Figura 3.2.1 - Árvore de Decisão", table_caption_style))

        # Constrói o PDF
        doc.build(elements)

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
            if os.path.exists(os.path.join('required_files', 'cabecalho.jpeg')):
                elements.append(ReportlabImage(os.path.join('required_files', 'cabecalho.jpeg'), width=500, height=50))
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
            # Filtra apenas os valores numéricos
            avg = avg.select_dtypes(include=[float, int])
            std = std.select_dtypes(include=[float, int])

            # Diretório para salvar os heatmaps
            temp_dir = "tempfiles"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            avg_heatmap_paths = self.generate_heatmap_fragments(avg, "Média Filtrada", "YlOrRd", temp_dir)
            std_heatmap_paths = self.generate_heatmap_fragments(std, "Desvio Padrão Filtrado", "gray", temp_dir)

            combined_image_paths = self.combine_images(avg_heatmap_paths, std_heatmap_paths, temp_dir)

            # Gera o PDF com as imagens combinadas e textos
            self.generate_pdf(name, combined_image_paths)

    def generate_heatmap_fragments(self, df, title, cmap, output_dir):
        heatmap_paths = []
        row_blocks = [df.iloc[i:i + 63, :] for i in range(0, len(df), 63)]

        for index, block in enumerate(row_blocks):
            height = 4 * (len(block) / 63)  # Ajusta a altura de acordo com as linhas (base 4in para 63 linhas)
            max_cols = 30
            width = 3 * min(len(block.columns), max_cols) / max_cols  # Ajusta a largura com base nas colunas

            fig, ax = plt.subplots(figsize=(width, height))  # Configura a largura e altura dinamicamente
            cax = ax.matshow(block, cmap=cmap)

            # Adiciona a barra de cores em cima
            cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1)
            cbar.ax.tick_params(labelsize=8)

            # Adiciona o título acima da barra de cores
            ax.set_title(f"{title} - Parte {index + 1}", pad=20, fontsize=5)

            # Adiciona as legendas dos eixos como números de 1 a n com tamanho de fonte reduzido
            ax.set_xticks(range(len(block.columns)))
            ax.set_xticklabels(range(1, len(block.columns) + 1), fontsize=3)
            ax.set_yticks(range(len(block)))
            ax.set_yticklabels(range(1, len(block) + 1), fontsize=3)

            # Salva o fragmento do heatmap
            heatmap_path = os.path.join(output_dir, f"heatmap_{title.replace(' ', '_').lower()}_{index + 1}.png")
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300)
            plt.close()

            heatmap_paths.append(heatmap_path)

        return heatmap_paths

    def combine_images(self, avg_heatmap_paths, std_heatmap_paths, output_dir):
        combined_image_paths = []

        for i in range(len(avg_heatmap_paths)):
            # Abre as duas imagens
            avg_img = Image.open(avg_heatmap_paths[i])
            std_img = Image.open(std_heatmap_paths[i])

            # Calcula a altura máxima e a largura combinada
            max_height = max(avg_img.height, std_img.height)
            total_width = avg_img.width + std_img.width

            # Cria uma nova imagem com o tamanho combinado
            combined_img = Image.new("RGB", (total_width, max_height))

            # Cola as imagens lado a lado
            combined_img.paste(avg_img, (0, 0))
            combined_img.paste(std_img, (avg_img.width, 0))

            # Salva a imagem combinada
            combined_image_path = os.path.join(output_dir, f"combined_heatmap_{i + 1}.png")
            combined_img.save(combined_image_path)

            combined_image_paths.append(combined_image_path)

        return combined_image_paths

    def generate_pdf(self, pdf_name, combined_image_paths):
        doc = SimpleDocTemplate(pdf_name, pagesize=A4, topMargin=20, bottomMargin=20)
        elements = []

        # Adiciona o cabeçalho no topo absoluto da página
        header_path = os.path.join("required_files", "cabecalho.jpeg")
        if os.path.exists(header_path):
            elements.append(ReportlabImage(header_path, width=500, height=100))
            elements.append(Spacer(1, 12))

        # Estilos para o texto
        styles = getSampleStyleSheet()
        style_title = ParagraphStyle(
            'Title',
            fontSize=14,
            leading=16,
            fontName="Helvetica-Bold",
            spaceAfter=14
        )

        style_caption = ParagraphStyle(
            'Caption',
            fontSize=10,
            fontName="Helvetica-Oblique",
            textColor=colors.blue
        )
        style_normal = styles["Normal"]

        explanation_text = """
        <b>Esta seção, assim como na seção 2, traz uma análise visual da base de dados, porém agora em 
        uma fatia dos dados escolida pelo usuário. Essa visualização é útil para analizar de forma mais detalhada elementos de 
        interesse da base de dados.</b>\n
        """

        elements.append(Paragraph("Seção 5 - Filtro de Triagem", style_title))
        elements.append(Paragraph(explanation_text, style_normal))
        elements.append(Spacer(1, 12))

        def pairwise(lst: list) -> list:
            pairs = [(lst[i], lst[i + 1]) for i in range(0, len(lst) - 1, 2)]
            if len(lst) % 2 != 0:
                pairs.append((lst[-1], None))
            return pairs

        if combined_image_paths:
            # Adiciona a primeira imagem e quebra a página
            elements.append(ReportlabImage(combined_image_paths[0], width=500, height=300))
            elements.append(Paragraph(f"<i>Figura 5.1 - Heatmap média e desvio padrão parte 1</i>", style_caption))
            elements.append(PageBreak())

            for idx, (i1, i2) in enumerate(pairwise(combined_image_paths[1:])):
                # Adiciona o restante das imagens duas por folha
                fig_idx = (idx+1)*2
                if i1:
                    elements.append(ReportlabImage(i1, width=500, height=250))
                    elements.append(Paragraph(f"<i>Figura 5.{fig_idx} - Heatmap média e desvio padrão parte {fig_idx}</i>", style_caption))
                    elements.append(Spacer(1, 12))
                if i2:
                    elements.append(ReportlabImage(i2, width=500, height=250))
                    elements.append(Paragraph(f"<i>Figura 5.{fig_idx+1} - Heatmap média e desvio padrão parte {fig_idx+1}</i>", style_caption))
                    elements.append(PageBreak())

        # Gera o PDF
        doc.build(elements)

        # Exclui as imagens combinadas
        for path in combined_image_paths:
            if os.path.exists(path):
                os.remove(path)


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

        # Adiciona coluna de índice no dataframe
        anomalias_df = anomalias_df.sort_index()
        anomalias_df.insert(0, ' ', anomalias_df.index)

        # Faz alguns ajustes do dataframe
        anomalias_df['Distância do Centroide'] = [f'{x:.3f}' for x in anomalias_df['Distância do Centroide']]
        anomalias_df[anomalias_df.columns.tolist()[6]] = [f'{x:.3f}' for x in anomalias_df[anomalias_df.columns.tolist()[6]]]
        anomalias_df['Fator mais influente'] = ['...'.join(x.rsplit(x[39:-8] or '...', 1)) for x in anomalias_df['Fator mais influente']]
        anomalias_df['Fator menos influente'] = ['...'.join(x.rsplit(x[39:-8] or '...', 1)) for x in anomalias_df['Fator menos influente']]
        self.ajustarDataFrames(anomalias_df, ['Municípios'], 13)
        self.ajustarDataFrames(anomalias_df, ['Fator mais influente', 'Fator menos influente'], 20)
        anomalias_df.columns = [' ', 'Municípios', 'X', 'Y', 'Grupo', self.dividirlinhas('Distância Centroide', 10),
                                'Fator', self.dividirlinhas('Fator mais influente', 28), self.dividirlinhas('Fator menos influente', 28)]

        # Constroi e gera o PDF
        page_w, page_h = letter
        texto1 = 'A análise de anomalias foi conduzida utilizando um Mapa Auto-Organizável (SOM) para identificar pontos de dados que se desviam significativamente do padrão observado. Com as coordenadas dos pontos no SOM, o centroide do mapa foi calculado. Este centroide é determinado utilizando a mediana das coordenadas x e y de todos os pontos, o que fornece uma medida menos sensível a outliers em comparação com a média. Então, são calculadas as distâncias dos pontos para o centroide do mapa. Pontos que apresentaram distâncias significativamente maiores em relação ao centroide foram identificados como anômalos. Estes pontos fora do cluster principal sugerem comportamentos ou características discrepantes dos dados normais, destacando-se por estarem afastados do padrão usual.'
        c = canvas.Canvas(name)
        c, h = self.gerarSecao(c,'t','Seção 6 - Anomalias',65)
        c, h = self.gerarSecao(c,'p',texto1,h)
        c, h = self.gerarSecaoTabela(c,h,anomalias_df.to_numpy(), anomalias_df.columns.tolist())
        c, h = self.gerarLegenda(c,'Tabela 6 - Tabela de anomalias', h+5)
        h = h+30
        c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
        c.saveState()
        c.showPage()
        c.save()

        # Remove coluna de índice do DataFrame
        anomalias_df = anomalias_df.drop([' '], axis=1)

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

    def gerarTabela(self,data,columns):
        data = [columns] + data

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 'CHAR'),
        ])

        col_widths = [26,70,28,28,32,48,48,100,105]

        table = Table(data, colWidths=col_widths)
        table.setStyle(style)
        return table

    def gerarTabelaPdf(self,c,data,h,start,columns):
        page_w, page_h = letter
        if(len(data)>start):
            data2 = []
            end = 0
            for i in range(len(data)-start+1):
                table = self.gerarTabela(data2,columns)
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
            c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            h=65
        return c, h

    def gerarSecaoTabela(self,c,h,dados,columns):
        start = 0
        start2 = 0
        while(True):
            c, h, start = self.gerarTabelaPdf(c,dados,h,start,columns)
            if(start==start2):
                break
            else:
                c, h = self.quebraPagina(c, h, 0)
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
        c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
        c.saveState()
        c.showPage()
        c.save()

        # Remove coluna de índice do DataFrame
        tabela_regioes = tabela_regioes.drop(['Índice'], axis=1)

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
        data = [['Índice','Nome Município','Mesorregião', 'Microrregião']] + data

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
            c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
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
            pdf1 = name.replace(".pdf", "1.pdf")
            c = canvas.Canvas(pdf1)
            c, h = self.gerarSecao(c,'t1',municipio,65)
            c, h = self.gerarSecao(c,'c',f"Resultado da Influência dos Fatores no(a) {data['output']}",h+30)
            c, h = self.gerarSecaoTabela(c,h,df1.to_numpy(), df1.columns.tolist(),1)
            c, h = self.gerarLegenda(c,f"Tabela 1 - Impacto dos Fatores na Taxa de {data['output']}", h+5)
            c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            c.save()

            page_w, page_h = letter
            pdf2 = name.replace(".pdf", "2.pdf")
            c = canvas.Canvas(pdf2)
            c, h = self.gerarSecao(c,'c',"Fatores que Mais Influenciaram",65)
            c, h = self.gerarSecaoTabela(c,h,df2.to_numpy(), df2.columns.tolist(),2)
            c, h = self.gerarLegenda(c,"Tabela 2 - Principais Fatores de Influência", h+5)
            c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            c.save()

            page_w, page_h = letter
            pdf3 = name.replace(".pdf", "3.pdf")
            c = canvas.Canvas(pdf3)
            c, h = self.gerarSecao(c,'c',f"{data['output']}",320)
            c, h = self.gerarSecaoTabela(c,h,df3.to_numpy(), df3.columns.tolist(),3)
            c, h = self.gerarLegenda(c,f"Tabela 3 - Comparação do(a) {data['output']} entre o Município e o seu Grupo", h+5)
            c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            c.save()

            page_w, page_h = letter
            pdf4 = name.replace(".pdf", "4.pdf")
            c = canvas.Canvas(pdf4)
            height = 320 if df4.shape[0] == 1 else 270 if df4.shape[0] == 2 else 220 if df4.shape[0] >= 3 and df4.shape[0] <= 6 else 130
            c, h = self.gerarSecao(c,'c',"Vizinhos Mais Próximos",height)
            c, h = self.gerarSecaoTabela(c,h,df4.to_numpy(), df4.columns.tolist(),4)
            c, h = self.gerarSecao(c,'c1',"OBS: A PROXIMIDADE ENVOLVE O CONJUNTO TOTAL DOS FATORES E SUAS SEMELHANÇAS, AO INVÉS DE QUESTÕES GEOGRÁFICAS.",h+8)
            c, h = self.gerarLegenda(c,f"Tabela 4 - Municípios Mais Semelhantes a {municipio}", h+5)
            h = h+30
            c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
            c.saveState()
            c.showPage()
            c.save()

            self.merge_pdfs([pdf1, pdf2, pdf3, pdf4], name)
            for f in filter(lambda a : os.path.exists(a) and a != os.path.join('required_files', 'capa.pdf'), [pdf1, pdf2, pdf3, pdf4]):
                os.remove(f)

    def merge_pdfs(self, pdf_list: 'list[str]', output_path: str):
        pdf_writer = PdfWriter()

        for pdf in filter(lambda a : os.path.exists(a), pdf_list):
            pdf_reader = PdfReader(pdf)
            for page in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page])

        with open(output_path, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

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

    def gerarTabela(self,data,columns,j):
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
            c.drawImage(os.path.join('required_files/cabecalho.jpeg'), inch-8, page_h-50,page_w-inch-52,50)
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
                                 'Fatores'      : ['...'.join(x.rsplit(x[58:] or '...', 1)) for x in shap_columns],
                                 'Valor'        : [f"{x:.2f}" for x in shape_results['data']],
                                 'Influência'   : [f"{x:.3f}" for x in shape_results['values']]})
        
        df2 = pd.DataFrame(data={'Positivamente': list(reversed(['...'.join(shap_columns[chave].rsplit(shap_columns[chave][152:] or '...', 1)) for chave, valor in indices_maiores_valores.items()])) + ['---'] * (3 - len(indices_maiores_valores)),
                                 '+'            : list(reversed([f"{valor:.3f}" for chave, valor in indices_maiores_valores.items()])) + ['---'] * (3 - len(indices_maiores_valores)),
                                 'Negativamente': list(reversed(['...'.join(shap_columns[chave].rsplit(shap_columns[chave][152:] or '...', 1)) for chave, valor in indices_menores_valores.items()])) + ['---'] * (3 - len(indices_menores_valores)),
                                 '-'            : list(reversed([f"{valor:.3f}" for chave, valor in indices_menores_valores.items()])) + ['---'] * (3 - len(indices_menores_valores))})
        
        df3 = pd.DataFrame(data={f"Grupo {data['grupo']}": [f"{data['nota_media_grupo']*100:.2f} %"],
                                  'Município'                : [f"{data['nota_individual']*100:.2f} %"]})

        df4 = pd.DataFrame(data={'Nome': [label for label in data['vizinhos']] if len(data['vizinhos']) > 0 else ['---']})
        
        return df1, df2, df3, df4
    