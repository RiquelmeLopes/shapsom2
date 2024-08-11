## Para executar no Docker:

1. docker build -t streamlit-app
2. docker run --cpus="4.0" --memory="4g" -p 8501:8501 streamlit-app

## Para executar localmente (streamlit):

1. pip install -r requirements.txt
2. streamlit run main_page.py
