## Para executar no Docker:

docker build -t streamlit-app .
docker run --cpus="4.0" --memory="4g" -p 8501:8501 streamlit-app

## Para executar localmente (streamlit):

pip install -r requirements.txt
streamlit run main_page.py
