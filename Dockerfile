FROM python:3.9-slim

WORKDIR /app

COPY streamlit_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "run_streamlit.py", "--server.address=0.0.0.0"]
