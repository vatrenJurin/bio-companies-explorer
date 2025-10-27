# Bio Companies Explorer (Streamlit)

An interactive web app to explore biological companies data from a CSV.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload your CSV and use the left sidebar to navigate.

## One‑click deploy (Streamlit Community Cloud)

1. Push these files to a **public GitHub repo**.
2. Go to https://share.streamlit.io/ and click **New app**.
3. Select your repo and choose `app.py` as the entry point.
4. Add `requirements.txt` if prompted. Deploy!

## Optional: Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

Build & run:
```bash
docker build -t bio-explorer .
docker run -p 8501:8501 bio-explorer
```

Open http://localhost:8501