# app/Dockerfile
FROM python:3.10

WORKDIR /app

# Copy application code
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/raalrwai/Multi-Agent-RAG-Utility-App/tree/Mickey .

# COPY . .

# # Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV RUNNING_IN_DOCKER=true

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8501"]
