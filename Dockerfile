# app/Dockerfile
FROM python:3.10

WORKDIR /app

# Copy application code
COPY . .

# # Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV RUNNING_IN_DOCKER=true

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8501"]
