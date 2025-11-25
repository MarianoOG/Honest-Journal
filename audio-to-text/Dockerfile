# From https://github.com/fentresspaul61B/Deploy-Whisper-On-GCP
# Updated Dockerfile which loads model in mem.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG ENVIRONMENT=LOCAL
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN mkdir models
RUN if [ "$ENVIRONMENT" = "PROD" ]; then \
        python -c "import whisper; whisper.load_model('large-v3-turbo', download_root='/app/models')"; \
    else \
        python -c "import whisper; whisper.load_model('small', download_root='/app/models')"; \
    fi

COPY fastapi_app.py /app/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]