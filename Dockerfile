FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libavcodec-extra && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/main.py ./main.py
COPY frontend ./frontend
ENV WHISPER_MODEL=small WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
