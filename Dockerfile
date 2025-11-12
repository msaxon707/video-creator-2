FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt

COPY main.py ./
CMD ["python", "main.py"]
