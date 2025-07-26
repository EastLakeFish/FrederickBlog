FROM python:3.10-slim

WORKDIR /app/Backend
COPY . /app/Backend

# OpenCV依赖于OpenGL库
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "uvicorn", "api:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]