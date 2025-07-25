FROM python:3.10-slim
WORKDIR /app

# 复制所需文件
COPY assets /app/assets
COPY ./checkpoints /app/checkpoints
COPY infer.py /app/infer.py
COPY model.py /app/model.py
COPY requirements.txt /app/requirements.txt

# 安装依赖
RUN python -m venv ./venv
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# 设置容器入口程序
ENTRYPOINT ["python", "infer.py"]