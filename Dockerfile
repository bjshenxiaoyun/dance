FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# mediapipe/librosa 等大包走官方 PyPI 在国内构建机上很慢，改用腾讯云内网镜像源，
# 构建机和镜像源同在腾讯云网络内，下载速度快很多；--prefer-binary 优先用预编译 wheel，
# 避免个别包在没有编译工具链的 slim 镜像里退回源码编译（更慢，甚至可能直接编译失败）。
RUN pip install --no-cache-dir --prefer-binary \
    -i https://mirrors.cloud.tencent.com/pypi/simple \
    -r requirements.txt

COPY . .

EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
