FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /verify

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY verify.py .

# 一次性验收服务：执行完毕即以退出码报告结果（0 通过 / 1 失败）
CMD ["python", "verify.py"]
