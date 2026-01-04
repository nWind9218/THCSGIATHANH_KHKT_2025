# =========================
# Dockerfile cho bot Python
# Port mở: 4000
# =========================

# 1. Sử dụng Python 3.11 slim (nhẹ, nhanh)
FROM python:3.11-slim

# 2. Biến môi trường để Python chạy mượt hơn
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Thiết lập thư mục làm việc trong container
WORKDIR /app

# 4. Cài gói hệ thống cần thiết (nếu có thư viện build từ source)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements.txt và cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ mã nguồn vào container
COPY . .

# 7. Mở port 4000 cho bot / webhook
EXPOSE 4000

# 8. Lệnh chạy bot khi container start
CMD ["python", "main.py"]
