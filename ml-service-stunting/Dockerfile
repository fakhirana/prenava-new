FROM python:3.10-slim

# Install system library yang dibutuhkan oleh LightGBM (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dan install dependencies terlebih dahulu agar ter-cache oleh Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode proyek dan model ke dalam container
COPY . .

# Expose port 5000 sesuai spesifikasi ml-service-stunting
EXPOSE 5000

# Pastikan Flask mendengarkan di 0.0.0.0 agar bisa diakses dari luar container
ENV FLASK_APP=api/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

CMD ["python", "api/app.py"]