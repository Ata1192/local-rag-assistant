# Ana imaj olarak hafif bir Python sürümü seçiyoruz
FROM python:3.11-slim

# Konteyner içerisindeki çalışma dizini
WORKDIR /app

# Sistem bağımlılıklarını kur (Eğer sqlite3 ekstra paket isterse veya derleme gerekirse)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Gereksinim dosyasını kopyala ve kütüphaneleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Projedeki tüm kodları konteynerin içine kopyala
COPY . .

# Streamlit uygulamasının dışarıyla konuşacağı port
EXPOSE 8501

# Container ayağa kalktığında çalıştırılacak komut
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
