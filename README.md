# 🤖 Offline Local RAG Assistant

Bu proje, tamamen yerel (bilgisayarınızda) internet bağlantısına ihtiyaç duymadan çalışan, kendi belgelerinizi (PDF, TXT, MD) okuyup anlayabilen ve size cevaplar verebilen bir RAG (Retrieval-Augmented Generation) asistanıdır.

Güvenlik veya gizlilik kaygısı olan şirketler, kurumlar veya bireyler için verilerin dışarı çıkmadığı güvenli bir yapay zeka arama motorudur. 

## 🚀 Özellikler
- **%100 Yerel Çalışır:** Hiçbir veriniz internete (OpenAI, Google vs.) gönderilmez.
- **Foundry Local SDK Entegrasyonu:** Güçlü ve hafif yerel modeller (Qwen2.5 vb.) kullanır.
- **Streamlit Arayüzü:** Kullanımı kolay, modern ve şık bir web sohbet arayüzü sunar.
- **Akıllı Alıntı (Citation):** Verdiği cevapları sizin belgelerinizden alır ve hangi belgenin hangi paragrafından okuduğunu size kaynak olarak gösterir.
- **Halüsinasyon Koruması:** Eğer sorunuzun cevabı belgelerde yoksa, uydurmak yerine kesin bir dille "Bilmiyorum" der.

## 🛠️ Kurulum Adımları

**1. Gereksinimler**
- Python 3.10 veya üzeri
- `pip` paket yöneticisi

**2. Sanal Ortam (Virtual Environment) Kurulumu**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Gerekli Kütüphanelerin Yüklenmesi**
```bash
pip install streamlit foundry-local-sdk
```

## 📖 Nasıl Kullanılır?

Sistem iki aşamalı çalışır: Önce belgeleri okutmak (Ingestion), sonra soru sormak (Retrieval).

### Adım 1: Belgeleri Yükleme (Ingestion)
1. Proje ana dizinindeki `docs/` klasörünün içine okutmak istediğiniz metin belgelerini (`.md`, `.txt`) atın.
2. Terminalden veritabanı oluşturucu scripti çalıştırın:
```bash
python src/ingest.py
```
3. Bu işlem belgeleri küçük parçalara (chunk) böler, vektörlere çevirir ve `knowledge_base.db` isimli SQLite veritabanına kaydeder.

### Adım 2: Asistanı Başlatma ve Soru Sorma (Retrieval)
1. Belgeler yüklendikten sonra sohbet arayüzünü başlatın:
```bash
streamlit run app.py
```
2. Tarayıcınızda açılan ekranda belgeleriniz hakkında sorular sormaya başlayabilirsiniz!

## 🏗️ Proje Yapısı
- `docs/`: Belgelerinizi koyacağınız klasör.
- `src/ingest.py`: Belgeleri okuyup vektör veritabanına kaydeden script.
- `src/retriever.py`: Sorduğunuz soruya en yakın belgeyi veritabanından bulan arama motoru kodu.
- `app.py`: Streamlit ile yazılmış ana sohbet arayüzü.
- `knowledge_base.db`: Tüm verilerinizin güvenle şifrelenip tutulduğu yerel veritabanı.

---
*Not: Sistem küçük ve yerel modeller kullandığı için karmaşık veya çapraz dilli (Örn: İngilizce belgeye Türkçe soru) sorgularda İngilizce terimleri de soruya dahil etmek performansı artırabilir.*
