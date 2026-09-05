# Local RAG Assistant

Bu proje, tamamen yerel (bilgisayarınızda) internet bağlantısına ihtiyaç duymadan çalışan, kendi belgelerinizi okuyup anlayabilen ve size profesyonel cevaplar verebilen bir RAG (Retrieval-Augmented Generation) asistanıdır. Son yapılan güncellemelerle birlikte proje, geniş çaplı veri setlerini (örnek: Hastalık-Semptom veritabanları) işleyebilen bir "Medikal Teşhis Asistanı" kapasitesine ulaşmıştır.

Güvenlik veya gizlilik kaygısı olan şirketler, kurumlar veya bireyler için verilerin dışarı çıkmadığı güvenli bir yapay zeka arama motorudur.

## Ozellikler

- %100 Yerel Calisir: Hiçbir veriniz internete (OpenAI, Google vs.) gönderilmez. Tüm işlemler bilgisayarınızda gerçekleşir.
- Arayuz Uzerinden Dosya Yukleme ve Yonetim: Belgeleri manuel olarak klasöre atmak yerine doğrudan Streamlit arayüzündeki sol menüyü kullanarak yükleyebilir ve istemediğiniz belgeleri tek tıkla silebilirsiniz (.txt, .md, .pdf, .docx, .csv desteklenir).
- Obsidian Entegrasyonu (Graph View): Veritabanındaki tüm belgeleri, chunk'ları ve aralarındaki ilişkileri Obsidian üzerinden görselleştirerek harita (Graph) görünümünde inceleyebilirsiniz. Uygulama arayüzünden tek tıkla Obsidian Vault oluşturulup açılabilir.
- Dinamik RAM/VRAM Yonetimi: Sistem belleğini korumak amacıyla RAG arama süreci (Embedding) ve Cevap üretme süreci (LLM) arasında modeller bellekten dinamik olarak yüklenip silinir (Unload). Bu sayede düşük donanımlı sistemlerde (Örn: 8GB RAM) bile ağır modeller sorunsuz çalışır.
- Akilli Alinti (Citation) ve Coklu Format Destegi: Verdiği cevapları sizin belgelerinizden alır ve hangi belgeden okuduğunu kaynak gösterir. Karmaşık CSV veri setlerini akıllı gruplama yöntemleriyle optimize ederek okuyabilir.
- Halusinasyon Korumasi: Eğer sorunuzun cevabı belgelerde yoksa, uydurmak yerine kesin bir dille bilgiyi bulamadığını belirtir (Medical asistan için kısmi eşleşmeleri de mantıklı çerçevede sunacak şekilde yapılandırılmıştır).

## Kurulum Adimlari

**1. Gereksinimler**
- Python 3.10 veya üzeri
- `pip` paket yöneticisi
- Foundry Local SDK (Yerel model yönetimi için)

**2. Kurulum**
Proje dizininde sanal ortam oluşturup gerekli kütüphaneleri yükleyin:
```bash
python -m venv venv
venv\Scripts\activate
pip install streamlit foundry-local-sdk pandas PyPDF2 python-docx
```

## Nasil Kullanilir?

Projeyi tek tıkla başlatmak için ana dizindeki `Start_RAG_Assistant.bat` dosyasını çalıştırabilirsiniz. Alternatif olarak terminalden şu komutu girebilirsiniz:

```bash
streamlit run app.py
```

### Belgeleri Yukleme ve Yonetme
1. Tarayıcıda açılan arayüzde, sol menüde bulunan "Belge Yükle" alanını kullanarak bilgisayarınızdaki verileri sisteme aktarabilirsiniz.
2. Yüklenen belgeler arka planda otomatik olarak analiz edilir, parçalara (chunk) bölünür, vektörlere çevrilir ve `knowledge_base.db` isimli SQLite veritabanına kaydedilir.
3. Yine sol menüdeki "Yüklü Belgeleri Yönet" kısmından sistemde kayıtlı dosyaları görebilir, yanlarındaki çarpı ikonuna basarak veritabanından tamamen silebilirsiniz.

### Obsidian ile Gorsellestirme
1. Sol menüde bulunan "Obsidian Vault Oluştur" butonuna tıklayın.
2. İşlem tamamlandığında "Obsidian'de Görüntüle" butonu belirecektir. Bu butona basarak veritabanınızın Graph (Harita) görünümünü Obsidian uygulaması üzerinde 3 boyutlu olarak inceleyebilirsiniz.

## Proje Yapisi
- `docs/`: Arayüzden yüklenen belgelerin geçici veya kalıcı olarak saklandığı klasör.
- `src/ingest.py`: Belgeleri (PDF, Word, CSV, TXT) okuyup vektör veritabanına akıllı algoritmalarla kaydeden script (örnek: CSV içerisindeki yüz binlerce satırı benzersiz gruplara ayırarak optimize eder).
- `src/retriever.py`: Sorduğunuz soruya en yakın belgeleri vektör ve metin tabanlı hibrid arama (FTS5 + Cosine Similarity) ile bulan motor.
- `app.py`: Streamlit ile yazılmış modern, kullanıcı dostu ana sohbet arayüzü.
- `knowledge_base.db`: Tüm verilerinizin ve sohbet geçmişinizin güvenle tutulduğu yerel veritabanı.
