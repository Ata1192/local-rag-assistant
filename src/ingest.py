"""
==========================================================
  LOCAL RAG ASSISTANT -- INGESTION SCRIPT (ingest.py)
  Belgeleri oku, parçala, vektöre çevir ve veritabanına kaydet
==========================================================

Bu dosya ne yapıyor?
  1. SQLite veritabanı oluşturur (veya varsa açar) ve içine bir tablo kurar.
  2. 'docs' klasöründeki belgeleri okur.
  3. Uzun belgeleri küçük paragraflara (chunk) ayırır.
     Neden? Çünkü RAG sistemleri tüm sayfayı değil, en alakalı kısa paragrafları arar.
  4. Foundry Local SDK'sını kullanarak bu paragrafları "embedding" (sayısal vektör) haline getirir.
  5. Hem orijinal metni hem de bu vektörü SQLite veritabanına kaydeder.
"""

import os
import sqlite3
import sys
import PyPDF2
import docx
import json
import csv
from tqdm import tqdm
from pathlib import Path
from foundry_local_sdk import Configuration, FoundryLocalManager

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR (CONFIGURATION)
# ─────────────────────────────────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base.db')
CHUNK_SIZE = 1000 # Kabaca kaç karakterlik parçalara böleceğiz

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 1: VERİTABANI (SQLite) KURULUMU
# ─────────────────────────────────────────────────────────────────────────────
def setup_database():
    """
    SQLite veritabanını ve 'chunks' tablosunu oluşturur.
    
    Tablo Yapısı:
    - id: Benzersiz kimlik
    - source_file: Bu metnin hangi dosyadan geldiği (kaynak belirtmek için)
    - content: Metnin kendisi (gerçek paragraf)
    - embedding: Vektör verisi (JSON formatında metin olarak saklayacağız)
    """
    print(f"[1/4] Veritabanı hazırlanıyor: {DB_PATH}")
    # Veritabanına bağlan (dosya yoksa otomatik yaratılır)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Mevcut tabloyu sil ve yeniden oluştur (her ingestion işleminde sıfırdan başlamak için)
    cursor.execute("DROP TABLE IF EXISTS chunks")
    
    # Yeni tablo yarat
    cursor.execute('''
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            content TEXT,
            embedding TEXT
        )
    ''')
    conn.commit()
    return conn

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 2: BELGELERİ OKUMA VE PARÇALAMA (CHUNKING)
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Uzun bir metni belirli boyutlardaki parçalara (chunk) ayırır.
    Gerçek dünya uygulamalarında kelime kelime (token) veya cümle cümle bölünür.
    Biz basitlik adına karakter sayısına ve paragraf boşluklarına göre böleceğiz.
    """
    # Önce paragraflara (çift satır atlama) göre bölmeye çalış
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # Eğer mevcut parça ve yeni paragrafın toplamı izin verilen boyutu aşmıyorsa, birleştir
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + "\n\n"
        else:
            # Sınır aşıldıysa, mevcut parçayı listeye ekle ve yenisine başla
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
            
    # Son kalan parçayı da ekle
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"PDF okuma hatasi ({filepath}): {e}")
    return text

def extract_text_from_docx(filepath):
    try:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Word okuma hatasi ({filepath}): {e}")
        return ""

def extract_text_from_csv(filepath):
    text = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return ""
            for row in reader:
                row_str = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                text += row_str + "\n\n"
    except Exception as e:
        print(f"CSV okuma hatasi ({filepath}): {e}")
    return text

def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext in ['.doc', '.docx']:
        return extract_text_from_docx(filepath)
    elif ext == '.csv':
        return extract_text_from_csv(filepath)
    return ""

def read_and_chunk_documents():
    """docs/ klasöründeki tüm desteklenen dosyaları okur ve parçalara böler."""
    print(f"\n[2/4] Belgeler okunuyor: {DOCS_DIR}")
    
    if not os.path.exists(DOCS_DIR):
        print(f"HATA: {DOCS_DIR} klasoru bulunamadi!")
        return []
        
    all_chunks = []
    # Klasördeki tüm dosyaları gez
    for filename in os.listdir(DOCS_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.txt', '.md', '.pdf', '.docx', '.csv']:
            file_path = os.path.join(DOCS_DIR, filename)
            content = extract_text(file_path)
            
            if not content.strip():
                continue
                
            print(f"      Okunuyor: {filename} ({len(content)} karakter)")
            # Metni parçala
            chunks = chunk_text(content)
            print(f"      -> {len(chunks)} parcaya bolundu.")
            
            # Hangi dosyadan geldiği bilgisini de parçaya ekleyelim
            for c in chunks:
                all_chunks.append({
                    "source": filename,
                    "content": c
                })
                
    return all_chunks

# ─────────────────────────────────────────────────────────────────────────────
# ADIM 3 & 4: EMBEDDING OLUŞTURMA VE VERİTABANINA KAYDETME
# ─────────────────────────────────────────────────────────────────────────────
def process_and_save_chunks(chunks, conn, embedding_client=None):
    """
    Metin parçalarını vektöre çevirir (embed eder) ve SQLite'a kaydeder.
    """
    if not chunks:
        print("Islenecek metin parcasi yok!")
        return
        
    loaded_locally = False
    if embedding_client is None:
        print(f"\n[3/4] Foundry Local baslatiliyor ve Embedding Modeli yukleniyor...")
        try:
            config = Configuration(app_name="local_rag_assistant")
            FoundryLocalManager.initialize(config)
        except Exception:
            pass
        manager = FoundryLocalManager.instance
        
        # Embedding modelini al
        embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
        try:
            embedding_model.load()
        except Exception as e:
            print(f"Varsayilan embedding modeli yuklenemedi: {e}. Alternatif varyant deneniyor...")
            for v in embedding_model.variants:
                try:
                    embedding_model.select_variant(v.id)
                    embedding_model.load()
                    print(f"Alternatif varyant yuklendi: {v.id}")
                    break
                except Exception:
                    continue
        embedding_client = embedding_model.get_embedding_client()
        loaded_locally = True
    
    print(f"\n[4/4] Toplam {len(chunks)} parca vektore cevriliyor ve veritabanina yaziliyor...")
    cursor = conn.cursor()
    
    # Sadece içerikleri liste olarak hazırlayalım ki topluca embed edebilelim
    texts_to_embed = [item["content"] for item in chunks]
    
    # Foundry Local ile batchler (küçük paketler) halinde embed edelim ki RAM taşmasın
    # NOT: 4GB VRAM ekran kartlarinda Chat Modeli ile ayni anda calisirken 
    # VRAM'in sismemesi icin batch_size cok kucuk (5) tutulmalidir!
    batch_size = 5
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embeddings"):
        batch = texts_to_embed[i:i+batch_size]
        try:
            response = embedding_client.generate_embeddings(batch)
            for j, item in enumerate(response.data):
                vector = item.embedding
                idx = i + j
                source = chunks[idx]["source"]
                content = chunks[idx]["content"]
                
                vector_json = json.dumps(vector)
                
                cursor.execute('''
                    INSERT INTO chunks (source_file, content, embedding)
                    VALUES (?, ?, ?)
                ''', (source, content, vector_json))
            conn.commit()
        except Exception as e:
            print(f"Batch {i} embed edilirken hata: {e}")
            
    print("      Tum veriler basariyla kaydedildi!")
    
    # Belleği temizle (Sadece bu script içinden yüklendiyse)
    if loaded_locally:
        embedding_model.unload()

# ─────────────────────────────────────────────────────────────────────────────
# DIŞARIDAN TEK DOSYA YÜKLEME FONKSİYONU (STREAMLIT İÇİN)
# ─────────────────────────────────────────────────────────────────────────────
def ingest_file(filepath, embedding_client=None, conn=None):
    """Tek bir dosyayı işleyip veritabanına ekler."""
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True

    filename = os.path.basename(filepath)
    content = extract_text(filepath)
    if not content.strip():
        if close_conn: conn.close()
        return 0
        
    chunks = chunk_text(content)
    chunk_dicts = [{"source": filename, "content": c} for c in chunks]
    
    process_and_save_chunks(chunk_dicts, conn, embedding_client=embedding_client)
    
    if close_conn:
        conn.close()
        
    return len(chunk_dicts)

# ─────────────────────────────────────────────────────────────────────────────
# ANA ÇALIŞTIRMA BLOĞU
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("==================================================")
    print("  VERİ İÇE AKTARMA (INGESTION) BAŞLIYOR")
    print("==================================================")
    
    # 1. Veritabanını kur
    conn = setup_database()
    
    # 2. Belgeleri oku ve parçala
    chunks = read_and_chunk_documents()
    
    # 3. Vektöre çevir ve veritabanına kaydet
    process_and_save_chunks(chunks, conn)
    
    # Veritabanı bağlantısını kapat
    conn.close()
    
    print("==================================================")
    print("  INGESTION TAMAMLANDI! knowledge_base.db HAZIR.")
    print("==================================================")

if __name__ == "__main__":
    main()
