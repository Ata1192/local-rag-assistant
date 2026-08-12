"""
==========================================================
  LOCAL RAG ASSISTANT -- RETRIEVER (Arama Motoru)
==========================================================

Bu dosyanın tek görevi var:
Kullanıcının sorduğu soruyu alır, SQLite veritabanındaki 
parçalarla karşılaştırır ve "en alakalı" X adet parçayı döndürür.
"""

import sqlite3
import json
import math
import os
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base.db')

def cosine_similarity(a, b):
    """İki vektör (sayı dizisi) arasındaki açıyı ölçerek benzerliği bulur."""
    if not a or not b:
        return 0.0
        
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def search_database(query_vector, text_query, top_k=2):
    """
    Sorunun vektörünü alır, veritabanındaki tüm vektörlerle karşılaştırır.
    En yüksek skoru (kosinüs benzerliği) alan parçaları döndürür.
    """
    if not os.path.exists(DB_PATH):
        return []
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. FTS5 Sanal Tablosunu (Keyword Arama İçin) Hazırla
    # Eğer tablo düzgün oluşturulmadıysa diye bir kereliğine düzeltelim:
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts 
        USING fts5(content)
    """)
    
    # FTS5 tablosu boş mu kontrol et, boşsa doldur
    cursor.execute("SELECT COUNT(*) FROM chunks_fts")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO chunks_fts(rowid, content) SELECT id, content FROM chunks")
        conn.commit()

    # Tüm kayıtları (Vektör + Metin) getir
    cursor.execute("SELECT id, source_file, content, embedding FROM chunks")
    rows = cursor.fetchall()
    
    results = []
    
    # Her bir belge parçası için:
    for row in rows:
        chunk_id, source, content, embedding_json = row
        # Metin olarak kaydettiğimiz JSON'u tekrar gerçek vektöre çeviriyoruz
        doc_vector = json.loads(embedding_json)
        
        # Soru ile bu belge parçasının benzerliğini ölç
        score = cosine_similarity(query_vector, doc_vector)
        
        # Sonuç listesine ekle
        results.append({
            "id": chunk_id,
            "source": source,
            "content": content,
            "vector_score": score
        })
        
    # Sadece Vektör skorlarına göre sırala
    results.sort(key=lambda x: x["vector_score"], reverse=True)
    
    vector_ranks = {}
    for rank, r in enumerate(results):
        vector_ranks[r["id"]] = rank + 1

    # 2. BM25 Kelime Araması (FTS5)
    # Gelen sorgudaki tüm geçmiş kelimeleri FTS için alalım. 
    # LLM zaten ilgisiz (eski aktör) kayıtlarını yanıtlarken eleyecektir.
    import re
    words = re.findall(r'\b\w{3,}\b', text_query)
    # Tekrarları sil
    words = list(set(words))
    
    fts_ranks = {}
    if words:
        # FTS sorgusu: kelime1 OR kelime2 OR kelime3...
        # Ancak FTS5 syntax hatası vermemesi için kelimeleri temizleyelim
        # ve anlamsız arama kelimelerini (stopwords) filtreleyelim.
        stop_words = {
            'hangi', 'yapımlarda', 'oynamıştır', 'oynadığı', 'ismini', 'bul', 
            'tüm', 'başka', 'kim', 'kimdir', 'nerede', 'film', 'filmlerde', 
            'dizi', 'dizilerde', 'bana', 'say', 'var', 'yok', 've', 'veya', 
            'ile', 'için', 'ne', 'neler', 'isim', 'isimleri', 'isimlerini', 
            'hakkında', 'nedir', 'nasıl'
        }
        
        clean_words = []
        for w in words:
            # Sadece harf ve rakam içeren kelimeleri al ve stopwords'te yoksa ekle
            if re.match(r'^[a-zA-Z0-9_ğüşıöçĞÜŞİÖÇ]+$', w) and w.lower() not in stop_words:
                clean_words.append(w)
                
        if clean_words:
            match_query = " OR ".join(clean_words)
            print(f"[DEBUG] FTS Query: {match_query}")
            try:
                cursor.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank", (match_query,))
                fts_rows = cursor.fetchall()
                for rank, fts_row in enumerate(fts_rows):
                    fts_ranks[fts_row[0]] = rank + 1
            except Exception as e:
                print(f"[FTS UYARISI] Kelime aramasi başarisiz oldu: {e}")
                
    conn.close()
    
    # 3. RRF (Reciprocal Rank Fusion) ile Skorları Birleştir
    # Tablo (CSV) verilerinde Kelime Eşleşmesi (İsim, Film Adı), Vektör eşleşmesinden çok daha değerlidir!
    # Bu yüzden FTS'in K sabitini düşük (daha yüksek puan), Vector'ün K sabitini yüksek tutuyoruz.
    K_vec = 60
    K_fts = 10
    final_results = []
    
    for r in results:
        v_rank = vector_ranks.get(r["id"], 10000)
        f_rank = fts_ranks.get(r["id"], 10000) # Kelime aramada çıkmadıysa çok düşük sıra ver
        
        # Sadece vektör benzerliği 0.05'in altındaysa ve kelime aramada da yoksa ele
        if r["vector_score"] < 0.05 and f_rank == 10000:
            continue
            
        rrf_score = (1.0 / (K_vec + v_rank)) + (1.0 / (K_fts + f_rank))
        
        final_results.append({
            "source": r["source"],
            "content": r["content"],
            "score": rrf_score,
            "vector_score": r["vector_score"],
            "fts_rank": f_rank
        })
        
    # Nihai birleşik RRF skoruna göre sırala
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Sadece en iyi (top_k) sonucu döndür
    return final_results[:top_k]

def get_relevant_context(query, top_k=2):
    """
    Uygulamadan (app.py) çağrılacak ana fonksiyondur.
    Soruyu vektöre çevirir, sonra arama fonksiyonunu tetikler.
    """
    # YENI MIMARI: Modeli burada (anlik olarak) yukle, kullan, sonra hemen VRAM'den sil!
    try:
        manager = FoundryLocalManager.instance
    except Exception:
        config = Configuration(app_name="local_rag_assistant")
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance
        
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    
    # Eger model halihazirda yukluyse bosuna tekrar yuklemeye veya varyant secmeye
    # calisip C++ motorunda memory leak (Connection Error) yaratmayalim!
    if not embedding_model.is_loaded:
        # KRTIIK OPTIMIZASYON: Soru sorma asamasinda GPU'yu Chat modeline (KV Cache) birakip,
        # Embedding'i islemciye (CPU) yonlendiriyoruz!
        for v in embedding_model.variants:
            if 'cpu' in v.id.lower():
                embedding_model.select_variant(v)
                break
        embedding_model.load()
        
    embedding_client = embedding_model.get_embedding_client()
    
    # Soruyu vektöre çevir (Foundry SDK generate_embeddings list bekler)
    response = embedding_client.generate_embeddings([query])
    
    if not response.data or not response.data[0].embedding:
        print("[UYARI] Foundry SDK sorgu icin vektor uretmedi (Bos veya tanimsiz sorgu olabilir).")
        return []
        
    query_vector = response.data[0].embedding
    
    # C++ ONNXRuntime çökmelerini (Connection error/silent crash) önlemek için 
    # CPU varyantını HER SORUDA yükleyip silmekten vazgeçiyoruz! 
    # İşlemci RAM'inde (System RAM) kalması GPU VRAM'i etkilemez. 
    # Bu yüzden unload() fonksiyonunu sildik.
    
    # Veritabanında hibrit ara (Vektör + Kelime)
    top_matches = search_database(query_vector, text_query=query, top_k=top_k)
    return top_matches
