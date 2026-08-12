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

def search_database(query_vector, top_k=2):
    """
    Sorunun vektörünü alır, veritabanındaki tüm vektörlerle karşılaştırır.
    En yüksek skoru (kosinüs benzerliği) alan parçaları döndürür.
    """
    if not os.path.exists(DB_PATH):
        return []
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tüm kayıtları getir
    cursor.execute("SELECT id, source_file, content, embedding FROM chunks")
    rows = cursor.fetchall()
    conn.close()
    
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
            "source": source,
            "content": content,
            "score": score
        })
        
    # En yüksek skordan (en ilgili) en düşüğe doğru sırala
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Sadece belirli bir benzerlik barajını (%5) geçen belgeleri kabul et
    filtered_results = [r for r in results if r["score"] >= 0.05]
    
    # Sadece en iyi (top_k) sonucu döndür
    return filtered_results[:top_k]

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
    
    # KRTIIK OPTIMIZASYON: Soru sorma asamasinda (sadece 1 cumle cevrildigi icin) 
    # GPU'yu Chat modeline (KV Cache) birakip, Embedding'i islemciye (CPU) yonlendiriyoruz!
    # Bu sayede ne VRAM cakisir ne de modeller birbirini kilitler. CPU tek bir cumleyi 0.1 saniyede cevirir.
    for v in embedding_model.variants:
        if 'cpu' in v.id.lower():
            embedding_model.select_variant(v)
            break
            
    if not embedding_model.is_loaded:
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
    
    # Veritabanında ara
    top_matches = search_database(query_vector, top_k=top_k)
    return top_matches
