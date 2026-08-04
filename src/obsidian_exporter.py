import os
import sqlite3
import shutil
import json
import re

def sanitize_filename(filename):
    """Dosya adlarında kullanılamayacak karakterleri temizler."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def export_to_obsidian(db_path="knowledge_base.db", output_dir="Obsidian_Vault"):
    """
    RAG veritabanındaki belgeleri, parçaları ve sohbetleri okuyup
    Obsidian'da açılmaya hazır içiçe geçmiş (wiki-linkli) bir Vault klasörü yaratır.
    """
    docs_dir = os.path.join(output_dir, "Documents")
    chunks_dir = os.path.join(output_dir, "Chunks")
    chats_dir = os.path.join(output_dir, "Chats")
    
    # Sadece kendi oluşturduğumuz klasörleri temizle, .obsidian klasörüne (pluginlere/ayarlara) DOKUNMA!
    for d in [docs_dir, chunks_dir, chats_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Belgeler ve Chunkları Aktar
    try:
        cursor.execute("SELECT id, source_file, content FROM chunks")
        chunks = cursor.fetchall()
        
        doc_links = {} # source_file -> [chunk_link1, chunk_link2]
        
        for chunk_id, source, content in chunks:
            if not source:
                source = "Bilinmeyen Kaynak"
            
            source_safe = sanitize_filename(source)
            chunk_filename = f"{source_safe} - Bölüm {chunk_id}"
            
            if source_safe not in doc_links:
                doc_links[source_safe] = []
                
            doc_links[source_safe].append(chunk_filename)
            
            # Chunk dosyasını yaz
            with open(os.path.join(chunks_dir, f"{chunk_filename}.md"), "w", encoding="utf-8") as f:
                f.write(f"**Kaynak Belge:** [[{source_safe}]]\n\n")
                f.write("---\n")
                f.write(f"{content}\n")
                
        # Ana belge (Document) dosyalarını yaz
        for doc, links in doc_links.items():
            # Orijinal dosyayı (PDF, CSV vs.) Vault'a kopyala
            original_path = os.path.join("docs", doc)
            if os.path.exists(original_path):
                shutil.copy2(original_path, os.path.join(docs_dir, doc))
                
            with open(os.path.join(docs_dir, f"{doc}.md"), "w", encoding="utf-8") as f:
                f.write(f"# ▤ {doc}\n\n")
                if os.path.exists(original_path):
                    f.write(f"**Orijinal Dosya:** [[{doc}]]\n\n")
                f.write("Bu belge, vektörel arama için aşağıdaki alt bölümlere ayrılmıştır:\n\n")
                for link in links:
                    f.write(f"- [[{link}]]\n")
    except sqlite3.OperationalError:
        print("Chunks tablosu bulunamadı.")
        
    # 2. Sohbetleri Aktar
    try:
        cursor.execute("SELECT id, title, created_at FROM chats")
        chats = cursor.fetchall()
        
        for chat_id, title, created_at in chats:
            title_safe = sanitize_filename(f"{title} - {chat_id[:8]}")
            
            # Sohbet geçmişini al
            try:
                cursor.execute("SELECT role, content, sources FROM chat_history WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
            except sqlite3.OperationalError:
                # Eski veritabanı şeması ise
                cursor.execute("SELECT role, content, NULL as sources FROM chat_history WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
                
            history = cursor.fetchall()
            
            with open(os.path.join(chats_dir, f"{title_safe}.md"), "w", encoding="utf-8") as f:
                f.write(f"# ↹ Sohbet: {title}\n")
                f.write(f"**Tarih:** {created_at}\n\n")
                f.write("---\n\n")
                
                for role, content, sources_json in history:
                    if role == "system":
                        if content == "MEMORY_CLEARED":
                            f.write("\n> [!info]\n> *Yapay zeka hafızası bu noktada sıfırlandı.*\n\n---\n\n")
                        continue
                        
                    display_role = "⯌ Kullanıcı" if role == "user" else "⊞ Yapay Zeka"
                    f.write(f"### {display_role}\n\n{content}\n\n")
                    
                    if sources_json:
                        try:
                            sources = json.loads(sources_json)
                            if sources:
                                f.write("> **☷ Kullanılan Kaynaklar:**\n")
                                seen_sources = set()
                                for s in sources:
                                    # Sohbetlerde direk ana belgeye link veriyoruz, böylece Graph View'da
                                    # sohbetler ile ana belgeler devasa yıldızlar gibi parlıyor!
                                    source_safe = sanitize_filename(s.get("source", "Bilinmeyen Kaynak"))
                                    if source_safe not in seen_sources:
                                        f.write(f"> - [[{source_safe}]]\n")
                                        seen_sources.add(source_safe)
                                f.write("\n")
                        except:
                            pass
                
    except sqlite3.OperationalError:
        print("Chats tablosu bulunamadı.")
        
    conn.close()
    return os.path.abspath(output_dir)

if __name__ == "__main__":
    out = export_to_obsidian("../knowledge_base.db", "../Obsidian_Vault")
    print(f"Exported to {out}")
