"""
==========================================================
  LOCAL RAG ASSISTANT -- STREAMLIT WEB ARAYÜZÜ (app.py)
==========================================================
Çalıştırmak için terminale şunu yazmalısın:
  venv\Scripts\streamlit run app.py
"""

import streamlit as st
import os
import sqlite3
import uuid
import json
import webbrowser
import urllib.parse
from foundry_local_sdk import Configuration, FoundryLocalManager
from src.retriever import get_relevant_context
from src.ingest import ingest_file
from src.obsidian_exporter import export_to_obsidian

# ─────────────────────────────────────────────────────────────────────────────
# 1. STREAMLIT SAYFA AYARLARI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yerel RAG Asistanı",
    page_icon="🤖",
    layout="centered"
)

st.title("Offline RAG Asistanı")
st.markdown("Cihazınızda çalışan, belgelerinizi okuyabilen yerel yapay zeka.")

# YENİ: Özel CSS Tasarımını Yükle
if os.path.exists("style.css"):
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELLERİ YÜKLEME (ÖNBELLEKLİ)
# ─────────────────────────────────────────────────────────────────────────────
# @st.cache_resource, Streamlit her yenilendiğinde bu fonksiyonun
# tekrar çalışmasını (modellerin tekrar yüklenmesini) engeller!
@st.cache_resource(show_spinner=False)
def init_models(embedding_model_name: str, chat_model_name: str):
    """Modelleri sadece bir kez belleğe yükler ve istemcileri döndürür."""
    # SDK'yı başlat
    try:
        config = Configuration(app_name="local_rag_assistant")
        FoundryLocalManager.initialize(config)
    except Exception:
        pass
        
    manager = FoundryLocalManager.instance
    
    # ---------------- GPU OFFLOAD ADIMLARI ----------------
    print("[LOG] CUDA Execution Provider'lar kontrol ediliyor/indiriliyor...")
    try:
        manager.download_and_register_eps()
        print("[LOG] CUDA EP'ler basariyla kaydedildi.")
    except Exception as e:
        print(f"[LOG] CUDA EP kaydedilirken hata: {e}")
    # ------------------------------------------------------
    
    print("[LOG] Embedding modeli hazirlaniyor...")
    embedding_model = manager.catalog.get_model(embedding_model_name)
    
    # Varsayilan olarak CUDA (GPU) kullanmasi icin hicbir seyi zorlamiyoruz.
    # Eger VRAM yetmezse FoundryLocalManager kendisi fallback yapabilir.
            
    embedding_model.download()
    # YENI MIMARI: Embedding modelini burada YUKLEMIYORUZ! (VRAM acmak icin)
    # Sadece indirildiginden (cache'de oldugundan) emin oluyoruz.
                
    print(f"[LOG] Chat modeli hazirlaniyor ({chat_model_name})...")
    chat_model = manager.catalog.get_model(chat_model_name)
    
    # SADECE CUDA Varyantını Bul ve Seç
    for v in chat_model.variants:
        if 'cuda' in v.id.lower():
            print(f"[LOG] CUDA Varyanti secildi: {v.id}")
            chat_model.select_variant(v)
            break
            
    print("[LOG] Chat modeli indiriliyor (varsa kontrol ediliyor)...")
    try:
        chat_model.download()
        print("[LOG] Chat modeli VRAM'e yukleniyor (Bu biraz zaman alabilir)...")
        chat_model.load()
        chat_client = chat_model.get_chat_client()
        # Kritik OOM Önlemi: KV Cache buffer boyutunu sınırla
        chat_client.settings.max_tokens = 512
        print("[LOG] Modeller tamamen hazir!")
    except Exception as e:
        print(f"[LOG] HATA: Model yuklenemedi: {e}")
        st.error(f"Seçilen model ({chat_model_name}) yüklenemedi. Bilgisayarınızın donanımı veya internet bağlantısı bu modeli desteklemiyor olabilir. Hata: {e}")
        st.stop()
        
    return chat_client

# Modelleri yüklüyoruz (ilk açılışta 5-10 saniye sürer, sonraki sorgularda anında)
with st.spinner("Modeller yükleniyor, lütfen bekleyin... (Sadece ilk açılışta)"):
    if "chat_model_name" not in st.session_state:
        st.session_state.chat_model_name = "ministral-3-3b-instruct-2512"
    chat_client = init_models("qwen3-embedding-0.6b", st.session_state.chat_model_name)

# ─────────────────────────────────────────────────────────────────────────────
# 3. VERİTABANI VE SOHBET GEÇMİŞİ (ÇOKLU SOHBET)
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("knowledge_base.db")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT chat_id FROM chat_history LIMIT 1")
    except sqlite3.OperationalError:
        # Eğer eski tablo yapısıysa (chat_id yoksa) sil
        conn.execute("DROP TABLE IF EXISTS chat_history")
        
    conn.execute('''CREATE TABLE IF NOT EXISTS chats 
                    (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT)''')
    try:
        conn.execute("ALTER TABLE chat_history ADD COLUMN sources TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

def get_all_chats():
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM chats ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]

def create_new_chat():
    chat_id = str(uuid.uuid4())
    conn = sqlite3.connect("knowledge_base.db")
    conn.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (chat_id, "Yeni Sohbet"))
    welcome_msg = "Merhaba! Sana nasıl yardımcı olabilirim? Belgelerimden bana sorular sorabilirsin."
    conn.execute("INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, "assistant", welcome_msg))
    conn.commit()
    conn.close()
    return chat_id

def load_chat_history(chat_id):
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT role, content, sources FROM chat_history WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    except sqlite3.OperationalError:
        # Schema migration fallback in case it hasn't run yet
        cursor.execute("SELECT role, content FROM chat_history WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
        rows = cursor.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in rows]
        
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        sources = json.loads(r[2]) if r[2] else None
        result.append({"role": r[0], "content": r[1], "sources": sources})
    return result

def save_message(chat_id, role, content, sources=None):
    conn = sqlite3.connect("knowledge_base.db")
    sources_json = json.dumps(sources) if sources else None
    
    try:
        conn.execute("INSERT INTO chat_history (chat_id, role, content, sources) VALUES (?, ?, ?, ?)", (chat_id, role, content, sources_json))
    except sqlite3.OperationalError:
        conn.execute("INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
        
    conn.commit()
    conn.close()

def update_chat_title(chat_id, new_title):
    with sqlite3.connect("knowledge_base.db") as conn:
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, chat_id))
        conn.commit()

def delete_chat(chat_id):
    with sqlite3.connect("knowledge_base.db") as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
        conn.commit()

def get_all_documents():
    docs = []
    try:
        with sqlite3.connect("knowledge_base.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT source_file FROM chunks")
            docs = [r[0] for r in cursor.fetchall()]
    except sqlite3.OperationalError:
        pass
    return docs

def delete_document(filename):
    # Veritabanından sil
    try:
        with sqlite3.connect("knowledge_base.db") as conn:
            conn.execute("DELETE FROM chunks WHERE source_file = ?", (filename,))
            conn.commit()
    except sqlite3.OperationalError:
        pass
    
    # Fiziksel dosyayı sil
    file_path = os.path.join("docs", filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Dosya silinemedi: {e}")

# --- INITIALIZATION ---
all_chats = get_all_chats()
if not all_chats:
    new_id = create_new_chat()
    st.session_state.current_chat_id = new_id
    all_chats = get_all_chats()
elif "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in [c["id"] for c in all_chats]:
    st.session_state.current_chat_id = all_chats[0]["id"]

current_chat_id = st.session_state.current_chat_id

# --- MAIN CHAT HEADER ---
chat_title = next((c["title"] for c in all_chats if c["id"] == current_chat_id), "Sohbet")
st.subheader(f"> {chat_title}")
st.divider()

st.session_state.messages = load_chat_history(current_chat_id)

user_query_input = st.chat_input("Bir soru sorun...")
is_trigger_active = "trigger_query" in st.session_state
is_currently_generating = is_trigger_active or bool(user_query_input and user_query_input.strip())

# Geçmiş mesajları ekrana çiz
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system" and msg["content"] == "MEMORY_CLEARED":
        st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #4CAF50; font-size: 0.9em;'><em>Hafıza sıfırlandı</em></div><br>", unsafe_allow_html=True)
        continue
        
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Eğer bu mesajın kaynakları varsa, geçmişte de olsa expander (açılır kutu) ile göster!
        if msg.get("sources") and msg["role"] == "assistant":
            grouped_sources = {}
            for m in msg["sources"]:
                if m['source'] not in grouped_sources:
                    grouped_sources[m['source']] = []
                grouped_sources[m['source']].append(m)
                
            with st.expander("Kullanılan Kaynakları İncele"):
                for source_name, chunks in grouped_sources.items():
                    st.markdown(f"**[ {source_name} ]**")
                    for chunk_idx, chunk in enumerate(chunks):
                        score_text = f" (Benzerlik: {chunk['score']:.4f})" if 'score' in chunk else ""
                        st.info(f"**Bölüm {chunk_idx+1}**{score_text}\n\n{chunk['content']}")
        if msg["content"].strip():
            # Üretim esnasında butonları tamamen gizle (hem kopyalama hem regenerate)
            if not is_currently_generating:
                cols = st.columns([0.08, 0.08, 0.84])
                with cols[0]:
                    if st.button("⎘", key=f"copy_{i}", help="Mesajı Kopyala"):
                        import pyperclip
                        try:
                            pyperclip.copy(msg["content"])
                            st.toast("Panoya kopyalandı! ⎘")
                        except Exception:
                            st.toast("Kopyalama başarısız!")
                
                if msg["role"] == "assistant" and i == len(st.session_state.messages) - 1 and len(st.session_state.messages) >= 2:
                    with cols[1]:
                        if st.button("⟳", key=f"regen_{i}", help="Yeniden Üret (Regenerate)"):
                            # Çift tıklamayı engellemek için kilit kontrolü
                            if not st.session_state.get("regen_locked", False):
                                st.session_state.regen_locked = True
                                last_user_msg = st.session_state.messages[-2]["content"]
                                # Veritabanından son 2 mesajı sil
                                conn = sqlite3.connect("knowledge_base.db")
                                conn.execute("DELETE FROM chat_history WHERE id IN (SELECT id FROM chat_history WHERE chat_id = ? ORDER BY id DESC LIMIT 2)", (current_chat_id,))
                                conn.commit()
                                conn.close()
                                # Arayüzü yeniden tetikle
                                st.session_state.trigger_query = last_user_msg
                                st.rerun()

# --- CHAT ACTIONS (BOTTOM) ---
cols = st.columns([0.85, 0.15])
with cols[1]:
    if st.button("× Temizle", help="Sohbet geçmişini siler ve yapay zekayı sıfırlar."):
        conn = sqlite3.connect("knowledge_base.db")
        conn.execute("INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)", (current_chat_id, "system", "MEMORY_CLEARED"))
        conn.commit()
        conn.close()
        st.session_state.messages = load_chat_history(current_chat_id)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# YENİ: SİDEBAR DOSYA YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set(get_all_documents())

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# F5 atıldığında veya sayfa yüklendiğinde `docs/` klasöründeki YENİ dosyaları otomatik tara ve ekle
new_files_to_ingest = [f for f in os.listdir("docs") if f.endswith(('.txt', '.md', '.pdf', '.docx', '.csv')) and f not in st.session_state.ingested_files]

if new_files_to_ingest:
    with st.spinner(f"{len(new_files_to_ingest)} yeni dosya tespit edildi, topluca ekleniyor..."):
        # HIZLANDIRMA: Döngü dışına çıkarıldı! Model bir kere silinir, tüm dosyalar işlenir, model geri yüklenir.
        manager = FoundryLocalManager.instance
        chat_model = manager.catalog.get_model(st.session_state.chat_model_name)
        try: chat_model.unload()
        except Exception: pass
        
        for filename in new_files_to_ingest:
            ingest_file(os.path.join("docs", filename), embedding_client=None)
            st.session_state.ingested_files.add(filename)
            
        try: chat_model.load()
        except Exception: pass

with st.sidebar:
    st.markdown("---")
    st.header("Veritabanı Haritası")
    if st.button("→ Obsidian Vault Oluştur", use_container_width=True, help="Tüm belgeleri ve sohbetleri görselleştirmek için Obsidian'a aktarır."):
        with st.spinner("Obsidian Vault oluşturuluyor..."):
            try:
                vault_path = export_to_obsidian("knowledge_base.db", "Obsidian_Vault")
                if vault_path:
                    st.success(f"Başarıyla oluşturuldu!\n\nObsidian uygulamasında şu klasörü açın:\n`{vault_path}`")
                else:
                    st.error("Veritabanı bulunamadı.")
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")
                
    if os.path.exists("Obsidian_Vault"):
        if st.button("→ Obsidian'de Görüntüle", use_container_width=True, help="Oluşturulan vault'u Obsidian uygulamasında otomatik açar."):
            # Doğrudan Vault adıyla açıyoruz (Yol hatalarını tamamen önler)
            encoded_vault = urllib.parse.quote("Obsidian_Vault")
            obsidian_url = f"obsidian://open?vault={encoded_vault}"
            webbrowser.open(obsidian_url)
                
    st.markdown("---")
    st.header("Sohbetler")
    
    if st.button("+ Yeni Sohbet Aç", use_container_width=True):
        new_id = create_new_chat()
        st.session_state.current_chat_id = new_id
        st.rerun()
        
    for chat in all_chats:
        col1, col2 = st.columns([0.80, 0.20])
        with col1:
            if chat["id"] == current_chat_id:
                st.button(f"● {chat['title']}", key=f"btn_{chat['id']}", disabled=True, use_container_width=True)
            else:
                if st.button(f"○ {chat['title']}", key=f"btn_{chat['id']}", use_container_width=True):
                    st.session_state.current_chat_id = chat["id"]
                    st.rerun()
        with col2:
            with st.popover("×"):
                st.write("Silinsin mi?")
                if st.button("Evet", key=f"del_{chat['id']}"):
                    delete_chat(chat['id'])
                    st.rerun()
        
    st.markdown("---")
    st.header("Belge Yükle")
    # Dosya yükleyiciye dinamik key veriyoruz ki yükleme bittiğinde sıfırlayabilelim (Ghost Upload bug fix)
    uploaded_file = st.file_uploader("PDF, Word, Metin veya CSV yükle", type=["txt", "md", "pdf", "docx", "csv"], key=f"uploader_{st.session_state.uploader_key}")
    
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.ingested_files:
            save_path = os.path.join("docs", uploaded_file.name)
            # Dosyayı docs klasörüne kaydet
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Belge veritabanına ekleniyor..."):
                # HIZLANDIRMA (YONTEM 1): VRAM'i bosalt ve koca GPU'yu Embedding modeline tahsis et!
                manager = FoundryLocalManager.instance
                chat_model = manager.catalog.get_model(st.session_state.chat_model_name)
                try: chat_model.unload()
                except Exception: pass
                
                # ingest_file fonksiyonunu çağırıp direkt embeddings oluştur (kendi yükleyip boşaltacak)
                chunks_added = ingest_file(save_path, embedding_client=None)
                
                # CHAT MODELINI GERI YUKLE
                try: chat_model.load()
                except Exception: pass
                if chunks_added > 0:
                    st.success(f"{uploaded_file.name} başarıyla eklendi! ({chunks_added} parça)")
                else:
                    st.warning("Belge okunamadı veya içi boş.")
            
            # Bu dosyanın eklendiğini oturum (session) hafızasına kaydet
            st.session_state.ingested_files.add(uploaded_file.name)
            # Uploader'ı sıfırlamak için key'i artır
            st.session_state.uploader_key += 1
            st.rerun()
        else:
            # Sadece bilgi göster
            st.success(f"{uploaded_file.name} zaten eklendi ve kullanıma hazır!")
            # Eğer kullanıcı dosyayı sildiyse ve uploader hala eski dosyayı tutuyorsa, sıfırla
            st.session_state.uploader_key += 1
            st.rerun()
    st.markdown("---")
    st.header("Yüklü Belgeleri Yönet")
    all_docs = get_all_documents()
    if not all_docs:
        st.info("Veritabanında yüklü belge yok.")
    else:
        for doc in all_docs:
            col1, col2 = st.columns([0.75, 0.25])
            with col1:
                st.caption(f"{doc}")
            with col2:
                with st.popover("×"):
                    st.write("Belge silinsin mi?")
                    if st.button("Evet", key=f"del_doc_{doc}"):
                        delete_document(doc)
                        if doc in st.session_state.ingested_files:
                            st.session_state.ingested_files.remove(doc)
                        st.rerun()
                
    st.markdown("---")
    st.header("Model Ayarları")
    
    # Sadece CUDA (GPU) destekli ve test edilmiş modeller
    available_models = [
        "ministral-3-3b-instruct-2512",
        "smollm3-3b",
        "qwen2.5-1.5b"
    ]
    
    # Eğer mevcut model listede yoksa başa dön
    current_index = available_models.index(st.session_state.chat_model_name) if st.session_state.chat_model_name in available_models else 0
    
    selected_model = st.selectbox(
        "Kullanılacak Chat Modeli (Sadece GPU Destekliler)",
        available_models,
        index=current_index
    )
    
    if selected_model != st.session_state.chat_model_name:
        # Önce eski modeli RAM/VRAM'den temizle (OOM yememek için)
        try:
            manager = FoundryLocalManager.instance
            old_model = manager.catalog.get_model(st.session_state.chat_model_name)
            if old_model.is_loaded:
                old_model.unload()
        except Exception:
            pass
            
        st.session_state.chat_model_name = selected_model
        st.rerun()
        
    if st.button("🔄 Motoru Yeniden Başlat (Reset)", help="Yapay zeka yanıt vermekte takılırsa (veya çökerse) arka plandaki motoru sıfırlayıp hafızayı boşaltır."):
        with st.spinner("Motor sıfırlanıyor, lütfen bekleyin..."):
            try:
                manager = FoundryLocalManager.instance
                # Yüklü tüm modelleri hafızadan at
                for m_id, model_obj in manager.catalog.models.items():
                    if model_obj.is_loaded:
                        model_obj.unload()
            except Exception:
                pass
        st.success("Motor başarıyla sıfırlandı! Tekrar soru sorabilirsiniz.")
        st.rerun()
                    
    st.markdown("---")
    st.header("Dışa Aktar")
    if st.session_state.messages:
        chat_export_text = ""
        for msg in st.session_state.messages:
            role = "Sen" if msg["role"] == "user" else "Asistan"
            chat_export_text += f"[{role}]: {msg['content']}\n\n"
            
        # Mevcut sohbetin başlığını dosya adı yapalım
        export_title = next((c["title"] for c in all_chats if c["id"] == current_chat_id), "sohbet")
        # Dosya adı için geçersiz karakterleri temizle
        safe_title = "".join(c for c in export_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        st.download_button(
            label="Mevcut Sohbeti İndir (.txt)",
            data=chat_export_text,
            file_name=f"{safe_title}.txt",
            mime="text/plain"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 4. KULLANICI SORU SORDUĞUNDA OLACAKLAR
# ─────────────────────────────────────────────────────────────────────────────
# Kullanıcı alttaki kutuya bir şeyler yazıp enter'a basarsa:
user_query = user_query_input

if "trigger_query" in st.session_state:
    user_query = st.session_state.trigger_query
    del st.session_state.trigger_query

if user_query and user_query.strip():
    # Sadece boşluklardan oluşan mesajları önlemek için .strip() kullandık
    
    # Sohbet başlığı 'Yeni Sohbet' ise başlığı kullanıcının sorusuyla güncelle
    current_title = next((c["title"] for c in all_chats if c["id"] == current_chat_id), "Yeni Sohbet")
    if current_title == "Yeni Sohbet":
        new_title = user_query[:25] + "..." if len(user_query) > 25 else user_query
        update_chat_title(current_chat_id, new_title)
    
    # Kullanıcının sorusunu ekrana ekle ve geçmişe kaydet
    with st.chat_message("user"):
        st.markdown(user_query)
    save_message(current_chat_id, "user", user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # --- RAG SÜRECİ BAŞLIYOR ---
    with st.chat_message("assistant"):
        with st.status("> Yapay zeka düşünüyor...", expanded=True) as status:
            st.write("> Veritabanında bağlam aranıyor...")
            # Sohbet hafızasını al (Sadece son sıfırlamadan sonrakileri al, ve en fazla 3 tane)
            active_history = []
            for m in reversed(st.session_state.messages):
                if m["role"] == "system" and m["content"] == "MEMORY_CLEARED":
                    break
                active_history.insert(0, m)
                
            recent_messages = active_history[-4:-1] if len(active_history) > 1 else []
            
            # SOHBET HAFIZASINI ARAMAYA (RETRIEVAL) DAHIL ET:
            # Sadece mevcut soruyu degil, bir onceki soruyu da birlestirerek arama yaparsak
            # baglam kopuklugunu (örn: "Peki onda kim oynuyor?" sorusundaki 'onda'nin ne oldugu) engelleriz.
            search_query = user_query
            past_user_msgs = [m["content"] for m in active_history[:-1] if m["role"] == "user"]
            if past_user_msgs:
                last_user_msg = past_user_msgs[-1][:100]
                search_query = f"{last_user_msg} | {user_query}"
            
            # Eskiden VRAM cöktügü icin (iki model ayni anda GPU'dayken) top_k=2 yapmistik.
            # Embedding CPU'da olsa da, Chat modelinin Context (İçerik) sınırı ve GPU KV Cache'i
            # 6 metin + sohbet geçmişini aynı anda kaldırmayıp çökebiliyor (OOM). 
            # Bu yüzden top_k'yi dengelemek adına 2'ye düşürüyoruz. (Eğer çok yavaşlarsa VRAM taşıyor demektir)
            matches = get_relevant_context(search_query, top_k=2)
            
            # Bulunan metinleri birleştir
            if matches:
                context_text = "\n\n".join([f"[Kaynak: {m['source']}]\n{m['content']}" for m in matches])
            else:
                context_text = "Veritabanında ilgili hiçbir bilgi bulunamadı."

        # Adım B: Modele ne yapacağını söyleyen System Prompt oluştur
        system_prompt = f"""You are a strict RAG extraction assistant. You MUST answer the user's question using ONLY the provided CONTEXT.

CRITICAL RULES:
1. NEVER use outside knowledge. Do not invent answers.
2. If the exact answer is not found in the CONTEXT, you MUST reply with exactly: "[BİLGİ YOK]"
3. Answer in the same language as the user's question (e.g. if the user asks in Turkish, answer in Turkish).

EXAMPLE:
Context: The sky is blue.
User: What color is the grass?
Assistant: [BİLGİ YOK]

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---"""

        # Geçmiş sohbeti User Prompt'unun içine göm
        user_prompt_with_history = "--- PAST CONVERSATION ---\n"
        if recent_messages:
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                # Geçmiş mesaj çok uzunsa kes (Özellikle asistanın önceki uzun cevapları VRAM'i patlatmasın diye)
                safe_content = msg['content'][:400] + "...(truncated)" if len(msg['content']) > 400 else msg['content']
                user_prompt_with_history += f"{role}: {safe_content}\n"
        
        user_prompt_with_history += f"\n--- NEW QUESTION ---\nUser: {user_query}\n\nCRITICAL REMINDER: You MUST NOT use outside knowledge or general definitions. If the exact answer is not in the CONTEXT above, output exactly: '[BİLGİ YOK]'"""

        # Tüm kuralları ve bağlamı TEK BİR KULLANICI MESAJI (user role) olarak birleştir.
        # Küçük yerel modeller (Qwen 1.5B vb.) "system" rolünü desteklemediği için görmezden gelebilir.
        combined_prompt = f"{system_prompt}\n\n{user_prompt_with_history}"

        # Adım C: Mesaj paketini hazırlama (Sadece User)
        chat_payload = [
            {"role": "user", "content": combined_prompt}
        ]
        
        # Adım D: Cevabı Streamlit'e akıtarak (streaming) yazdır
        # st.write_stream, metin geldikçe ekrana yazar
        def generate_response():
            try:
                for chunk in chat_client.complete_streaming_chat(chat_payload):
                    if not chunk.choices: # Stream bitiş sinyali gelirse atla
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content # yield = parçayı anında ekrana yolla
            except Exception as e:
                # Kullanıcı yayını keserse (Stop) hatayı yut
                if "cancel" not in str(e).lower():
                    yield f"\n\n[Sistem Hatası: {str(e)}]"
                    
        # Cevabı ekranda göster
        status.update(label="Yanıt üretiliyor...", state="running")
        full_response = st.write_stream(generate_response)
        status.update(label="Yanıt tamamlandı!", state="complete", expanded=False)
        
        # Cevabın altına kaynakları göster (Kullanıcı hangi belgeden geldiğini görsün)
        if matches and "[BİLGİ YOK]" not in full_response:
            # Küçük modeller metin içinde kaynak ismi vermeyi beceremediği için,
            # aramada bulduğumuz tüm kaynakları doğrudan UI'da gösteriyoruz.
            used_sources = matches
            
            if used_sources:
                # Aynı dosyadan gelen farklı parçaları (chunk'ları) grupla
                grouped_sources = {}
                for m in used_sources:
                    if m['source'] not in grouped_sources:
                        grouped_sources[m['source']] = []
                    grouped_sources[m['source']].append(m)
                    
                with st.expander("Kullanılan Kaynakları İncele"):
                    for source_name, chunks in grouped_sources.items():
                        st.markdown(f"### [ {source_name} ]")
                        for i, chunk in enumerate(chunks):
                            st.info(f"**Bölüm {i+1}** (Benzerlik: {chunk['score']:.4f})\n\n{chunk['content']}")
                        
        # Son olarak modelin cevabını da geçmişe kaydet
        used_sources = [m for m in matches if m['source'] in full_response] if matches else None
        save_message(current_chat_id, "assistant", full_response, sources=used_sources)
        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": used_sources})
        st.session_state.is_generating = False
        st.session_state.regen_locked = False
        st.rerun()

# === OTOMATİK KAYDIRMA (AUTO-SCROLL) ===
# Sayfa her yenilendiğinde (mesaj gidip/geldiğinde) görünümü en aşağıya kaydırır
st.components.v1.html(
    """
    <script>
        const scrollElement = window.parent.document.querySelector('.main');
        if (scrollElement) {
            scrollElement.scrollTo({top: scrollElement.scrollHeight, behavior: 'smooth'});
        }
    </script>
    """,
    height=0
)
