"""
==========================================================
  LOCAL RAG ASSISTANT -- Hafta 1 Testi
  "Hello Model" -- Embedding + Chat modeli yukle ve test et
==========================================================

Bu dosya ne yapiyor?

  1. Foundry Local SDK'yi baslatir (AI motorunu hazirlar)
  2. Bir "embedding modeli" indirir ve yukler
     -> Embedding: Metni sayi dizisine (vektore) ceviren model
     -> Bunu RAG'de "anlam arama" icin kullanacagiz
  3. Birkaç cumleyi embed eder - sonuclari ekrana yazar
  4. Bir "chat modeli" indirir ve yukler
     -> Chat modeli: Sorulari cevaplayabilen LLM (GPT gibi, ama yerel)
  5. Basit bir soru sorar ve cevabi ekrana yazar

Bu test, tum sistemin duzgun kuruldugunu dogrular.
"""

import math
from foundry_local_sdk import Configuration, FoundryLocalManager


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 1: Yardimci Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """
    Cosine Similarity (Kosinus Benzerligi) hesaplar.

    Ne ise yarar?
    Iki vektorun ne kadar "benzer yonde" oldugunu olcer.
    Sonuc 0.0 ile 1.0 arasindadir:
      -> 1.0 = tamamen ayni anlam
      -> 0.0 = hic ilgisi yok

    RAG'de sunun yapacagiz:
      Kullanicinin sorusunu vektore cevir,
      belgelerle karsilastir,
      en yuksek skoru alan belgeyi al.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 2: Ana Program
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LOCAL RAG ASSISTANT -- Hafta 1 Kurulum Testi")
    print("=" * 60)

    # ── ADIM 1: SDK Baslatma ─────────────────────────────────────────────────
    #
    # Ne yapiyor?
    # FoundryLocalManager, Foundry Local'in "kontrol merkezi"dir.
    # Modelleri indirip yonetmekten, yukleyip calistirmaktan sorumludur.
    # app_name, modellerin saklandigi klasoru belirler (cache dizini).
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/5] SDK baslatiliyor...")
    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    print("      [OK] SDK baslatildi!")

    # ── ADIM 2: Embedding Modeli ─────────────────────────────────────────────
    #
    # Embedding modeli nedir?
    # Metni alir ve onu binlerce sayidan olusan bir vektore donusturur.
    # Ornegin: "Kedi sut icer" -> [0.23, -0.87, 0.11, 0.56, ...]
    # Benzer anlamli cumleler birbirine yakin vektorler uretir.
    # Bu sayede "anlam bazli arama" yapabiliriz.
    #
    # Model: qwen3-embedding-0.6b
    # -> "qwen3": Model ailesi (Alibaba'nin acik kaynak modeli)
    # -> "embedding": Bu modelin ozelligi (embed etmek icin)
    # -> "0.6b": 0.6 milyar parametre (kucuk ve hizli)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/5] Embedding modeli hazirlaniyor...")
    print("      Model: qwen3-embedding-0.6b (~500 MB)")
    print("      (Ilk calistirmada indirilir, sonraki cache'den yuklenir)")

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\r      Indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print("\r      [OK] Embedding modeli indirildi!            ")

    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()
    print("      [OK] Embedding modeli bellege yuklendi!")

    # ── ADIM 3: Embedding Testi ───────────────────────────────────────────────
    #
    # Ne yapiyor?
    # 3 farkli cumleyi embed ediyoruz (vektore ceviriyoruz).
    # Sonra aralarindaki cosine similarity'yi hesapliyoruz.
    # Beklenti: Ilk iki cumle birbiriyle daha cok ilgili olmali.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/5] Embedding testi yapiliyor...")

    test_sentences = [
        "Artificial intelligence models can run on a computer.",
        "LLM models can be used without an internet connection.",
        "The weather is very nice and sunny today.",
    ]

    response = embedding_client.generate_embeddings(test_sentences)
    embeddings = [item.embedding for item in response.data]

    print(f"      Vektor boyutu: {len(embeddings[0])} boyutlu")

    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\n      Benzerlik Testi:")
    print(f"      Cumle 1: AI modelleri hakkinda")
    print(f"      Cumle 2: LLM (dil modeli) hakkinda")
    print(f"      Cumle 3: Hava durumu hakkinda")
    print(f"\n      Cumle1 <-> Cumle2 (ikisi de AI hakkinda): {sim_1_2:.4f}")
    print(f"      Cumle1 <-> Cumle3 (biri AI, biri hava):   {sim_1_3:.4f}")

    if sim_1_2 > sim_1_3:
        print("\n      [OK] Beklenen sonuc: AI cumleleri birbirine daha yakin!")
    else:
        print("\n      [!] Beklenmedik sonuc, ama test tamamlandi.")

    # ── ADIM 4: Chat Modeli ───────────────────────────────────────────────────
    #
    # Chat modeli nedir?
    # Kullanicinin sorularini anlayan ve metin uretebilen LLM.
    # GPT-4 gibi dusun, ama tamamen senin bilgisayarinda calisiyor.
    # RAG'de soyle kullanacagiz:
    #   "Su belgelere bakarak bu soruyu cevapla" diyecegiz.
    #
    # Model: qwen2.5-0.5b
    # -> 500 milyon parametre (cok kucuk, hizli ama sinirli)
    # -> Ileride phi-3.5-mini gibi daha buyuk modele gecebiliriz
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/5] Chat modeli hazirlaniyor...")
    print("      Model: qwen2.5-0.5b (~400 MB)")

    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.download(
        lambda p: print(f"\r      Indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print("\r      [OK] Chat modeli indirildi!            ")

    chat_model.load()
    chat_client = chat_model.get_chat_client()
    print("      [OK] Chat modeli belleğe yuklendi!")

    # ── ADIM 5: Chat Testi ────────────────────────────────────────────────────
    #
    # Ne yapiyor?
    # Chat modeline bir soru soruyoruz.
    # messages listesi: Konusma gecmisi gibi dusun.
    #   "system" -> Modele "sen kimsin, nasil davranacaksin" talimati
    #   "user"   -> Kullanicinin sorusu
    #
    # Bu yapi RAG'in temelini olusturuyor. Ileride system mesajina
    # belgelerden getirilen baglami (context) ekleyecegiz.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5/5] Chat testi yapiliyor...")
    print("      Soru: 'What is RAG? Explain briefly.'")
    print("      Cevap: ", end="", flush=True)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Answer briefly and clearly."
        },
        {
            "role": "user",
            "content": "What is RAG (Retrieval-Augmented Generation)? Explain in 2-3 sentences."
        }
    ]

    for chunk in chat_client.complete_streaming_chat(messages):
        # Son chunk bos gelebilir (stream bitis sinyali) -- None kontrolu
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

    print("\n")

    # ── Temizlik ──────────────────────────────────────────────────────────────
    #
    # Neden unload ediyoruz?
    # Modeller RAM'de yer kaplar. Isimiz bitince bellegi serbest birakiyoruz.
    # Ancak model dosyalari disk'te (cache) kalir -- tekrar indirmek gerekmez.
    # ─────────────────────────────────────────────────────────────────────────
    print("Modeller bellekten kaldiriliyor...")
    embedding_model.unload()
    chat_model.unload()

    print("=" * 60)
    print("  [OK] TUM TESTLER BASARILI!")
    print("  Foundry Local, Embedding ve Chat modeli calisiyor.")
    print("  Siradaki adim: Belge tabanli RAG pipeline'i kurmak.")
    print("=" * 60)


if __name__ == "__main__":
    main()