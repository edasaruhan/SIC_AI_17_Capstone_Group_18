# Sıkça Sorulan Sorular — OnBrand AdCopy

**Oluşturulma:** 2026-09-11 · Samsung Innovation Campus AI in Marketing, Group 18

---

### S1: CTR modelinin R²'si neden çok düşük? Model bozuk mu?

Hayır, model bozuk değil. Gerçek veri (`Social_Media_Engagement_Dataset.csv`) üzerinde content-only (yalnızca metin) modelin R² ≈ −0.05 çıkması, veride metin → etkileşim ilişkisinin istatistiksel olarak çok zayıf olduğunun **bulgusudur**, bir hata değildir. Kısa metinli sosyal medya gönderilerinde etkileşim kestiriminin zorluğu literatürde de bilinmektedir (Bai et al., 2025). Sistem bu sınırlılığı dürüstçe raporlar.

---

### S2: Raporla tutarlı mı? Eski R²=0.81 nerede?

0.81 R²'si, demosunua özel olarak üretilmiş sentetik verideki bir **kuralı öğrenme** başarısıdır (`synthetic_demo_xgboost_content_model.pkl`). Bu model ayrı, açıkça etiketli ve varsayılan kapalıdır; ana/pipeline modeli değildir. Her iki sonuç da `evaluation_report.md` §2'de明確 olarak belgelenmiştir.

---

### S3: Sentetik veri neden	var? Hangi model kullanılıyor?

Kural-veri modeli, **ölçüm doğrulaması** (health check) içindir: text feature pipeline'ın сколько wirklich kuralı öğrenip öğrenmediğini, frp.training/inference uyumunu doğrular. Ana model gerçek veriden eğitilir; sentetik model yalnızca "gösterim modu"ndadır.

---

### S4: Sıralama neden min-max normalizasyonu yapıyor?

Ham CTR skorları (gerçek veride ~5-8 aralığında) çok düşük kontrast üretir. Varyantların birbirine göre sıralamasını sağlamak için, **aynı istekte üretilen varyant grubu içinde** min-max normalizasyonu uygulanır (P1.2). Ham değerler kartta ayrıca gösterilir — manipülasyon değildir.

---

### S5: Neden 4 üslup var? Nedenriminator?

Pazarlama literatüründe ve tasarımında "duygusal", "bilgilendirici", "aciliyet" ve "mizahi" ana üslup kategorileri olarak yaygın kabul görmektedir (Aaker 1996, Keller 2013). Sistem variedáslar üretir; kullanıcı slider ile performans/marka ağırlığını ayarlayarak sıralamayı kontrol eder.

---

### S6: LLM maliyetli mi? API anahtarı gerekli mi?

Gemini 3.6-flash ücretsiz katmanı günde 20 istek sunar; proje suresince yeterlidir. `cache/generations.db` (SQLite) aynı (model, prompt) çiftleri için HTTP çağrısını atlayarak maliyeti ve gecikmeyi azaltır. Anahtar `.env`'de tutulur; `.gitignore` ile versiyon kontrolünden dışlanır.

---

### S7: Marka uyumu skoru nasıl calibre edilir?

SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) ile cosine similarity hesaplanır; off-brand baseline (≈0.30 max-sim) ile tam eşleşme (1.0) arasına lineer kalibrasyon uygulanır. `SCORE_FLOOR=0.1` sayesinde hiçbir metin tam 0.0 almaz (P1.1). Gerçek 10-metin kanıtı: `reports/brand_alignment_proof.csv`.

---

### S8: Ağırlık neden %60/40?

Performans (CTR) ağırlığının marka uyumundan hafif baskın olduğu varsayılan, konsept notundaki "performans odaklı" senaryoyu yansıtır. Kullanıcı slider ile 0–100 arası istediği gibi ayarlayabilir; bu deneysel bir karardır.

---

### S9: Gerçek-world'de kullanabilir miyim?

Bu bir prototip/tez projesidir. Gerçek dünya kullanıma geçiş için:
1. Gerçek marka verisiyle fine-tuning (veya domain-specific engagement verisi)
2. İnsan-değerlendirmeli A/B test doğrulaması
3. Düşük R²'nin real-world yansıması analiz edilmeli (opsiyon: context features ekleme)
4. API kotası ve maliyet yönetimi

---

### S10: Sıfırdan nasıl çalıştırırım?

`README.md` "Sıfırdan reprodüksiyon" bölümündeki 7 adımı izleyin. SBERT modeli ilk çalıştırmada otomatik indirilir. Python 3.11+ gerekir.