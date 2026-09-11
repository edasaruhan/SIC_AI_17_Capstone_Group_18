# OnBrand AdCopy - Marka Sesi Odaklı, Performansı Ölçülmüş Yapay Zekâ Reklam Metni Üretici

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![ML](https://img.shields.io/badge/Model-XGBoost%20%7C%20Sentence--BERT-orange.svg)
![GenAI](https://img.shields.io/badge/LLM-Google%20Gemini-green.svg)
![UI](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Tests](https://img.shields.io/badge/Tests-63%20passed-brightgreen.svg)
![Status](https://img.shields.io/badge/Samsung%20Innovation%20Campus-AI%20in%20Marketing-blueviolet.svg)

**OnBrand AdCopy**, dijital pazarlama ekipleri için performans odaklı (CTR) reklam metinleri üretirken markanın kurumsal ses tonunu koruyan, düşük uyumlu metinleri kapalı döngü (*closed-loop*) mekanizmasıyla otomatik düzelten ve her skorun arkasını şeffaflık ilkesiyle açıklayan bütünleşik bir yapay zekâ karar destek sistemidir.

Sistem tek bir ekranda: **paralel çoklu üslup üretimi** (Gemini), **metin-içi NLP öznitelikleriyle XGBoost CTR tahmini**, **çok dilli SBERT anlamsal marka uyumu** ve **LLM tabanlı otomatik yeniden yazım** döngüsünü birleştirir.

---

## ✨ Neden OnBrand AdCopy?

- **4 ayrı üslupta paralel üretim** - Duygusal, Bilgilendirici, Aciliyet ve Mizahi varyantlar aynı anda üretilir (`ThreadPoolExecutor(4)`).
- **Marka ses ciddiyeti** - Samsung, Duolingo ve Nike için elle küratörlü referans metinleri ve stil kuralları; yeni markalar sihirbazla tek tıkla eklenebilir.
- **Tutarlı ve açıklanabilir skorlar** - Her kartta CTR Benchmark (`/100`) ve SBERT Marka Uyumu (`/100`) mutlak olarak gösterilir; **Bileşik Skor bu iki sayının ağırlıklı ortalamasının ta kendisidir** - hesap makinesiyle çarptığınızda bile tutar.
- **Kota dayanıklılığı** - Ücretsiz API kotası dolduğunda sistem model fallback zincirine otomatik geçer (`gemini-2.5-flash` → `gemini-3.5-flash-lite` → `gemini-3.5-flash`).
- **Kapalı döngü otomatik düzeltme** - Marka uyumu eşiğin altında kalan metinler LLM ile yeniden yazılır; öncesi/sonrası karşılaştırması kartta gösterilir.
- **Önbellek ve hız** - Thread-safe SQLite önbelleği, `@st.cache_data` ile anında yüklenen doğrulama tabloları.

---

## 🧠 Problem ve Çözüm

- **Problem:** Performans odaklı reklam metinleri (tıklama tuzakları, aşırı aciliyet vurgusu) zamanla şirketin kurumsal marka sesinden sapar. Geleneksel kılavuz kontrolleri ve manuel denetimler yavaş, öznel ve ölçeklenemezdir.
- **Çözüm:** Çoklu üslupta reklam metni üretimi, XGBoost tabanlı CTR tahmini, Sentence-BERT semantik marka uyum skorlaması ve LLM tabanlı otomatik yeniden yazım döngüsünü tek bir platformda birleştirmek - ve tüm bunları her skoru savunulabilir kılacak şekilde sertifikalandırmak.

---

## ⚙️ Sistem Mimarisi ve İş Akışı

```text
[Kullanıcı Girdisi: Ürün, Hedef Kitle, Marka]
                  │
                  ▼
     [1. LLM Çoklu Üslup Üretimi - PARALEL (ThreadPoolExecutor × 4)]
              │ (Duygusal · Bilgilendirici · Aciliyet · Mizahi)
              ▼
     [2. NLP Öznitelik Çıkarımı]      [3. SBERT Vektör Gömme]
        │   (16 metin-içi özellik)        │
        ▼                               ▼
  [XGBoost CTR Tahmini]           [Kosinüs Benzerliği (Marka Uyumu)]
        │                               │
        │          [ctr_benchmark]      │          [brand_score]
        └──────────────┬────────────────┘
                       ▼
      [4. Bileşik Skor = w·Benchmark + (1−w)·Marka]
                       ▼
           ┌───────────┴───────────┐
           ▼                       ▼
  [5. LLM Otomatik Düzeltme]  [6. Yayına Hazır Metinler]
      (uyum < %70 ise)            (uyum ≥ %70 ise)
           │
           ▼
  [Yeniden skorlama + öncesi/sonrası gösterimi]
```

---

## 📊 Keşifsel Veri Analizi (EDA) Çıktıları

`Social Media Engagement Dataset` (1.197 gerçek İngilizce satır) üzerinde yapılan analizlerin öne çıkan bulguları:

- **Metin Uzunluğu vs. CTR:** Kelime sayısının etkileşim üzerinde hafif negatif eğilimi var ($r \approx -0.04$). Kısa ve vurucu metinler daha yüksek performans gösteriyor.
- **Aciliyet Paradoksu:** "Hemen", "Kaçırma", "Sınırlı" gibi agresif aciliyet ifadeleri içeren gönderilerin ortalama CTR'ı (%15.95), içermeyenlere (%21.51) kıyasla **daha düşük**. Kullanıcılar itip-kakıcı tonu cezalandırıyor.
- **Korelasyon Analizi:** Dilbilgisel özniteliklerin CTR ile doğrusal korelasyonu düşük → doğrusal olmayan modellerin (XGBoost) tercih edilmesini doğrular.

| Kelime Sayısı vs CTR | Özniteliklerin CTR Etkisi | Korelasyon Isı Haritası |
|:---:|:---:|:---:|
| ![Word Count vs CTR](reports/eda_figures/word_count_vs_ctr.png) | ![Feature Impact](reports/eda_figures/ctr_by_features.png) | ![Correlation Heatmap](reports/eda_figures/correlation_heatmap.png) |

---

## 🛠️ Yapay Zekâ ve Skorlama Moturu

### CTR Tahmini (XGBoost)

Metnin 16 metin-içi NLP özniteliği (duygu, aciliyet, biçimlendirme, CTA, kelime çeşitliliği, ...) üzerinden etkileşim potansiyeli tahmin edilir.

| Model | Kapsam | Test R² | RMSE |
|---|---|---|---|
| XGBoost (full öznitelikler) | metin + istatistiksel öznitelikler | **0.871** | 22.7 |
| Linear (karşılaştırma) | aynı öznitelikler | 0.900 | 20.1 |
| XGBoost (content-only) | yalnızca metin-içi öznitelikler | −0.05 (bulgu) | 64.9 |

> **Bilimsel dürüstlük notu:** Metin tek başına, gösterim/takipçi bilgisi olmadan etkileşimin yalnızca bir kısmını açıklar. Content-only R² değerinin düşük olması başarısızlık değil; metin → etkileşim sinyalinin şablon tekrarlı verideki zayıflığının **dürüstçe raporlanan bir bulgusudur** (Bai et al. 2025 ile uyumlu). Bu yüzden skorlar ham olarak değil, logaritmik kalibre edilmiş **CTR Benchmark** ile sunulur:

```
ctr_benchmark = clip( ln(1 + raw_ctr) / ln(1 + 35) × 100, 10, 98 )
```

### Marka Uyumu (SBERT)

Çok dilli `paraphrase-multilingual-MiniLM-L12-v2` ile metnin, markanın onaylı referans reklamlarına maksimum kosinüs benzerliği 0–100 ölçekte skorlanır. Markalar arası temel sapma (~0.30) doğrusal kalibrasyonla ayarlanır ve skor asla tam 0.0 dönmez (`SCORE_FLOOR=0.1`).

| Marka | Kendi referansı ort. | Farklı marka ort. | Karşıtlık |
|---|---|---|---|
| Samsung | 100.0 | 14.8 | %85+ |
| Duolingo | 100.0 | 12.7 | %87+ |
| Nike | 100.0 | 17.9 | %82+ |

### Bileşik Skorlama

Kartta görünen **iki mutlak skorun** kullanıcı seçimli ağırlıklı ortalaması:

```
Bileşik = w_ctr × CTR_Benchmark + (1 − w_ctr) × Marka_Uyumu
```

Örnek: `0.6 × 61.0 + 0.4 × 74.5 = 66.4` - ekranda tam olarak bu gösterilir; batch-içi göreceli uçurumlar (ör. 10/95) üretilmez.

---

## 🔁 Kapalı Döngü Otomatik Düzeltme

Üretilen bir varyantın marka uyumu eşiğin altındaysa (`%70`), LLM marka stil kurallarına ve referanslarına dayanarak metni yeniden yazar; uyum ve CTR yeniden hesaplanır:

- Düzeltme **marka uyumunu artırırken CTR düşüşünü %5 ile sınırlar**.
- Kartta *"↺ Marka tonuna göre yeniden yazıldı"* rozeti + öncesi/sonrası karşılaştırması gösterilir.
- CTR modelinde yeniden yazım hâlâ düşük uyumluysa metin **insan onayına** işaretlenir (başarısız senaryoda otomatik kabul edilmez).

---

## 🚀 Hızlı Başlangıç (Sıfırdan Reproduksiyon)

```bash
# 1) Sanal ortam
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # macOS / Linux

# 2) Bağımlılıklar (requirements.txt = pip freeze çıktısı, sabitlenmiş)
pip install -r requirements.txt

# 3) API anahtarı (zorunlu · ücretsiz)
#    https://aistudio.google.com/apikey adresinden al
cp .env.example .env            # GOOGLE_API_KEY=... doldur

# 4) Ana CTR modelini eğit (yalnızca gerçek veriyle)
python scripts/train_main_model.py

# 5) (İsteğe bağlı) Marka uyumu kanıt tablosu + insan doğrulama CSV
python scripts/p1_brand_proof.py
python scripts/p2_human_eval.py

# 6) Testler (63 geçmeli)
pytest tests/ -v

# 7) Arayüz
streamlit run app/streamlit_app.py
```

> **SBERT modeli** ilk çalıştırmada otomatik indirilir (~470 MB, `SBERT_MODEL` ile değiştirilebilir). CTR modeli, öznitelik mühendisliği ve marka skorlaması tamamen **yerelde** çalışır; API çağrısı yalnızca üretim/düzeltme adımlarında yapılır.

---

## 📦 Kullanım Senaryoları

```bash
# Pipeline'ı Python'da çalıştır (API anahtarı gerekir)
python -c "from src.pipeline import run_pipeline; \
import json; print(json.dumps(run_pipeline('Galaxy S24 Ultra', 'teknoloji meraklısı 25-35', 'samsung'), ensure_ascii=False, indent=2))"

# Yeni marka ekleme (UI): kenar çubuğu → "Yeni Marka Ekle (Sihirbaz)"
# marka_reference.json'a yazılır, st.rerun() ile dropdown'da anında görünür.
```

Gerçek pipeline çıktısı senaryoları: `reports/demo_scenarios.json`.

---

## 🧪 Test ve Raporlama Altyapısı

**63 test**, 9 ayrı dosyada:

| Modül | Kapsam |
|---|---|
| `test_data_prep` | temizleme, duplikasyon, dil filtresi |
| `test_features` | 16 özniteliğin çıkarımı (TR/EN) |
| `test_sentiment` | duygu / toksiklik puanlama |
| `test_brand_alignment` | skor asla 0.0 değil, kanıt tablosu |
| `test_ctr_model` | gerçek-veri ana model, demo artefakt ayrımı |
| `test_composite` | benchmark monoktonluğu, mutlak-skora birebir bileşik |
| `test_cache` | SQLite thread safety |
| `test_generator_retry` | kota/fallback/üstel geri çekilme |
| `test_pipeline_end_to_end` | uçtan uca akış |

### Otomatik üretilen raporlar

- `reports/evaluation_report.md` - P0–P4 yol haritasının kriter değerlendirmesi ve test sonuçları
- `reports/model_card.md` - model kartı (veri, öznitelikler, sınırlılıklar)
- `reports/brand_alignment_proof.csv` - P1.1 marka uyumu 10-metin kanıtı
- `reports/sbert_human_eval.csv` - 15 metinlik insan doğrulama seti (`human_label` sütunu elle doldurulup `p2_analyze_human.py` ile analiz edilir)
- `reports/ablation_results.csv` - LR/RF/XGB × özellik alt kümesi
- `reports/demo_scenarios.json` - gerçek üretim senaryoları
- `models/metrics.json` - eğitim metrikleri (R², RMSE, MAE, öznitelik önemi)

---

## 🛡️ Güvenlik ve Dirençlilik

- **API anahtarı gizli:** `.env` git-ignored; key hiçbir kodda sabit değil, `src/config.py` üzerinden yüklenir.
- **Kota yönetimi:** Ücretsiz katman günde ~20 istek/model. Sistem 429/404/5xx hatalarını sınıflandırır, **model fallback zincirine** geçer ve geçici hatalarda üstel geri çekilme (2→4→8 sn) uygular.
- **Thread-safe önbellek:** SQLite (`cache/generations.db`) `Lock` + `check_same_thread=False` ile eşzamanlı erişime dayanıklıdır; tekrar eden istekler API'yi harcamaz.
- **UI dayanıklılığı:** Model dosyası eksikse uyarı, üretim hatalarında Türkçe ve aksiyon alınabilir mesajlar.

---

## 📁 Depo Yapısı

```bash
onbrand-adcopy/
├── app/
│   ├── streamlit_app.py          # Arayüz (mutlak skor kartları, marka sihirbazı, Altair grafik)
│   ├── theme.css                 # marka stil teması
│   └── assets/                   # logo
├── src/
│   ├── config.py                 # .env yükleme + model/fallback zinciri
│   ├── generator.py              # paralel üretim + retry + fallback
│   ├── composite.py              # sklearn: benchmark + ağırlıklı bileşik skor
│   ├── ctr_model.py              # XGBoost eğitim/tahmin (ana + demo sabitleri)
│   ├── brand_alignment.py        # SBERT marka uyumu + doğrulama
│   ├── autocorrect.py            # LLM kapalı döngü düzeltme
│   ├── data_prep.py / features.py / sentiment.py
│   ├── cache.py                  # thread-safe SQLite önbelleği
│   └── pipeline.py               # uçtan uca orkestrasyon
├── scripts/
│   ├── train_main_model.py       # ana model (gerçek veri) + metrics.json
│   ├── build_model.py            # (demo) sentetik veri + ayrı artefakt
│   ├── p1_brand_proof.py         # marka uyumu kanıt tablosu
│   ├── p2_human_eval.py          # insan doğrulama seti üretimi
│   ├── p2_analyze_human.py       # etiket analizi
│   └── ablation.py               # LR/RF/XGB ablasyonu
├── data/
│   ├── brand_reference.json                      # marka kuralları ve referansları
│   ├── Social_Media_Engagement_Dataset.csv       # GERÇEK ana eğitim verisi
│   └── synthetic_ad_performance.csv              # (demo) sentetik, ayrı
├── models/                       # XGBoost artefaktları + metrics.json + feature_importance*
├── reports/                      # evaluation_report, model_card, kanıt CSV/JSON'ları
├── notebooks/01_eda.ipynb        # keşifsel veri analizi
├── cache/generations.db          # SQLite önbellek (çalışma zamanı)
├── tests/                        # 63 test
├── requirements.txt              # pip freeze (sabitlenmiş)
├── .env.example
└── baslat.bat                    # tek tıkla başlatma
```

---

## 💡 Sık Sorulanlar

- **"Neden CTR Benchmark, direkt CTR vermiyorsunuz?"** - Ham CTR tahmini metinler arasında çok düşük kontrastlıdır (%5–12). Logaritmik kalibrasyon bunu tanıdık 0–100 puana taşır ve `clip(., 10, 98)` ile aşırı uçları yumuşatır.
- **"Yeni marka nasıl eklerim?"** - Arayüzde kenar çubuğu → *Yeni Marka Ekle (Sihirbaz)* → ID, ad, ton ve ≥3 referans metni gir → *Markayı Kaydet*.
- **"Sentetik demo verisi gerçek sistemde kullanılıyor mu?"** - Hayır. `src/` yalnızca gerçek CSV ile eğitilen ana modeli kullanır; sentetik model ayrı artefakt (`synthetic_demo_*`) olarak durur, varsayılan kapalıdır.
- **"API kotam bitti, çalışır mı?"** - Üretim dışındaki her şey (skorlama, marka uyumu, analiz) yerelde çalışır. Üretim için sistem fallback zinciriyle diğer modellenmiş modellere geçer; kota gece yarısı (UTC) sıfırlanır.

---
