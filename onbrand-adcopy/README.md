# OnBrand AdCopy

AI-powered ad copy generator that creates brand-aligned, engagement-optimized ad copy variants.

## Setup

```bash
pip install -r requirements.txt
```

Set the Gemini API key: create a key at https://aistudio.google.com/apikey,
then copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY=...`.

> `gemini-3.6-flash` (via `google-genai`) is used for generation and
> auto-correction. The model name is configurable with `GEMINI_MODEL` in `.env`
> (default `gemini-3.6-flash`). Generation is called with default parameters —
> passing temperature/max_output_tokens on this model triggers its thinking mode
> and truncates the answer. The key is treated as secret: it is loaded from
> `.env` (git-ignored) via `src/config.py`, never hardcoded. The CTR model,
> SBERT brand scorer (`paraphrase-multilingual-MiniLM-L12-v2`, overridable with
> `SBERT_MODEL` — downloads once on first run) and feature engineering all run
> locally with no API calls.

## Model architecture (P0.1 — bilimsel dürüstlük)

**Ana model** yalnızca gerçek Kaggle verisiyle (`Social_Media_Engagement_Dataset.csv`,
1.197 İngilizce satır) eğitilir — `src/` içinde sentetik veri referansı yoktur.

| Model | Veri | Content R² | Not |
|---|---|---|---|
| Ana (gerçek veri) | Social_Media_Engagement_Dataset.csv | **−0.05** | Düşük kontrast, dürüst sınırlılık (bulgu) |
| Demo (sentez) | synthetic_ad_performance.csv (3500 EN/TR) | 0.808 | Varsayılan kapalı, ayrı artefakt |

`src/` modülleri (`predict_ctr`, `composite_score`, pipeline) yalnızca ana
gerçek-veri modelini kullanır. Sentetik "demo modu" ayrı dosyalarda (`src/synthetic_demo_model.py`
olmadan — `scripts/build_model.py` ile üretilir ve `DEMO_CONTENT_MODEL` olarak
`ctr_model.py` içinde okunur), UI'da açıkça etiketli ve varsayılan kapalıdır.

Brand alignment: `paraphrase-multilingual-MiniLM-L12-v2` SBERT; “asla 0.0 dönmeyen”
kalibrasyon (`SCORE_FLOOR=0.1`); P1.1 kanıtı: `reports/brand_alignment_proof.csv`.

## Run

```bash
# Ana modeli eğit (yalnızca gerçek veriyle, content + full)
python scripts/train_main_model.py

# (İsteğe bağlı) Demo modeli eğit (sentez veri, ayrı artefaktlar)
python scripts/build_model.py --rows 3500

# P1.1 — Marka uyumu 10-metin kanıtı tablosu
python scripts/p1_brand_proof.py

# P2.1 — Ablasyon tablosu (gerçek veri, LR/RF/XGB × özellik alt kümeleri)
python scripts/ablation.py

# P2.2 — SBERT insan doğrulama CSV (15 metin, human_label boş)
python scripts/p2_human_eval.py

# P2.2 — İnsan etiketi doldurulduktan sonra analiz
python scripts/p2_analyze_human.py

# EDA
jupyter notebook notebooks/01_eda.ipynb

# Pipeline (API anahtarı gerektirir)
python -c "from src.pipeline import run_pipeline; print(run_pipeline('Ürün', 'hedef', 'samsung'))"

# Streamlit app
streamlit run app/streamlit_app.py

# Testler (60 geçmeli, P0/P1/P2 regresyon testleri dahil)
pytest tests/ -v
```

## Sıfırdan reprodüksiyon (P3.4)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
# SBERT modeli ilk çalıştırmada otomatik indirilir (~470 MB)
cp .env.example .env           # GOOGLE_API_KEY ekle
python scripts/train_main_model.py
python scripts/p1_brand_proof.py
pytest tests/ -v
streamlit run app/streamlit_app.py
```

`requirements.txt` = `pip freeze` çıktısıdır; bir platformdaritos.js/debug
atlanabilir. Python 3.11+ gerekir (test Python 3.14 ile çalışıyor).
