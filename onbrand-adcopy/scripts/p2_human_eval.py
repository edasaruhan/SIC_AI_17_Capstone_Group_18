"""P2.2 — SBERT marka uyumu insan doğrulama iskeleti.

reports/sbert_human_eval.csv üretir: her metin için 3 marka skoru + hangi markanın
en yüksek skor aldığı (SBERT tahmini) ve doldurulmayı bekleyen 'human_label'
sütunu. human_label doldurulduktan sonra scripts/p2_analyze_human.py ile
SBERT tahmini vs insan etiketi korelasyon/AUC hesabı yapılır.

15 metnin kaynağı bilinir (source_brand) — SBERT tahmininin kaynakla uyumu,
"insan etiketi"nin makul bir vekili olarak da raporlanır.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.brand_alignment import brand_alignment_score, load_brand_data  # noqa: E402

# P1.1'deki 10 metne ek 5 metin. Hepsinde kaynak marka bilinir.
TEXTS = [
    ("samsung", "en", "Meet Galaxy AI — the phone that learns how you live. Trade in and upgrade today."),
    ("duolingo", "en", "Got five minutes? That's all it takes. Keep the streak, keep the language."),
    ("nike", "en", "Just do it — today's run sets tomorrow's finish line."),
    ("samsung", "en", "The Fold unfolds more than the screen. Productivity, reimagined."),
    ("duolingo", "en", "Hoot hoot! Fifteen minutes a day — no boring textbooks, just wins."),
    ("nike", "en", "You don't run a race to finish second. Own the course, own the day."),
    ("samsung", "tr", "Galaxy AI ile tanış: senin hayatını öğrenen telefon. Takas fırsatı bugünle sınırlı."),
    ("duolingo", "tr", "Beş dakikan mı var? Bu kadarı yeter. Serini sürdür, kaçırma."),
    ("nike", "tr", "Sadece yap. Bugünkü koşu, yarınki bitiş çizgisi."),
    ("samsung", "tr", "Katlanır ekran, katlanmayan beklentiler. Galaxy Z serisi gerçekten farklı."),
    ("duolingo", "tr", "Her gün biraz Türkçe? Bir baykuş sana hatırlatır. 15 dakika, bir seri."),
    ("nike", "tr", "Rekorlar gece çalışanların değil, her sabah dışarı çıkanlarındır."),
    ("samsung", "en", "Your phone should keep up, not the other way around. Galaxy S series."),
    ("duolingo", "en", "A language a year — start tonight, streak slowly becomes a habit."),
    ("nike", "tr", "Ayakkabıyı giydin. Şimdi o kapıyı aç ve koşmaya başla."),
]
BRANDS = ["samsung", "duolingo", "nike"]


def main():
    brand_data = load_brand_data()
    rows = []
    for i, (source, lang, text) in enumerate(TEXTS, start=1):
        scores = {b: brand_alignment_score(text, b, brand_data) for b in BRANDS}
        predicted = max(scores, key=scores.get)
        rows.append({
            "idx": i,
            "source_brand": source,
            "language": lang,
            "text": text,
            "score_samsung": round(scores["samsung"], 2),
            "score_duolingo": round(scores["duolingo"], 2),
            "score_nike": round(scores["nike"], 2),
            "sbert_predicted_brand": predicted,
            "sbert_correct": predicted == source,
            "human_label": "",  # kullanıcı tarafından doldurulur (samsung/duolingo/nike)
        })

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "reports" / "sbert_human_eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    acc = df["sbert_correct"].mean()
    print(f"Kaydedildi: {out}  ({len(df)} metin)")
    print(f"SBERT tahmini → kaynak marka uyumu: {acc:.0%} "
          f"({int(df['sbert_correct'].sum())}/{len(df)})")
    print("İnsan doğrulaması için human_label sütununu doldurun, "
          "ardından scripts/p2_analyze_human.py çalıştırın.")


if __name__ == "__main__":
    main()