"""P1.1 — Marka uyumu skorunun canlılık kanıtı.

10 üretilmiş reklam metnine (5 Türkçe, 5 İngilizce) ait marka uyumu skorlarını
hesaplar ve reports/brand_alignment_proof.csv olarak kaydeder.

Doğrulamak istediğimiz iddia: hiçbir gerçek metin tam 0.0 almaz, tüm skorlar
> 0'dır ve her metnin kaynak markası, diğer markalara göre daha yüksek skor alır.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.brand_alignment import brand_alignment_score, load_brand_data  # noqa: E402

# Kaynak markası bilinen 10 metin (metinler LLM tarzında üretilmiş örneklerdir;
# referans kopyalarının kendisi DEĞİLDİR — böylece skor döngüsel olarak yüksek çıkmaz).
TEXTS = [
    ("samsung", "en", "Meet Galaxy AI — the phone that learns how you live. Trade in and upgrade today."),
    ("duolingo", "en", "Got five minutes? That's all it takes. Keep the streak, keep the language."),
    ("nike", "en", "Just do it — today's run sets tomorrow's finish line."),
    ("samsung", "en", "The Fold unfolds more than the screen. Productivity, reimagined."),
    ("duolingo", "en", "Hoot hoot! Fifteen minutes a day — no boring textbooks, just wins."),
    ("samsung", "tr", "Galaxy AI ile tanış: senin hayatını öğrenen telefon. Takas fırsatı bugünle sınırlı."),
    ("duolingo", "tr", "Beş dakikan mı var? Bu kadarı yeter. Serini sürdür, kaçırma."),
    ("nike", "tr", "Sadece yap. Bugünkü koşu, yarınki bitiş çizgisi."),
    ("samsung", "tr", "Katlanır ekran, katlanmayan beklentiler. Galaxy Z serisi gerçekten farklı."),
    ("duolingo", "tr", "Her gün biraz Türkçe? Bir baykuş sana hatırlatır. 15 dakika, bir seri."),
]
BRANDS = ["samsung", "duolingo", "nike"]


def main():
    brand_data = load_brand_data()
    rows = []
    failures = []
    for source, lang, text in TEXTS:
        scores = {b: brand_alignment_score(text, b, brand_data) for b in BRANDS}
        own = scores[source]
        others = {b: s for b, s in scores.items() if b != source}
        max_other = max(others.values())
        rows.append({
            "text": text,
            "language": lang,
            "source_brand": source,
            **{f"score_{b}": round(scores[b], 2) for b in BRANDS},
            "own_score": round(own, 2),
            "max_cross_score": round(max_other, 2),
            "own_higher_than_cross": own > max_other,
        })
        if own <= 0:
            failures.append((source, lang, "own_score <= 0"))
        if not own > max_other:
            failures.append((source, lang, "own not higher than cross"))

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "reports" / "brand_alignment_proof.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    all_positive = bool((df[["score_samsung", "score_duolingo", "score_nike"]] > 0).all().all())

    print("\n=== P1.1 Marka uyumu canlılık kanıtı ===")
    for r in rows:
        flag = "OK " if r["own_higher_than_cross"] and r["own_score"] > 0 else "FAIL"
        print(f"  [{r['language']}] {flag} {r['source_brand']:>8} own={r['own_score']:6.2f} "
              f"cross={r['max_cross_score']:6.2f} | {r['text'][:55]}...")
    print(f"\nHepsi > 0: {all_positive}")
    print(f"Kaynak marka > diğerleri: {not failures} ({len(rows) - len(failures)}/{len(rows)})")
    print(f"Tablo kaydedildi: {out}")
    if failures:
        print("Hatalar:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()