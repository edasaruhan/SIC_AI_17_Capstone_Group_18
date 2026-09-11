"""P2.2 — İnsan etiketi vs SBERT tahmini analizi.

reports/sbert_human_eval.csv doluysa (human_label dolu veya kaynak doğrusu
olarak kullanılıyorsa) çalıştırılır:
- kaynak marka doğruluğu (accuracy)
- SBERT skoru ile ikili "doğru marka" göstergesi arasındaki korelasyon (nokta-biserial)
- ROC-AUC (SBERT skoru, doğru-etiket ayırt ediciliği)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    csv = Path(__file__).resolve().parent.parent / "reports" / "sbert_human_eval.csv"
    df = pd.read_csv(csv)

    # İnsan etiketi yoksa kaynak markayı "insan doğruluğu" vekili olarak kullan.
    use_human = bool(df["human_label"].fillna("").astype(str).str.strip().any())
    truth_col = "human_label" if use_human else "source_brand"
    truth = df[truth_col].astype(str).str.strip().map(lambda v: v.lower())

    # Metnin kendi gerçek markasına aldığı skoru ve en yüksek skoru al.
    own_scores, max_scores, correct = [], [], []
    for _, row in df.iterrows():
        scores = {b: row[f"score_{b}"] for b in ["samsung", "duolingo", "nike"]}
        own_scores.append(scores[truth[row.name]])
        max_scores.append(max(scores.values()))
        correct.append(row["sbert_predicted_brand"].lower() == truth[row.name])

    own_scores = np.array(own_scores)
    max_scores = np.array(max_scores)
    correctness = np.array(correct, dtype=int)

    acc = correctness.mean()
    tau = np.corrcoef(own_scores, correctness)[0, 1]
    # ROC-AUC yalnızca iki sınıf ayrışıyorsa anlamlıdır.
    auc = None
    if len(np.unique(correctness)) == 2 and np.sum(correctness) > 0:
        auc = roc_auc_score(correctness, own_scores)

    print(f"Etiket kaynağı: {'human_label' if use_human else 'source_brand (insan vekili)'}")
    print(f"Doğruluk (toplam):          {acc:.1%} ({int(correctness.sum())}/{len(df)})")
    print(f"SBERT own-skoru ↔ doğruluk korelasyonu (nokta-biserial):  {tau:.3f}")
    print(f"ROC-AUC (own skoru):        {'%.3f' % auc if auc is not None else 'N/A'}")
    print(f"Ortalama own-skor (doğru):  {own_scores[correctness == 1].mean():.1f}  "
          f"yanlış: {own_scores[correctness == 0].mean():.1f}")
    if auc is None:
        print("Not: tek sınıf ayrışımı yok; AUC hesaplamak için daha fazla etiket gerekir.")


if __name__ == "__main__":
    main()