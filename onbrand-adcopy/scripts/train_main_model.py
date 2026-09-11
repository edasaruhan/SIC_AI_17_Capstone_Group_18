"""Ana CTR modelini yalnızca gerçek CSV verisiyle eğitir.

P0.1: Raporlanan/ana model bu script ile eğitilir.
Sentetik 'gösterim modu' modeli scripts/build_model.py ile ayrı eğitilir.
models/metrics.json tüm eğitim metriklerini kaydeder.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import load_and_clean
from src.features import add_features
from src.ctr_model import train_ctr_model, MODELS_DIR

if __name__ == "__main__":
    print("Gerçek veri yükleniyor...")
    df = load_and_clean()  # varsayılan: Social_Media_Engagement_Dataset.csv, dil=en
    df = add_features(df)
    print(f"Eğitim verisi: {len(df)} satır (gerçek CSV)")
    results = train_ctr_model(df)  # output_prefix="" → ana model dosyaları

    cm = results["content_metrics"]
    fm = results["xgb_metrics"]
    lm = results["lr_metrics"]
    print("\n=== Ana Model (gerçek veri) ===")
    print(f"Full:   test R²={fm['r2']:.4f}, RMSE={fm['rmse']:.4f}")
    print(f"Linear: test R²={lm['r2']:.4f}, RMSE={lm['rmse']:.4f}")
    print(f"Content:test R²={cm['r2']:.4f}, RMSE={cm['rmse']:.4f}, CV R²={results['content_cv_r2_mean']:.4f}")
    print("Modeller models/ dizinine kaydedildi.")

    # Görev 4: metrics.json çıktısı
    metrics = {
        "dataset": "Social_Media_Engagement_Dataset.csv",
        "n_rows": len(df),
        "full_xgb": {"r2": fm["r2"], "rmse": fm["rmse"], "mae": fm["mae"]},
        "full_linear": {"r2": lm["r2"], "rmse": lm["rmse"], "mae": lm["mae"]},
        "content_xgb": {"r2": cm["r2"], "rmse": cm["rmse"], "mae": cm["mae"]},
        "content_cv_r2_mean": results["content_cv_r2_mean"],
        "content_importances": results["content_feature_importances"],
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrikler kaydedildi: {metrics_path}")
