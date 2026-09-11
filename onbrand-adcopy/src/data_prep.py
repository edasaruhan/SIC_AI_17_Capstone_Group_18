import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_and_clean(csv_path: str | Path | None = None, languages: tuple[str, ...] = ("en",)) -> pd.DataFrame:
    if csv_path is None:
        csv_path = DATA_DIR / "Social_Media_Engagement_Dataset.csv"

    df = pd.read_csv(csv_path)
    total = len(df)
    logger.info(f"Loaded {total} rows from {csv_path}")

    if languages is not None:
        n_before = len(df)
        df = df[df["language"].isin(languages)]
        n_after = len(df)
        logger.info(f"Filtered languages={languages}: {n_before} -> {n_after} rows (dropped {n_before - n_after})")

    n_before = len(df)
    df = df[df["impressions"] > 0]
    n_after = len(df)
    logger.info(f"Filtered impressions > 0: {n_before} -> {n_after} rows (dropped {n_before - n_after})")

    df["ctr_proxy"] = (
        (df["likes_count"] + df["shares_count"] + df["comments_count"]) / df["impressions"]
    ) * 100

    clip_value = df["ctr_proxy"].quantile(0.99)
    n_before_clip = len(df)
    df["ctr_proxy"] = df["ctr_proxy"].clip(upper=clip_value)
    n_clipped = (df["ctr_proxy"] == clip_value).sum()
    logger.info(f"99th percentile clip value: {clip_value:.4f}")
    logger.info(f"Rows clipped to ceiling: {n_clipped}")

    n_before = len(df)
    required_cols = ["sentiment_score", "toxicity_score"]
    df = df.dropna(subset=required_cols)
    n_after = len(df)
    logger.info(f"NaN check ({required_cols}): {n_before} -> {n_after} rows (dropped {n_before - n_after})")

    logger.info(f"Final dataset: {len(df)} rows")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Final row count: {len(df)}")
    print(df.head())
