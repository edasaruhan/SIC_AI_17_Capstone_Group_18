import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_prep import load_and_clean
from src.features import add_features, FEATURE_COLS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "eda_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_and_clean()
df = add_features(df)

# Figure 1: Word count vs CTR proxy scatter + OLS trend
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(df["word_count"], df["ctr_proxy"], alpha=0.3, s=10, edgecolors="none")
z = np.polyfit(df["word_count"], df["ctr_proxy"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["word_count"].min(), df["word_count"].max(), 100)
ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f"OLS trend (slope={z[0]:.2f})")
ax1.set_xlabel("Word Count")
ax1.set_ylabel("CTR Proxy (%)")
ax1.set_title("Word Count vs. CTR Proxy")
ax1.legend()
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "word_count_vs_ctr.png", dpi=150)
plt.close(fig1)
print("Saved: word_count_vs_ctr.png")

# Figure 2: Bar chart - mean CTR for has_question and has_urgency
fig2, axes = plt.subplots(1, 2, figsize=(10, 5))

q_means = df.groupby("has_question")["ctr_proxy"].mean()
axes[0].bar(["No Question", "Has Question"], [q_means.get(0, 0), q_means.get(1, 0)], color=["#4C72B0", "#DD8452"])
axes[0].set_ylabel("Mean CTR Proxy (%)")
axes[0].set_title("Mean CTR by Has Question")

u_means = df.groupby("has_urgency")["ctr_proxy"].mean()
axes[1].bar(["No Urgency", "Has Urgency"], [u_means.get(0, 0), u_means.get(1, 0)], color=["#55A868", "#C44E52"])
axes[1].set_ylabel("Mean CTR Proxy (%)")
axes[1].set_title("Mean CTR by Has Urgency")

fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "ctr_by_features.png", dpi=150)
plt.close(fig2)
print("Saved: ctr_by_features.png")

# Figure 3: Correlation heatmap
model_features = ["char_count", "word_count", "has_question", "has_exclamation", "urgency_count", "sentiment_score", "toxicity_score", "ctr_proxy"]
corr_matrix = df[model_features].corr()
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax3, square=True)
ax3.set_title("Feature Correlation Heatmap")
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig3)
print("Saved: correlation_heatmap.png")

print(f"\nEDA complete. Figures saved to {OUTPUT_DIR}")
