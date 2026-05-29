"""
Visualization module — generates all figures for EDA and model evaluation.
All plots are saved to output/figures/ and also returned as Figure objects
so Streamlit can render them directly without re-reading from disk.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, FIGURES_DIR, METRICS_DIR, CATEGORIES

logger = setup_logging("visualization")

sns.set_theme(style="whitegrid", font_scale=1.0)

PALETTE = {
    "intern_level":       "#FF6B6B",
    "junior_level":       "#FFA07A",
    "mid_level":          "#4ECDC4",
    "senior_level":       "#45B7D1",
    "lead_architect":     "#2C3E50",
    "template_boilerplate": "#95E1D3",
    "low_value":          "#AAAAAA",
}


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {path}")


# ── Category distribution ─────────────────────────────────────────────────────

def plot_category_distribution(df: pd.DataFrame, label_col: str = "weak_label") -> plt.Figure:
    col = label_col if label_col in df.columns else "category"
    counts = df[col].value_counts().reindex(CATEGORIES, fill_value=0)
    colors = [PALETTE.get(c, "#999") for c in counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels([c.replace("_", "\n") for c in counts.index], rotation=0, fontsize=8)
    ax1.set_title("Repository Category Distribution", fontweight="bold")
    ax1.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=8)

    non_zero = [(c, v) for c, v in zip(counts.index, counts.values) if v > 0]
    ax2.pie(
        [v for _, v in non_zero],
        labels=[c.replace("_", "\n") for c, _ in non_zero],
        colors=[PALETTE.get(c, "#999") for c, _ in non_zero],
        autopct="%1.1f%%", startangle=140, pctdistance=0.82,
    )
    ax2.set_title("Category Proportions", fontweight="bold")

    plt.tight_layout()
    _save(fig, "category_distribution.png")
    return fig


# ── Signal box-plots ──────────────────────────────────────────────────────────

def plot_signal_boxplots(df: pd.DataFrame, label_col: str = "weak_label") -> plt.Figure:
    col = label_col if label_col in df.columns else "category"
    signals = ["stars_count", "contributors_count", "weekly_commit_freq",
               "releases_count", "readme_length", "closed_prs_count"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, sig in enumerate(signals):
        ax = axes[i]
        if sig not in df.columns:
            ax.set_visible(False)
            continue
        data, labels_plot = [], []
        for cat in CATEGORIES:
            subset = df[df[col] == cat][sig].dropna()
            if len(subset):
                cap = subset.quantile(0.95)
                data.append(subset.clip(upper=cap).values)
                labels_plot.append(cat.replace("_", "\n"))

        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, cat in zip(bp["boxes"], [c for c in CATEGORIES if len(df[df[col] == c][sig].dropna())]):
            patch.set_facecolor(PALETTE.get(cat, "#999"))
            patch.set_alpha(0.75)
        ax.set_xticklabels(labels_plot, fontsize=7)
        ax.set_title(sig.replace("_", " ").title(), fontweight="bold", fontsize=10)
        ax.set_ylabel("Value")

    plt.suptitle("Repository Signals by Engineering Level", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "signal_boxplots.png")
    return fig


# ── Boolean signal heatmap ────────────────────────────────────────────────────

def plot_bool_signals(df: pd.DataFrame, label_col: str = "weak_label") -> plt.Figure:
    col = label_col if label_col in df.columns else "category"
    bool_cols = ["has_ci_cd", "has_tests", "has_license", "is_fork", "is_template"]
    available = [c for c in bool_cols if c in df.columns]
    if not available:
        return plt.figure()

    heat = pd.DataFrame(
        {sig: [df[df[col] == cat][sig].mean() * 100 for cat in CATEGORIES] for sig in available},
        index=CATEGORIES,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax,
                linewidths=0.5, vmin=0, vmax=100,
                xticklabels=[s.replace("_", " ").title() for s in available],
                yticklabels=[c.replace("_", "\n") for c in CATEGORIES])
    ax.set_title("% of Repositories with Boolean Signal by Category", fontweight="bold")
    plt.tight_layout()
    _save(fig, "bool_signals_heatmap.png")
    return fig


# ── Feature correlation ───────────────────────────────────────────────────────

def plot_feature_importance(df: pd.DataFrame, label_col: str = "weak_label") -> plt.Figure:
    col = label_col if label_col in df.columns else "category"
    maturity_order = {
        "intern_level": 0, "junior_level": 1, "mid_level": 2,
        "senior_level": 3, "lead_architect": 4,
        "template_boilerplate": 1.5, "low_value": -1,
    }
    df2 = df.copy()
    df2["_m"] = df2[col].map(maturity_order)

    features = [
        "stars_count", "forks_count", "contributors_count", "weekly_commit_freq",
        "releases_count", "readme_length", "closed_prs_count",
        "has_ci_cd", "has_tests", "has_license", "topics_count",
    ]
    available = [f for f in features if f in df2.columns]
    corrs = []
    for feat in available:
        try:
            corr = df2[[feat, "_m"]].dropna().corr().iloc[0, 1]
            if not np.isnan(corr):
                corrs.append((feat, corr))
        except Exception:
            pass

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    names = [c[0].replace("_", " ").title() for c in corrs]
    values = [c[1] for c in corrs]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson Correlation with Engineering Maturity Level")
    ax.set_title("Signal Correlation with Repository Maturity", fontweight="bold")
    for bar, val in zip(bars, values):
        offset = 0.02 if val >= 0 else -0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=9)
    plt.tight_layout()
    _save(fig, "feature_importance.png")
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict = None) -> plt.Figure:
    if metrics is None:
        mp = METRICS_DIR / "evaluation_results.json"
        if not mp.exists():
            return plt.figure()
        with open(mp) as f:
            metrics = json.load(f)

    cm = np.array(metrics.get("confusion_matrix", []))
    if cm.size == 0:
        return plt.figure()

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
        xticklabels=[c.replace("_", "\n") for c in CATEGORIES],
        yticklabels=[c.replace("_", "\n") for c in CATEGORIES],
        linewidths=0.4, vmin=0, vmax=1,
    )
    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_macro", 0)
    ax.set_title(f"Normalized Confusion Matrix  (Accuracy={acc:.3f}, F1-macro={f1:.3f})",
                 fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _save(fig, "confusion_matrix.png")
    return fig


# ── Per-class bar chart ───────────────────────────────────────────────────────

def plot_per_class_metrics(metrics: dict = None) -> plt.Figure:
    if metrics is None:
        mp = METRICS_DIR / "evaluation_results.json"
        if not mp.exists():
            return plt.figure()
        with open(mp) as f:
            metrics = json.load(f)

    pc = metrics.get("per_class", {})
    if not pc:
        return plt.figure()

    cats = list(pc.keys())
    prec = [pc[c]["precision"] for c in cats]
    rec  = [pc[c]["recall"]    for c in cats]
    f1   = [pc[c]["f1"]        for c in cats]
    x = np.arange(len(cats))
    w = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, prec, w, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x,     rec,  w, label="Recall",    color="#4CAF50", alpha=0.85)
    ax.bar(x + w, f1,   w, label="F1",        color="#FF9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics: Precision / Recall / F1", fontweight="bold")
    ax.legend()
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save(fig, "per_class_metrics.png")
    return fig


# ── All-at-once ───────────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame, metrics: dict = None):
    plot_category_distribution(df)
    plot_signal_boxplots(df)
    plot_bool_signals(df)
    plot_feature_importance(df)
    if metrics:
        plot_confusion_matrix(metrics)
        plot_per_class_metrics(metrics)
    plt.close("all")
    logger.info("All plots generated.")


if __name__ == "__main__":
    from src.utils import LABELED_DIR
    df = pd.read_csv(LABELED_DIR / "repos_labeled.csv")
    generate_all_plots(df)
