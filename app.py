"""
GitHub Repository Intelligence — Streamlit Application
4 tabs: Problem & Methodology | EDA | Model Results | Interactive Explorer
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.utils import (
    BASE_DIR, LABELED_DIR, SPLITS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR,
    CATEGORIES, CATEGORY_DESCRIPTIONS, ID2LABEL, LABEL2ID,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Repository Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_labeled_df():
    path = LABELED_DIR / "repos_labeled.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics():
    path = METRICS_DIR / "evaluation_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_comparison():
    path = METRICS_DIR / "baseline_comparison.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def run_pipeline():
    """Run the full data → label → train → evaluate pipeline."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "run_pipeline.py")],
        capture_output=True, text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def predict_single(row_dict: dict, label_col="weak_label") -> str:
    """Rule-based prediction for interactive tab (no model load needed)."""
    from src.llm_labeling import rule_based_label
    return rule_based_label(row_dict)["category"]


def predict_with_model(texts: list) -> list:
    """Load best available model and predict a list of texts."""
    sklearn_path = MODELS_DIR / "sklearn_baseline.pkl"
    bert_dir = MODELS_DIR / "bert_classifier"

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if bert_dir.exists():
            tok = AutoTokenizer.from_pretrained(str(bert_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
            model.eval()
            preds = []
            for i in range(0, len(texts), 8):
                batch = texts[i: i + 8]
                enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    out = model(**enc)
                preds.extend(out.logits.argmax(-1).tolist())
            return [ID2LABEL[p] for p in preds]
    except Exception:
        pass

    if sklearn_path.exists():
        import joblib
        pipe = joblib.load(str(sklearn_path))
        return [ID2LABEL[p] for p in pipe.predict(texts)]

    # fallback rule-based
    from src.llm_labeling import rule_based_label
    from src.summarization import build_repo_summary
    return [rule_based_label({"text_summary": t})["category"] for t in texts]


CATEGORY_COLORS = {
    "intern_level": "#FF6B6B", "junior_level": "#FFA07A",
    "mid_level": "#4ECDC4", "senior_level": "#45B7D1",
    "lead_architect": "#2C3E50", "template_boilerplate": "#95E1D3",
    "low_value": "#AAAAAA",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔬 Repo Intelligence")
    st.markdown("**Track A** — Hiring Repository Intelligence")
    st.divider()

    labeled_path = LABELED_DIR / "repos_labeled.csv"
    model_path = MODELS_DIR / "sklearn_baseline.pkl"
    bert_path = MODELS_DIR / "bert_classifier"
    metrics_path = METRICS_DIR / "evaluation_results.json"

    status_data = "✅ Data ready" if labeled_path.exists() else "❌ No data"
    status_model = (
        "✅ BERT trained" if bert_path.exists()
        else ("✅ Sklearn trained" if model_path.exists() else "❌ No model")
    )
    status_metrics = "✅ Metrics ready" if metrics_path.exists() else "❌ No metrics"

    st.markdown(f"**Pipeline status**\n\n{status_data}\n\n{status_model}\n\n{status_metrics}")
    st.divider()

    if st.button("▶ Run Full Pipeline", use_container_width=True):
        with st.spinner("Running pipeline (this may take a few minutes)…"):
            ok, log = run_pipeline()
        if ok:
            st.success("Pipeline completed!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Pipeline failed — check logs.")
            with st.expander("Log"):
                st.text(log[:3000])

    st.divider()
    st.caption("Model: DistilBERT (or sklearn fallback)\nData: GitHub API / synthetic\nLabels: DeepSeek / rule-based")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Problem & Methodology",
    "📊 Exploratory Analysis",
    "🤖 Model Results",
    "🔍 Interactive Explorer",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Problem & Methodology
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Problem & Methodology")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Project Objective")
        st.markdown(
            """
This system evaluates GitHub repositories and classifies them according to the
**engineering maturity** they reflect — not the developer personally, but the
project itself. The goal is to help:

- 🏢 **Recruiters** quickly screen portfolios
- 🚀 **Startups** identify strong open-source candidates
- 🎓 **Accelerators** assess technical quality of applicants
- 🔍 **Engineering managers** benchmark candidate projects

The system is built on a **weak supervision** approach:
1. Collect repository metadata from the GitHub API
2. Generate text summaries from metadata signals
3. Use a cheap LLM (DeepSeek) to create *weak labels*
4. Fine-tune a DistilBERT classifier on those labels
5. Evaluate and iterate
            """
        )

        st.subheader("Engineering Maturity Categories")
        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            color = CATEGORY_COLORS.get(cat, "#999")
            st.markdown(
                f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
                f'color:white;font-size:0.85em">**{cat}**</span> — {desc}',
                unsafe_allow_html=True,
            )
            st.markdown("")

    with col2:
        st.subheader("GitHub Signals Used")
        signals = {
            "⭐ stars_count": "Community popularity proxy",
            "🍴 forks_count": "Reuse & adoption signal",
            "👥 contributors_count": "Team size / collaboration",
            "🔄 weekly_commit_freq": "Development velocity",
            "⚙️ has_ci_cd": "DevOps maturity",
            "📝 readme_length": "Documentation quality",
            "🔖 releases_count": "Versioning discipline",
            "🔀 closed_prs_count": "Code-review culture",
            "🐛 open_issues_count": "Community engagement",
            "🧪 has_tests": "Quality assurance",
            "📅 repo_age_days": "Project longevity",
            "🕐 last_activity_days": "Project recency",
            "🌐 language_count": "Technical breadth",
            "📜 has_license": "Open-source maturity",
            "🏷️ topics_count": "Discoverability / SEO",
        }
        for sig, desc in signals.items():
            st.markdown(f"- **{sig}**: {desc}")

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Prompt Strategy (LLM Labeling)")
        st.markdown(
            """
**System message**: Defines all 7 categories with precise boundaries to reduce
ambiguity. Uses formal engineering language so the LLM applies domain knowledge.

**User message**: Includes the full text summary (all 15 signals in natural
language), forcing the LLM to reason from evidence rather than repo name alone.

**Design choices**:
- Temperature = 0.1 → low label variance
- JSON mode → machine-parseable output
- Confidence field → filter uncertain labels (< 0.55 dropped)
- Synthetic data bypasses LLM → uses ground-truth labels directly
            """
        )

        st.subheader("Dataset Construction")
        st.markdown(
            """
| Stage | Method |
|---|---|
| Collection | GitHub REST API or synthetic generator |
| Sampling | Stratified across all 7 categories |
| Text repr. | Structured paragraph from metadata |
| Labeling | DeepSeek Chat → rule-based fallback |
| Split | 70 / 15 / 15 (train / val / test) |
| Filter | Drop labels with confidence < 0.55 |
            """
        )

    with col4:
        st.subheader("Limitations")
        st.markdown(
            """
**Data limitations**
- Synthetic data may not capture all real-world edge cases
- LLM labels contain noise (estimated 10-20% error rate on ambiguous repos)
- GitHub API rate limits constrain real-data collection speed

**Model limitations**
- DistilBERT truncates at 256 tokens — very long summaries are cut
- Categories like `junior_level` and `mid_level` have overlapping signals
- `template_boilerplate` is hard to distinguish from `low_value` without repo content

**Ethical considerations**
- The system evaluates the *repository*, not the *person*
- Developer demographics are not captured or used
- Results should supplement, not replace, human technical review
- Gaming risk: motivated actors can inflate stars/forks artificially
            """
        )

        st.subheader("Repository Selection Strategy")
        st.markdown(
            """
Repositories were selected to cover all 7 maturity categories through
targeted GitHub search queries:

- **High maturity**: `stars:>1000 language:Python has:topics`
- **Mid-level**: `stars:50..500 language:Python has:topics`
- **Junior/Intern**: `stars:0..10 tutorial homework`
- **Templates**: `is:template`
- **Low value**: `stars:0 pushed:<2022`

Synthetic data uses empirically tuned distributions based on known patterns
in open-source software (e.g., typical fork ratios, commit frequencies).
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Exploratory Analysis
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Exploratory Data Analysis")

    df = load_labeled_df()

    if df.empty:
        st.warning("No data found. Click **▶ Run Full Pipeline** in the sidebar to generate it.")
        st.stop()

    label_col = "weak_label" if "weak_label" in df.columns else "category"

    # ── Dataset overview ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Repositories", f"{len(df):,}")
    col2.metric("Categories", df[label_col].nunique())
    col3.metric("Avg Stars", f"{df['stars_count'].mean():.0f}" if "stars_count" in df.columns else "N/A")
    col4.metric("With CI/CD", f"{df['has_ci_cd'].mean()*100:.0f}%" if "has_ci_cd" in df.columns else "N/A")

    st.divider()

    # ── Category distribution ─────────────────────────────────────────────────
    st.subheader("Category Distribution")
    st.markdown(
        "_Why this chart_: Understanding label balance is critical. Severe imbalance "
        "biases the classifier toward majority classes, requiring class-weighted loss."
    )
    from src.visualization import (
        plot_category_distribution, plot_signal_boxplots,
        plot_bool_signals, plot_feature_importance,
    )
    fig = plot_category_distribution(df, label_col)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── Signal distributions ──────────────────────────────────────────────────
    st.subheader("Signal Distributions by Category")
    st.markdown(
        "_Why these plots_: Box plots reveal whether each signal separates maturity "
        "levels. Strong separation → useful discriminating feature for the model."
    )
    fig2 = plot_signal_boxplots(df, label_col)
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # ── Boolean signals ───────────────────────────────────────────────────────
    st.subheader("Boolean Signal Presence by Category (%)")
    st.markdown(
        "_Why this heatmap_: Boolean signals (CI/CD, tests, license) are strong "
        "discriminators — senior/lead repos nearly always have them; intern repos almost never do."
    )
    fig3 = plot_bool_signals(df, label_col)
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()

    # ── Feature correlation ───────────────────────────────────────────────────
    st.subheader("Signal Correlation with Engineering Maturity")
    st.markdown(
        "_Why this chart_: Pearson correlation shows which raw signals are most "
        "predictive. Contributors and stars are strongly positive; last_activity_days "
        "is negative (more inactive = lower maturity)."
    )
    fig4 = plot_feature_importance(df, label_col)
    st.pyplot(fig4)
    plt.close(fig4)

    st.divider()

    # ── Numeric stats table ───────────────────────────────────────────────────
    st.subheader("Signal Statistics by Category")
    numeric_signals = ["stars_count", "forks_count", "contributors_count",
                       "weekly_commit_freq", "releases_count", "readme_length"]
    available = [s for s in numeric_signals if s in df.columns]
    if available:
        stats = (
            df.groupby(label_col)[available]
            .median()
            .round(1)
            .rename(columns=lambda c: c.replace("_count", "").replace("_", " ").title())
        )
        st.dataframe(stats.style.background_gradient(cmap="YlOrRd", axis=0), use_container_width=True)
        st.caption("Values shown are medians per category.")

    # ── Analytical questions ──────────────────────────────────────────────────
    with st.expander("📌 Analytical Questions (required)"):
        st.markdown(
            """
**Q1 — Maturity signals**

| Level | Key signals |
|---|---|
| Intern | stars≈0, solo, no CI/CD, no tests, minimal README |
| Junior | stars<30, 1-3 contributors, partial README, occasional commits |
| Mid | CI/CD + tests, 3-12 contributors, regular commits, some releases |
| Senior | 100+ stars, 5+ contributors, extensive README, CI/CD + tests + releases |
| Lead | 1000+ stars, 20+ contributors, weekly releases, full DevOps |

**Q2 — Low-value vs boilerplate detection**

*Low-value*: stars≤2, no PRs, no releases, inactive >1 year, README <200 chars.
*Boilerplate*: fork-to-star ratio > 3, template flag, minimal commits after creation,
README present but not customized.

**Q3 — Business value**

- Recruiters: quickly triage large portfolio batches, focus human review on high-signal repos
- Startups: benchmark founder's OSS work before first technical conversation
- Accelerators: score technical portfolio as part of application process
- Interview process: use predicted maturity as a conversation starter, not a gate

**Q4 — Methodological sensitivity**

Changing category definitions (e.g., merging *intern* + *junior* into one class)
reduces confusion at the boundary but loses precision for recruiters who care
about the difference. Alternative prompt framing (asking the LLM to score 1-10
instead of classify) yields more nuanced weak labels at the cost of needing
calibration before use.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Results
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Model Results")

    metrics = load_metrics()
    comparison = load_comparison()

    if not metrics:
        st.warning("No evaluation metrics found. Run the pipeline first.")
        st.info("The pipeline trains a DistilBERT (or sklearn TF-IDF) classifier and evaluates on the held-out test set.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader(f"Model: {metrics.get('model_used', 'Unknown')}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.3f}")
    c2.metric("Precision", f"{metrics.get('precision_macro', 0):.3f}")
    c3.metric("Recall",    f"{metrics.get('recall_macro', 0):.3f}")
    c4.metric("F1 Macro",  f"{metrics.get('f1_macro', 0):.3f}")
    c5.metric("F1 Weighted", f"{metrics.get('f1_weighted', 0):.3f}")

    st.caption(
        "Metrics computed on the held-out 15% test set. "
        "Macro averages weight all classes equally regardless of support."
    )
    st.divider()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion Matrix")
        from src.visualization import plot_confusion_matrix, plot_per_class_metrics
        fig_cm = plot_confusion_matrix(metrics)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_b:
        st.subheader("Per-Class F1 / Precision / Recall")
        fig_pc = plot_per_class_metrics(metrics)
        st.pyplot(fig_pc)
        plt.close(fig_pc)

    st.divider()

    # ── Per-class table ───────────────────────────────────────────────────────
    st.subheader("Per-Class Performance Table")
    pc = metrics.get("per_class", {})
    if pc:
        pc_df = pd.DataFrame(pc).T.round(3)
        pc_df.index.name = "Category"
        st.dataframe(
            pc_df.style.background_gradient(cmap="RdYlGn", subset=["f1"]),
            use_container_width=True,
        )

    st.divider()

    # ── Baseline comparison ───────────────────────────────────────────────────
    st.subheader("Baseline vs. Model Comparison")
    if comparison:
        baseline = comparison.get("baseline", {})
        model_c = comparison.get("model", {})
        if baseline and model_c:
            comp_rows = []
            for met in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
                comp_rows.append({
                    "Metric": met.replace("_", " ").title(),
                    "Majority-Class Baseline": round(baseline.get(met, 0), 3),
                    metrics.get("model_used", "Model"): round(model_c.get(met, 0), 3),
                })
            comp_df = pd.DataFrame(comp_rows).set_index("Metric")
            st.dataframe(comp_df.style.highlight_max(axis=1, color="#d4f1b4"), use_container_width=True)
    else:
        st.info("Run `python src/evaluation.py` to generate baseline comparison.")

    st.divider()

    # ── Error analysis ────────────────────────────────────────────────────────
    st.subheader("Top Misclassifications (Error Analysis)")
    errors = metrics.get("top_errors", [])
    if errors:
        err_df = pd.DataFrame(errors)
        st.dataframe(err_df, use_container_width=True)
        st.markdown(
            """
**Common error patterns**:
- `intern_level` ↔ `junior_level`: Both have low stars and few contributors — the model
  relies heavily on CI/CD and test presence to separate them.
- `mid_level` ↔ `senior_level`: Boundary is fuzzy; the model tends to under-predict
  `senior_level` for repos with moderate stars but strong CI/CD.
- `template_boilerplate` ↔ `low_value`: Both have low activity; the fork-ratio
  signal helps the model, but synthetic data limits realism here.
            """
        )
    else:
        st.info("No error data available.")

    # ── Methodology notes ─────────────────────────────────────────────────────
    with st.expander("📌 Methodological notes"):
        st.markdown(
            """
**Why DistilBERT?**
DistilBERT retains ~97% of BERT's performance at 40% fewer parameters and 60% faster inference,
making it ideal for quick fine-tuning on a laptop without a strong GPU.

**Why TF-IDF + Logistic Regression as baseline?**
It captures keyword patterns in the summary text (e.g., "no CI/CD", "20+ contributors")
and runs without GPU, making it a strong and fast fallback.

**Weak supervision rationale**
Manual annotation of 500+ repos is expensive and subjective. Using DeepSeek (or rule-based
heuristics) as the initial annotator gives us scalable, consistent labels — even if imperfect.
The downstream BERT model can generalise beyond the annotation noise.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Interactive Explorer
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.header("Interactive Repository Explorer")

    df = load_labeled_df()
    has_data = not df.empty
    label_col = "weak_label" if "weak_label" in df.columns else "category"

    # ── Section A: Filter & Browse Dataset ───────────────────────────────────
    st.subheader("Browse Dataset")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_cats = st.multiselect(
            "Filter by Category",
            options=CATEGORIES,
            default=CATEGORIES,
        )
    with col_f2:
        min_stars = st.number_input("Min Stars", min_value=0, value=0, step=1)
    with col_f3:
        max_stars = st.number_input("Max Stars", min_value=0, value=100000, step=100)

    if has_data and selected_cats:
        filtered = df[
            (df[label_col].isin(selected_cats)) &
            (df["stars_count"] >= min_stars) &
            (df["stars_count"] <= max_stars)
        ]

        st.markdown(f"**{len(filtered):,} repositories** match your filters.")

        display_cols = [
            "full_name", label_col, "stars_count", "forks_count",
            "contributors_count", "has_ci_cd", "has_tests", "releases_count",
            "weekly_commit_freq", "last_activity_days",
        ]
        show_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols].rename(columns={label_col: "category"}).reset_index(drop=True),
                     use_container_width=True, height=300)

        if st.checkbox("Show text summary for first result"):
            if "text_summary" in filtered.columns and len(filtered):
                st.text(filtered["text_summary"].iloc[0])

    elif not has_data:
        st.warning("No data loaded. Run the pipeline first.")

    st.divider()

    # ── Section B: Predict a custom repository ────────────────────────────────
    st.subheader("Predict Repository Category")
    st.markdown("Enter repository characteristics and get an engineering maturity prediction.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        p_name = st.text_input("Repository name", value="my-awesome-project")
        p_desc = st.text_input("Description", value="A REST API backend service")
        p_lang = st.selectbox("Main language", ["Python", "JavaScript", "TypeScript", "Go", "Java", "Rust", "Other"])
        p_stars = st.number_input("Stars", min_value=0, value=50, step=1)
        p_forks = st.number_input("Forks", min_value=0, value=10, step=1)
    with col_p2:
        p_contributors = st.number_input("Contributors", min_value=1, value=3, step=1)
        p_commit_freq = st.number_input("Commits/week (avg)", min_value=0.0, value=5.0, step=0.5)
        p_has_ci = st.checkbox("Has CI/CD workflows", value=True)
        p_has_tests = st.checkbox("Has test suite", value=True)
        p_has_license = st.checkbox("Has license", value=True)
    with col_p3:
        p_open_issues = st.number_input("Open issues", min_value=0, value=10, step=1)
        p_closed_prs = st.number_input("Closed PRs", min_value=0, value=25, step=1)
        p_releases = st.number_input("Releases", min_value=0, value=5, step=1)
        p_readme_len = st.number_input("README length (chars)", min_value=0, value=3000, step=100)
        p_age_days = st.number_input("Repository age (days)", min_value=1, value=365, step=1)
        p_last_activity = st.number_input("Days since last commit", min_value=0, value=7, step=1)

    if st.button("🔮 Predict Category", type="primary"):
        row_dict = {
            "name": p_name, "description": p_desc, "main_language": p_lang,
            "stars_count": p_stars, "forks_count": p_forks,
            "contributors_count": p_contributors, "weekly_commit_freq": p_commit_freq,
            "has_ci_cd": int(p_has_ci), "has_tests": int(p_has_tests),
            "has_license": int(p_has_license), "open_issues_count": p_open_issues,
            "closed_prs_count": p_closed_prs, "releases_count": p_releases,
            "readme_length": p_readme_len, "repo_age_days": p_age_days,
            "last_activity_days": p_last_activity, "topics_count": 3,
            "language_count": 1, "is_fork": 0, "is_template": 0,
        }

        from src.summarization import build_repo_summary
        summary = build_repo_summary(row_dict)
        row_dict["text_summary"] = summary

        # Try trained model first
        preds = predict_with_model([summary])
        predicted_cat = preds[0] if preds else predict_single(row_dict)

        color = CATEGORY_COLORS.get(predicted_cat, "#999")
        st.markdown(
            f'<div style="background:{color};padding:16px;border-radius:8px;'
            f'color:white;font-size:1.3em;font-weight:bold;text-align:center">'
            f'Predicted Category: {predicted_cat}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Definition**: {CATEGORY_DESCRIPTIONS.get(predicted_cat, '')}")
        with st.expander("View generated text summary (model input)"):
            st.text(summary)

    st.divider()

    # ── Section C: Category examples ─────────────────────────────────────────
    st.subheader("Representative Examples per Category")
    if has_data:
        chosen_cat = st.selectbox("Select a category to view examples", CATEGORIES)
        examples = df[df[label_col] == chosen_cat].head(3)
        for _, row in examples.iterrows():
            with st.expander(f"📁 {row.get('full_name', row.get('name', 'unknown'))}"):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown(f"**Stars**: {int(row.get('stars_count', 0)):,}")
                    st.markdown(f"**Forks**: {int(row.get('forks_count', 0)):,}")
                    st.markdown(f"**Contributors**: {int(row.get('contributors_count', 1))}")
                    st.markdown(f"**Commit freq**: {row.get('weekly_commit_freq', 0):.1f}/week")
                with col_e2:
                    st.markdown(f"**CI/CD**: {'Yes' if row.get('has_ci_cd') else 'No'}")
                    st.markdown(f"**Tests**: {'Yes' if row.get('has_tests') else 'No'}")
                    st.markdown(f"**Releases**: {int(row.get('releases_count', 0))}")
                    st.markdown(f"**Last active**: {int(row.get('last_activity_days', 0))} days ago")
                if "text_summary" in row and pd.notna(row["text_summary"]):
                    st.text(row["text_summary"])
    else:
        st.info("Run the pipeline to load repository examples.")
