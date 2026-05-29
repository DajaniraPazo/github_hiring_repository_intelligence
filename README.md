# GitHub Hiring Repository Intelligence

> **Track A** — Engineering Maturity Classification via Weak Supervision + DistilBERT

AI-powered system that collects GitHub repository metadata, generates text summaries,
applies weak supervision via LLM labeling (DeepSeek), and fine-tunes a DistilBERT
classifier to estimate the **engineering maturity level** reflected by a repository.

---

## What Does This Project Do?

Given a GitHub repository, the system predicts which of 7 maturity categories it belongs to:

| Category | Description |
|---|---|
| `intern_level` | Simple scripts, homework, personal experiments |
| `junior_level` | Basic projects, limited structure, 1-3 contributors |
| `mid_level` | Good practices, CI/CD, tests, regular commits |
| `senior_level` | Complex systems, extensive docs, strong CI/CD, many contributors |
| `lead_architect` | Large-scale projects, 20+ contributors, high community engagement |
| `template_boilerplate` | Cloned templates, starters, minimal customization |
| `low_value` | Abandoned, incomplete, tutorial copies, very low engagement |

The goal is to **evaluate the repository**, not judge the developer personally.

---

## Track Selected

**Track A — Hiring Repository Intelligence**

---

## Repositories Analyzed

- **Collection method**: GitHub REST API (search queries by maturity profile) or synthetic generator
- **Sample size**: ~500 repositories (default)
- **Sampling strategy**: Targeted queries per maturity level:
  - High maturity: `stars:>1000 language:Python has:topics`
  - Mid-level: `stars:50..500 language:Python`
  - Junior/Intern: `stars:0..10 tutorial homework`
  - Templates: `is:template`
  - Low-value: `stars:0 pushed:<2022`
- **Synthetic fallback**: When no GitHub token is provided, a realistic synthetic dataset is generated using empirical distributions derived from known OSS patterns.

---

## GitHub Signals Used (15 signals)

| Signal | Type | Rationale |
|---|---|---|
| `stars_count` | numeric | Community popularity proxy |
| `forks_count` | numeric | Reuse and adoption indicator |
| `contributors_count` | numeric | Team size / collaboration depth |
| `weekly_commit_freq` | numeric | Development velocity |
| `has_ci_cd` | boolean | DevOps / automation maturity |
| `readme_length` | numeric | Documentation quality |
| `releases_count` | numeric | Versioning discipline |
| `closed_prs_count` | numeric | Code-review culture |
| `open_issues_count` | numeric | Community engagement |
| `has_tests` | boolean | Quality assurance culture |
| `repo_age_days` | numeric | Project longevity |
| `last_activity_days` | numeric | Project recency / abandonment risk |
| `language_count` | numeric | Technical breadth |
| `has_license` | boolean | Open-source maturity |
| `topics_count` | numeric | Discoverability / SEO investment |

---

## How Summaries Were Created

Each repository is converted to a structured natural-language paragraph:

```
Repository: awesome-api
Description: A production-grade REST API
Language: Python (2 languages total)
Popularity: 450 stars, 80 forks
Team: 8 contributors
Activity: regular commit activity (6.2 commits/week), updated within the past month
Infrastructure: has CI/CD workflows, includes automated tests, licensed
Collaboration: 15 open issues, 92 closed PRs
Maturity: 12 releases, 4800-char README, 5 topics
Age: established project (420 days old)
```

This format lets both LLMs (as labeled annotators) and BERT (as classifier) reason
over the same information that a human engineer would scan.

---

## Prompt Design

**System message** defines all 7 categories with precise boundaries using engineering
language, so the LLM applies domain knowledge rather than guessing.

**User message** includes the full text summary, forcing the LLM to reason from
multiple signals rather than a single proxy.

**Hyperparameters**: temperature=0.1, JSON mode, confidence field (labels with
confidence < 0.55 are dropped from training).

**DeepSeek** was chosen for its very low cost (~$0.001 / 1K tokens) and JSON-mode
support. A rule-based fallback handles cases when no API key is available.

---

## Dataset Split

| Split | Fraction | Usage |
|---|---|---|
| Training | 70% | Fine-tuning DistilBERT |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (held out) |

Stratified by category to maintain class proportions in all splits.
Minimum confidence threshold: 0.55 (removes uncertain LLM labels).

---

## Model Used

**Primary**: `distilbert-base-uncased` (HuggingFace Transformers)
- 40% fewer parameters than BERT-base
- 60% faster inference
- Retains ~97% of BERT's performance
- Fine-tuned for 3 epochs, AdamW, lr=2e-5, batch=16, max_len=256

**Fallback**: TF-IDF + Logistic Regression (when PyTorch is unavailable)
- `max_features=15000`, `ngram_range=(1,2)`, `sublinear_tf=True`
- `C=1.0`, `class_weight=balanced`

---

## Final Metrics

Metrics are computed on the held-out 15% test split after training.

**Synthetic data (rule-based labels):**
- Accuracy: ~1.00, F1 Macro: ~1.00
- *Expected*: synthetic data has ground-truth labels matching the exact signal distributions. The TF-IDF model learns summary text patterns trivially. This validates the pipeline, not real-world performance.

**Realistic expected performance (real GitHub data + LLM labels):**
- Accuracy: ~0.65–0.78, F1 Macro: ~0.60–0.72
- Harder classes: `junior_level` ↔ `mid_level` boundary is the main confusion zone

Run `python run_pipeline.py` to see your specific results in `output/metrics/evaluation_results.json`.

---

## Main Limitations

1. **Synthetic data** — Real GitHub data would surface edge cases not captured by parametric distributions
2. **Label noise** — LLM annotations have ~10-20% error rate on ambiguous repositories
3. **Signal availability** — CI/CD and test detection requires extra API calls (rate-limited)
4. **Category overlap** — `junior_level` ↔ `mid_level` boundary is especially ambiguous
5. **Temporal bias** — Older repositories appear less active, potentially misclassified as `low_value`

---

## Potential Business Applications

- **Talent acquisition**: Screen OSS portfolios at scale before technical interviews
- **Startup evaluation**: Accelerators / investors assessing founder technical quality
- **API product**: SaaS tool for hiring platforms (GitHub integration)
- **Internal benchmarking**: Engineering teams benchmarking their OSS presence

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set environment variables

```bash
# For real GitHub data
export GITHUB_TOKEN=your_github_token

# For LLM-powered labeling
export DEEPSEEK_API_KEY=your_deepseek_key
```

### 3. Run the full pipeline

```bash
# Synthetic data + rule-based labels (no API keys needed)
python run_pipeline.py

# With LLM labeling (requires DEEPSEEK_API_KEY)
python run_pipeline.py --llm
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Run the Streamlit App

```bash
streamlit run app.py
```

The app has **4 tabs**:

| Tab | Content |
|---|---|
| **Problem & Methodology** | Objective, signal descriptions, prompt strategy, limitations |
| **Exploratory Analysis** | Distribution plots, signal comparisons, correlation analysis |
| **Model Results** | Confusion matrix, per-class F1, baseline comparison, error analysis |
| **Interactive Explorer** | Browse dataset, filter by category, predict custom repositories |

Use the **▶ Run Full Pipeline** button in the sidebar to run all stages automatically.

---

## Repository Structure

```
github_hiring_repository_intelligence/
├── app.py                    # Streamlit 4-tab application
├── run_pipeline.py           # Full pipeline orchestrator
├── requirements.txt
├── README.md
│
├── src/
│   ├── utils.py              # Paths, labels, config
│   ├── github_collector.py   # GitHub API + synthetic generator
│   ├── preprocessing.py      # Feature engineering
│   ├── summarization.py      # Text summary generation
│   ├── llm_labeling.py       # DeepSeek + rule-based weak labels
│   ├── train.py              # DistilBERT / sklearn training
│   ├── evaluation.py         # Metrics, confusion matrix, error analysis
│   └── visualization.py      # All plots
│
├── data/
│   ├── raw/                  # repos_raw.csv
│   ├── processed/            # repos_processed.csv, repos_with_summaries.csv
│   ├── labeled/              # repos_labeled.csv
│   └── splits/               # train.csv, val.csv, test.csv
│
├── models/
│   └── trained_models/       # bert_classifier/ or sklearn_baseline.pkl
│
├── output/
│   ├── figures/              # PNG plots
│   ├── tables/               # CSV summaries
│   └── metrics/              # JSON evaluation results
│
└── video/
    └── link.txt              # Video explanation link
```
