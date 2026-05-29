"""
Full pipeline orchestrator — run this to generate data, labels, model, and metrics.

Usage:
    python run_pipeline.py [--llm]  # --llm uses DeepSeek API (requires DEEPSEEK_API_KEY)
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.utils import setup_logging, load_config, ensure_dirs, LABELED_DIR, FIGURES_DIR

logger = setup_logging("pipeline")


def main(use_llm: bool = False):
    t0 = time.time()
    ensure_dirs()
    cfg = load_config()
    api_key = cfg.get("deepseek_api_key", "") if use_llm else ""

    # ── Stage 1: Data collection ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1: Data collection")
    logger.info("=" * 60)
    from src.github_collector import generate_synthetic_dataset, GitHubCollector
    github_token = cfg.get("github_token", "")
    if github_token:
        logger.info("Using GitHub API …")
        collector = GitHubCollector(github_token)
        df_raw = collector.collect_dataset(num_per_category=40)
    else:
        logger.info("No GITHUB_TOKEN — generating synthetic dataset …")
        df_raw = generate_synthetic_dataset(num_samples=500, seed=cfg.get("random_seed", 42))
    logger.info(f"  {len(df_raw)} repositories collected")

    # ── Stage 2: Preprocessing ────────────────────────────────────────────────
    logger.info("Stage 2: Preprocessing & feature engineering")
    from src.preprocessing import preprocess
    df_proc = preprocess()
    logger.info(f"  Processed shape: {df_proc.shape}")

    # ── Stage 3: Summarization ────────────────────────────────────────────────
    logger.info("Stage 3: Generating text summaries")
    from src.summarization import generate_summaries
    df_summ = generate_summaries()
    logger.info(f"  Summaries generated: {len(df_summ)}")

    # ── Stage 4: LLM weak labeling ────────────────────────────────────────────
    logger.info("Stage 4: Weak labeling")
    from src.llm_labeling import label_dataset
    df_labeled = label_dataset(api_key=api_key, use_llm=bool(api_key))
    logger.info(f"  Label distribution:\n{df_labeled['weak_label'].value_counts().to_string()}")

    # ── Stage 5: Training ─────────────────────────────────────────────────────
    logger.info("Stage 5: Model training")
    from src.train import run_training_pipeline
    results = run_training_pipeline(input_path=LABELED_DIR / "repos_labeled.csv")
    logger.info(f"  Training results: {results}")

    # ── Stage 6: Evaluation ───────────────────────────────────────────────────
    logger.info("Stage 6: Evaluation")
    from src.evaluation import evaluate_model, run_baseline_comparison
    metrics, test_df = evaluate_model()
    if len(test_df):
        run_baseline_comparison(test_df)

    # ── Stage 7: Visualizations ───────────────────────────────────────────────
    logger.info("Stage 7: Generating plots")
    from src.visualization import generate_all_plots
    generate_all_plots(df_labeled, metrics if metrics else None)

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    if metrics:
        logger.info(f"  Accuracy : {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1-macro : {metrics.get('f1_macro', 0):.4f}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full repository intelligence pipeline.")
    parser.add_argument("--llm", action="store_true", help="Use DeepSeek LLM for weak labeling (requires DEEPSEEK_API_KEY)")
    args = parser.parse_args()
    main(use_llm=args.llm)
