"""
Weak labeling pipeline.

Primary:   DeepSeek Chat API (cheap, JSON-mode available)
Fallback:  Rule-based classifier using domain-knowledge thresholds

Prompt design rationale:
  - System message defines exact category names and their boundaries.
  - User message includes the full text summary so the LLM sees ALL signals.
  - Temperature=0.1 reduces label variance; JSON-mode ensures parseable output.
  - Confidence field lets us filter low-certainty labels before training.

Limitations:
  - LLM may conflate template_boilerplate and low_value for very sparse repos.
  - Rule-based fallback is deterministic but ignores subtle textual signals.
  - Synthetic data already has ground-truth labels, so LLM is skipped for them.
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    setup_logging, PROCESSED_DIR, LABELED_DIR,
    CATEGORIES, LABEL2ID,
)

logger = setup_logging("llm_labeling")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

_SYSTEM = """You are an expert software engineering evaluator assessing GitHub repository maturity.
Classify the repository into exactly one of these categories:

1. intern_level     — simple scripts, homework, personal experiments, minimal structure
2. junior_level     — basic project, some README, 1-3 contributors, little CI/CD or testing
3. mid_level        — good engineering practices, CI/CD, tests, regular commits, 2-10 contributors
4. senior_level     — complex system, extensive docs, CI/CD, many contributors, regular releases
5. lead_architect   — large-scale project, architecture docs, 20+ contributors, high community activity
6. template_boilerplate — template/starter/boilerplate repo, cloned framework, minimal customization
7. low_value        — abandoned, tutorial copy, incomplete, very low engagement

Respond ONLY with valid JSON:
{"category": "<name>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}"""


def _build_prompt(summary: str) -> str:
    return f"Analyze this GitHub repository and classify it:\n\n{summary}\n\nRespond with JSON only."


def call_deepseek(summary: str, api_key: str, model: str = "deepseek-chat") -> Optional[dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_prompt(summary)},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            if result.get("category") in CATEGORIES:
                return result
        else:
            logger.warning(f"DeepSeek {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
    return None


def rule_based_label(row: dict) -> dict:
    """
    Deterministic rule-based classifier used when LLM is unavailable.
    Thresholds are derived from empirical patterns in OSS repositories.
    """
    stars = float(row.get("stars_count", 0))
    forks = float(row.get("forks_count", 0))
    contributors = float(row.get("contributors_count", 1))
    commit_freq = float(row.get("weekly_commit_freq", 0))
    has_ci = bool(int(row.get("has_ci_cd", 0)))
    closed_prs = float(row.get("closed_prs_count", 0))
    releases = float(row.get("releases_count", 0))
    readme_len = float(row.get("readme_length", 0))
    has_tests = bool(int(row.get("has_tests", 0)))
    last_activity = float(row.get("last_activity_days", 0))
    has_license = bool(int(row.get("has_license", 0)))
    is_fork = bool(int(row.get("is_fork", 0)))
    is_template = bool(int(row.get("is_template", 0)))

    # Template / boilerplate
    if is_template or (forks > stars * 3 and stars < 200 and last_activity > 180):
        return {"category": "template_boilerplate", "confidence": 0.78, "reasoning": "High fork ratio or template flag"}

    # Low value
    if stars <= 2 and closed_prs <= 2 and releases == 0 and last_activity > 365 and readme_len < 200:
        return {"category": "low_value", "confidence": 0.82, "reasoning": "Minimal engagement, long inactive"}

    # Lead / architect
    if contributors >= 20 and stars >= 1000 and has_ci and has_tests and releases >= 20:
        return {"category": "lead_architect", "confidence": 0.84, "reasoning": "Large team, high popularity, mature lifecycle"}

    # Senior
    if contributors >= 5 and stars >= 100 and has_ci and has_tests and releases >= 5 and readme_len >= 2000:
        return {"category": "senior_level", "confidence": 0.80, "reasoning": "Multi-contributor, CI/CD, tests, docs, releases"}

    # Mid-level
    if (has_ci or has_tests) and (closed_prs >= 10 or releases >= 2 or commit_freq >= 3):
        return {"category": "mid_level", "confidence": 0.74, "reasoning": "CI/CD or tests with regular development activity"}

    # Low value (second pass)
    if stars <= 1 and commit_freq <= 0.5 and readme_len < 100 and contributors == 1 and last_activity > 100:
        return {"category": "low_value", "confidence": 0.76, "reasoning": "Solo project, nearly no commits, no docs"}

    # Junior
    if contributors <= 3 and stars <= 50 and (readme_len >= 100 or closed_prs >= 1):
        return {"category": "junior_level", "confidence": 0.66, "reasoning": "Small team, basic structure present"}

    # Intern
    if contributors == 1 and stars <= 5 and commit_freq <= 2 and not has_ci:
        return {"category": "intern_level", "confidence": 0.70, "reasoning": "Solo project, minimal signals"}

    return {"category": "junior_level", "confidence": 0.55, "reasoning": "Default — ambiguous signals"}


def label_dataset(
    input_path=None,
    output_path=None,
    api_key: str = "",
    use_llm: bool = True,
    max_llm_calls: int = 300,
    delay: float = 0.5,
) -> pd.DataFrame:
    input_path = Path(input_path) if input_path else PROCESSED_DIR / "repos_with_summaries.csv"
    output_path = Path(output_path) if output_path else LABELED_DIR / "repos_labeled.csv"

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    labels, confs, reasons, sources = [], [], [], []
    llm_calls = 0

    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # Synthetic data has ground-truth category — skip LLM
        if row_dict.get("source") == "synthetic" and row_dict.get("category") in CATEGORIES:
            result = {"category": row_dict["category"], "confidence": 0.92, "reasoning": "synthetic ground truth"}
            src = "synthetic"
        elif use_llm and api_key and llm_calls < max_llm_calls:
            summary = row_dict.get("text_summary") or ""
            if not summary:
                from src.summarization import build_repo_summary
                summary = build_repo_summary(row_dict)
            res = call_deepseek(summary, api_key)
            if res:
                result, src, llm_calls = res, "llm_deepseek", llm_calls + 1
                time.sleep(delay)
            else:
                result, src = rule_based_label(row_dict), "rule_based_fallback"
        else:
            result, src = rule_based_label(row_dict), "rule_based"

        labels.append(result["category"])
        confs.append(result.get("confidence", 0.7))
        reasons.append(result.get("reasoning", ""))
        sources.append(src)

        if idx % 100 == 0:
            logger.info(f"  {idx}/{len(df)} labeled — {llm_calls} LLM calls")

    df["weak_label"] = labels
    df["label_confidence"] = confs
    df["label_reasoning"] = reasons
    df["label_source"] = sources
    df["label_id"] = df["weak_label"].map(LABEL2ID)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} labeled repos → {output_path}")
    logger.info(f"\n{df['weak_label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    import os
    key = os.getenv("DEEPSEEK_API_KEY", "")
    df = label_dataset(api_key=key, use_llm=bool(key))
    print(df["weak_label"].value_counts())
