"""
Repository text summary generator.

Converts numeric/boolean metadata into a human-readable paragraph suitable
for both LLM labeling prompts and BERT input sequences.

Design rationale:
  - Structured natural-language text lets the BERT model use both lexical
    patterns ("no tests", "weekly releases") and numerical context.
  - The format mirrors how a human engineer would describe a project in one
    paragraph, grounding the model in realistic language.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, PROCESSED_DIR

logger = setup_logging("summarization")


def build_repo_summary(row: dict) -> str:
    name = row.get("name", "unknown")
    _raw_desc = row.get("description")
    description = str(_raw_desc).strip() if _raw_desc and str(_raw_desc) not in ("nan", "None", "") else "No description provided"
    language = row.get("main_language", "Unknown")
    stars = int(row.get("stars_count", 0))
    forks = int(row.get("forks_count", 0))
    contributors = int(row.get("contributors_count", 1))
    commit_freq = float(row.get("weekly_commit_freq", 0))
    has_ci = bool(int(row.get("has_ci_cd", 0)))
    open_issues = int(row.get("open_issues_count", 0))
    closed_prs = int(row.get("closed_prs_count", 0))
    releases = int(row.get("releases_count", 0))
    readme_len = int(row.get("readme_length", 0))
    has_tests = bool(int(row.get("has_tests", 0)))
    age_days = int(row.get("repo_age_days", 0))
    last_activity = int(row.get("last_activity_days", 0))
    lang_count = int(row.get("language_count", 1))
    has_license = bool(int(row.get("has_license", 0)))
    topics_count = int(row.get("topics_count", 0))
    is_fork = bool(int(row.get("is_fork", 0)))
    is_template = bool(int(row.get("is_template", 0)))

    fork_tag = " (forked repository)" if is_fork else ""
    template_tag = " (template repository)" if is_template else ""
    ci_text = "has CI/CD workflows" if has_ci else "no CI/CD setup"
    tests_text = "includes automated tests" if has_tests else "no tests found"
    license_text = "licensed" if has_license else "no license"

    if commit_freq >= 20:
        activity_desc = "very high commit activity"
    elif commit_freq >= 5:
        activity_desc = "regular commit activity"
    elif commit_freq >= 1:
        activity_desc = "moderate commit activity"
    else:
        activity_desc = "low or sporadic commits"

    if last_activity <= 7:
        recency = "updated within the past week"
    elif last_activity <= 30:
        recency = "updated within the past month"
    elif last_activity <= 180:
        recency = "updated within 6 months"
    else:
        recency = f"last updated {last_activity} days ago (possibly inactive)"

    if age_days < 30:
        age_desc = "brand-new project"
    elif age_days < 180:
        age_desc = "recently created"
    elif age_days < 730:
        age_desc = "established project"
    else:
        age_desc = "long-running project"

    summary = (
        f"Repository: {name}{fork_tag}{template_tag}\n"
        f"Description: {description}\n"
        f"Language: {language} ({lang_count} language{'s' if lang_count != 1 else ''} total)\n"
        f"Popularity: {stars} stars, {forks} forks\n"
        f"Team: {contributors} contributor{'s' if contributors != 1 else ''}\n"
        f"Activity: {activity_desc} ({commit_freq:.1f} commits/week), {recency}\n"
        f"Infrastructure: {ci_text}, {tests_text}, {license_text}\n"
        f"Collaboration: {open_issues} open issue{'s' if open_issues != 1 else ''}, "
        f"{closed_prs} closed PR{'s' if closed_prs != 1 else ''}\n"
        f"Maturity: {releases} release{'s' if releases != 1 else ''}, "
        f"{readme_len}-char README, {topics_count} topic{'s' if topics_count != 1 else ''}\n"
        f"Age: {age_desc} ({age_days} days old)"
    )
    return summary


def generate_summaries(input_path=None, output_path=None) -> pd.DataFrame:
    input_path = Path(input_path) if input_path else PROCESSED_DIR / "repos_processed.csv"
    output_path = Path(output_path) if output_path else PROCESSED_DIR / "repos_with_summaries.csv"

    logger.info(f"Generating summaries from {input_path}")
    df = pd.read_csv(input_path)
    df["text_summary"] = df.apply(lambda r: build_repo_summary(r.to_dict()), axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} summaries → {output_path}")
    return df


if __name__ == "__main__":
    df = generate_summaries()
    print(df["text_summary"].iloc[0])
