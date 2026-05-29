import os
import json
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = DATA_DIR / "labeled"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models" / "trained_models"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"

CATEGORIES = [
    "intern_level",
    "junior_level",
    "mid_level",
    "senior_level",
    "lead_architect",
    "template_boilerplate",
    "low_value",
]

CATEGORY_DESCRIPTIONS = {
    "intern_level": "Simple scripts, homework assignments, personal experiments with minimal structure",
    "junior_level": "Basic projects with some README, 1-3 contributors, limited CI/CD or testing",
    "mid_level": "Good engineering practices, CI/CD, tests, regular commits, multiple contributors",
    "senior_level": "Complex systems, extensive docs, strong CI/CD, many contributors, regular releases",
    "lead_architect": "Large-scale projects, architecture docs, large teams (20+ contributors), high community engagement",
    "template_boilerplate": "Standard template repos, boilerplate starters, cloned frameworks with minimal customization",
    "low_value": "Abandoned projects, tutorial copies, incomplete experiments, very low engagement",
}

LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}


def setup_logging(name: str = "github_intelligence") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, LABELED_DIR, SPLITS_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = None) -> dict:
    default = {
        "github_token": os.getenv("GITHUB_TOKEN", ""),
        "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model_name": "distilbert-base-uncased",
        "max_length": 256,
        "batch_size": 16,
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "num_repos_to_collect": 500,
        "test_size": 0.15,
        "val_size": 0.15,
        "random_seed": 42,
    }
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            default.update(json.load(f))
    return default


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
