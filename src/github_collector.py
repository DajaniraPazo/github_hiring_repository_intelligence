"""
GitHub API collector and synthetic dataset generator.

Real collection: provide GITHUB_TOKEN env var.
Synthetic fallback: generates realistic data for each engineering maturity category.
"""

import re
import time
import random
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, RAW_DIR, load_config

logger = setup_logging("github_collector")
GITHUB_API_BASE = "https://api.github.com"


class GitHubCollector:
    def __init__(self, token: str = ""):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github.v3+json"})
        if token:
            self.session.headers["Authorization"] = f"token {token}"

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = min(max(0, reset - time.time()) + 1, 60)
                logger.warning(f"Rate limited — waiting {wait:.0f}s")
                time.sleep(wait)
                return None
            logger.debug(f"HTTP {r.status_code} for {url}")
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def _last_page(self, url: str, params: dict = None) -> int:
        r = self.session.get(url, params={**(params or {}), "per_page": 1})
        link = r.headers.get("Link", "")
        m = re.search(r'page=(\d+)>; rel="last"', link)
        if m:
            return int(m.group(1))
        data = r.json() if r.status_code == 200 else []
        return len(data) if isinstance(data, list) else 0

    def get_contributors_count(self, owner: str, repo: str) -> int:
        return self._last_page(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contributors", {"anon": "true"}) or 1

    def get_commit_frequency(self, owner: str, repo: str) -> float:
        data = self._get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/stats/participation")
        if data and "all" in data:
            recent = data["all"][-12:]
            return round(sum(recent) / max(len(recent), 1), 2)
        return 0.0

    def get_releases_count(self, owner: str, repo: str) -> int:
        return self._last_page(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/releases")

    def get_closed_prs_count(self, owner: str, repo: str) -> int:
        return self._last_page(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls", {"state": "closed"}
        )

    def has_ci_cd(self, owner: str, repo: str) -> bool:
        data = self._get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/.github/workflows")
        return isinstance(data, list) and len(data) > 0

    def has_tests(self, owner: str, repo: str) -> bool:
        for d in ["tests", "test", "__tests__", "spec"]:
            if self._get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{d}"):
                return True
        return False

    def get_readme_length(self, owner: str, repo: str) -> int:
        data = self._get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme")
        return data.get("size", 0) if data else 0

    def extract_repo_features(self, owner: str, repo_name: str) -> Optional[dict]:
        info = self._get(f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}")
        if not info:
            return None

        def parse_dt(s):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        now = datetime.now(parse_dt(info.get("created_at", "2020-01-01T00:00:00Z")).tzinfo)
        created = parse_dt(info.get("created_at", "2020-01-01T00:00:00Z"))
        pushed = parse_dt(info.get("pushed_at", "2020-01-01T00:00:00Z"))

        return {
            "repo_id": info.get("id"),
            "owner": owner,
            "name": repo_name,
            "full_name": info.get("full_name", f"{owner}/{repo_name}"),
            "description": info.get("description", "") or "",
            "main_language": info.get("language") or "Unknown",
            "stars_count": info.get("stargazers_count", 0),
            "forks_count": info.get("forks_count", 0),
            "open_issues_count": info.get("open_issues_count", 0),
            "topics": info.get("topics", []),
            "topics_count": len(info.get("topics", [])),
            "has_license": info.get("license") is not None,
            "is_fork": info.get("fork", False),
            "is_template": info.get("is_template", False),
            "repo_age_days": (now - created).days,
            "last_activity_days": (now - pushed).days,
            "watchers_count": info.get("watchers_count", 0),
            "network_count": info.get("network_count", 0),
            "contributors_count": self.get_contributors_count(owner, repo_name),
            "readme_length": self.get_readme_length(owner, repo_name),
            "has_ci_cd": self.has_ci_cd(owner, repo_name),
            "releases_count": self.get_releases_count(owner, repo_name),
            "closed_prs_count": self.get_closed_prs_count(owner, repo_name),
            "weekly_commit_freq": self.get_commit_frequency(owner, repo_name),
            "has_tests": self.has_tests(owner, repo_name),
            "language_count": 1,
            "collected_at": datetime.now().isoformat(),
            "source": "github_api",
        }

    def search_repos(self, query: str, n: int = 30) -> list:
        repos, page = [], 1
        while len(repos) < n:
            data = self._get(
                f"{GITHUB_API_BASE}/search/repositories",
                {"q": query, "sort": "updated", "order": "desc", "per_page": min(30, n), "page": page},
            )
            if not data or not data.get("items"):
                break
            repos.extend(data["items"])
            if len(data["items"]) < 30:
                break
            page += 1
            time.sleep(1)
        return repos[:n]

    def collect_dataset(self, num_per_category: int = 40) -> pd.DataFrame:
        logger.info("Starting GitHub data collection")
        queries = [
            "stars:>1000 language:Python has:topics pushed:>2023-01-01",
            "stars:>500 forks:>100 language:JavaScript has:workflows",
            "stars:50..500 language:Python has:topics has:license",
            "stars:10..100 language:JavaScript",
            "stars:0..10 language:Python tutorial",
            "stars:0..5 language:Python homework",
            "is:template stars:>5",
            "stars:0 pushed:<2022-01-01",
        ]
        seen, all_features = set(), []
        for q in queries:
            if len(all_features) >= num_per_category * 7:
                break
            for repo in self.search_repos(q, num_per_category):
                fn = repo.get("full_name", "")
                if fn in seen:
                    continue
                seen.add(fn)
                owner, name = fn.split("/", 1) if "/" in fn else ("unknown", fn)
                logger.info(f"  {fn}")
                feat = self.extract_repo_features(owner, name)
                if feat:
                    all_features.append(feat)
                time.sleep(0.5)
        df = pd.DataFrame(all_features)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DIR / "repos_raw.csv", index=False)
        logger.info(f"Collected {len(df)} repos")
        return df


# ---------------------------------------------------------------------------
# Synthetic dataset generator — used when no GitHub token is available
# ---------------------------------------------------------------------------

_CATEGORY_CONFIG = {
    "intern_level": dict(
        weight=0.15, stars=(0, 8), forks=(0, 3), contributors=(1, 2),
        commit_freq=(0.1, 3.0), ci_prob=0.05, issues=(0, 3), prs=(0, 3),
        releases=(0, 1), readme=(0, 600), tests_prob=0.06, age=(7, 400),
        last_act=(30, 800), langs=(1, 2), license_prob=0.12, topics=(0, 2),
    ),
    "junior_level": dict(
        weight=0.20, stars=(0, 30), forks=(0, 8), contributors=(1, 3),
        commit_freq=(0.5, 6.0), ci_prob=0.20, issues=(0, 12), prs=(0, 12),
        releases=(0, 3), readme=(50, 2500), tests_prob=0.22, age=(30, 800),
        last_act=(14, 400), langs=(1, 3), license_prob=0.28, topics=(0, 4),
    ),
    "mid_level": dict(
        weight=0.20, stars=(10, 600), forks=(5, 150), contributors=(2, 12),
        commit_freq=(3.0, 18.0), ci_prob=0.72, issues=(5, 60), prs=(10, 120),
        releases=(2, 25), readme=(800, 6000), tests_prob=0.72, age=(90, 1500),
        last_act=(0, 100), langs=(1, 5), license_prob=0.72, topics=(2, 7),
    ),
    "senior_level": dict(
        weight=0.15, stars=(100, 6000), forks=(50, 1200), contributors=(5, 35),
        commit_freq=(10.0, 35.0), ci_prob=0.90, issues=(20, 250), prs=(50, 600),
        releases=(10, 60), readme=(2500, 18000), tests_prob=0.90, age=(365, 3650),
        last_act=(0, 35), langs=(2, 7), license_prob=0.92, topics=(3, 12),
    ),
    "lead_architect": dict(
        weight=0.10, stars=(1000, 60000), forks=(500, 12000), contributors=(20, 300),
        commit_freq=(20.0, 120.0), ci_prob=0.98, issues=(50, 1200), prs=(200, 6000),
        releases=(20, 300), readme=(5000, 35000), tests_prob=0.98, age=(730, 5500),
        last_act=(0, 14), langs=(3, 12), license_prob=0.99, topics=(5, 18),
    ),
    "template_boilerplate": dict(
        weight=0.10, stars=(1, 150), forks=(20, 600), contributors=(1, 4),
        commit_freq=(0.1, 3.0), ci_prob=0.60, issues=(0, 25), prs=(0, 15),
        releases=(0, 5), readme=(500, 9000), tests_prob=0.45, age=(180, 2200),
        last_act=(90, 800), langs=(1, 4), license_prob=0.80, topics=(2, 9),
    ),
    "low_value": dict(
        weight=0.10, stars=(0, 5), forks=(0, 4), contributors=(1, 1),
        commit_freq=(0.0, 1.0), ci_prob=0.04, issues=(0, 4), prs=(0, 2),
        releases=(0, 1), readme=(0, 350), tests_prob=0.04, age=(1, 3700),
        last_act=(180, 1900), langs=(1, 1), license_prob=0.07, topics=(0, 2),
    ),
}

_PREFIXES = {
    "intern_level": ["my-", "test-", "learning-", "practice-", "homework-"],
    "junior_level": ["simple-", "basic-", "mini-", "demo-", "sample-"],
    "mid_level": ["awesome-", "better-", "clean-", "fast-", "cool-"],
    "senior_level": ["enterprise-", "scalable-", "production-", "robust-", "advanced-"],
    "lead_architect": ["", "open-", "core-", "platform-", "engine-"],
    "template_boilerplate": ["template-", "boilerplate-", "starter-", "scaffold-"],
    "low_value": ["untitled-", "test-", "backup-", "old-", "temp-"],
}

_LANGUAGES = ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++", "C#", "Ruby", "PHP"]
_SUBJECTS = ["app", "api", "tool", "lib", "service", "project", "system", "bot", "web", "cli"]

_DESCRIPTIONS = {
    "intern_level": [
        "My first Python project", "Learning Flask basics", "CS homework assignment",
        "Todo list app practice", "Calculator in Python", "Personal learning project",
        "Simple web scraper practice", "Beginner machine learning experiments",
    ],
    "junior_level": [
        "REST API built with Flask", "Simple CRUD application", "Portfolio website backend",
        "Discord bot for my server", "Weather app using public APIs", "Blog platform clone",
        "Task management app", "Basic e-commerce backend",
    ],
    "mid_level": [
        "Scalable REST API with authentication and tests", "CLI tool for automating deployments",
        "Data pipeline for processing CSV files", "Microservice for user management",
        "Open-source library for data validation", "Full-stack web application with CI/CD",
        "Python SDK for third-party API integration", "Automated testing framework extension",
    ],
    "senior_level": [
        "Production-grade distributed system with Kubernetes", "High-performance message queue processor",
        "Multi-tenant SaaS backend with extensive docs", "Real-time analytics engine",
        "Open-source framework for building APIs", "Database migration and management tool",
        "Security audit and compliance toolkit", "Cloud-native observability platform",
    ],
    "lead_architect": [
        "Core infrastructure platform used by thousands", "Open-source orchestration framework",
        "Developer toolchain for cloud deployments", "Widely adopted authentication library",
        "Cross-platform build system", "Distributed database engine",
        "Enterprise API gateway with plugins", "Unified observability and monitoring platform",
    ],
    "template_boilerplate": [
        "Starter template for React + TypeScript", "FastAPI project boilerplate",
        "Django REST framework scaffold", "Next.js starter with TailwindCSS",
        "Node.js microservice template", "Python package boilerplate",
        "Docker + CI/CD template", "MLOps project starter kit",
    ],
    "low_value": [
        "", "test", "untitled", "backup of my old project",
        "just trying something", "test123", "copy of tutorial code",
        "WIP project (abandoned)", "forked to try later",
    ],
}


def _rand_int(lo, hi):
    lo, hi = int(lo), int(hi)
    return np.random.randint(lo, max(lo + 1, hi + 1))


def generate_synthetic_dataset(num_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)
    rows, repo_id = [], 10_000

    for category, cfg in _CATEGORY_CONFIG.items():
        count = max(10, int(num_samples * cfg["weight"]))
        for i in range(count):
            stars = max(0, _rand_int(*cfg["stars"]))
            forks = max(0, int(stars * np.random.uniform(0.05, 0.4) + _rand_int(0, 5)))
            contributors = _rand_int(*cfg["contributors"])
            commit_freq = round(np.random.uniform(*cfg["commit_freq"]), 2)
            open_issues = _rand_int(*cfg["issues"])
            closed_prs = _rand_int(*cfg["prs"])
            releases = _rand_int(*cfg["releases"])
            readme_len = _rand_int(*cfg["readme"])
            age_days = _rand_int(*cfg["age"])
            last_act = _rand_int(*cfg["last_act"])
            lang_count = _rand_int(*cfg["langs"])
            topics_count = _rand_int(*cfg["topics"])

            prefix = random.choice(_PREFIXES[category])
            name = f"{prefix}{random.choice(_SUBJECTS)}-{i}"
            language = random.choice(_LANGUAGES)
            owner = f"user_{random.randint(1000, 99999)}"

            rows.append(
                {
                    "repo_id": repo_id + i,
                    "owner": owner,
                    "name": name,
                    "full_name": f"{owner}/{name}",
                    "description": random.choice(_DESCRIPTIONS.get(category, ["A software project"])),
                    "main_language": language,
                    "stars_count": stars,
                    "forks_count": forks,
                    "open_issues_count": open_issues,
                    "topics": [f"topic{j}" for j in range(topics_count)],
                    "topics_count": topics_count,
                    "has_license": np.random.random() < cfg["license_prob"],
                    "is_fork": category in ("low_value", "template_boilerplate") and np.random.random() < 0.30,
                    "is_template": category == "template_boilerplate" and np.random.random() < 0.40,
                    "repo_age_days": age_days,
                    "last_activity_days": last_act,
                    "watchers_count": max(0, int(stars * np.random.uniform(0.1, 0.3))),
                    "network_count": max(0, int(forks * np.random.uniform(0.8, 1.2))),
                    "contributors_count": contributors,
                    "readme_length": readme_len,
                    "has_ci_cd": np.random.random() < cfg["ci_prob"],
                    "releases_count": releases,
                    "closed_prs_count": closed_prs,
                    "weekly_commit_freq": commit_freq,
                    "has_tests": np.random.random() < cfg["tests_prob"],
                    "language_count": lang_count,
                    "collected_at": datetime.now().isoformat(),
                    "category": category,
                    "source": "synthetic",
                }
            )
        repo_id += count

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "repos_raw.csv", index=False)
    logger.info(f"Generated {len(df)} synthetic repos → {RAW_DIR / 'repos_raw.csv'}")
    return df


if __name__ == "__main__":
    cfg = load_config()
    if cfg.get("github_token"):
        collector = GitHubCollector(cfg["github_token"])
        df = collector.collect_dataset(num_per_category=40)
    else:
        logger.info("No GITHUB_TOKEN found — generating synthetic dataset.")
        df = generate_synthetic_dataset(500)
    print(f"Dataset shape: {df.shape}")
    print(df["category"].value_counts() if "category" in df.columns else df.head())
