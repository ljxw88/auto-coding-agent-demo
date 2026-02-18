"""
Shared utilities for the research agent:
- LLM client (Anthropic Claude)
- Arxiv search
- Git helpers
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
AGENTS_DIR = REPO_ROOT / "agents"
BASELINES_DIR = REPO_ROOT / "baselines"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PROMPTS_DIR = REPO_ROOT / "prompts"
BACKLOG_FILE = REPO_ROOT / "ideas_backlog.json"
CONFIG_FILE = REPO_ROOT / "research-config.json"
RESEARCH_LOG = REPO_ROOT / "research_log.md"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def load_backlog() -> dict:
    with open(BACKLOG_FILE) as f:
        return json.load(f)


def save_backlog(backlog: dict) -> None:
    with open(BACKLOG_FILE, "w") as f:
        json.dump(backlog, f, indent=2)


def load_baseline(name: str = "original_baseline") -> dict:
    path = BASELINES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def save_baseline(name: str, data: dict) -> None:
    path = BASELINES_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def next_experiment_id() -> str:
    existing = sorted(EXPERIMENTS_DIR.glob("exp_*.json"))
    if not existing:
        return "exp_001"
    last = existing[-1].stem  # e.g. exp_042
    num = int(last.split("_")[1]) + 1
    return f"exp_{num:03d}"


# ---------------------------------------------------------------------------
# LLM client (Anthropic)
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """Call Anthropic Claude API. Requires ANTHROPIC_API_KEY env var."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    config = load_config()
    model = model or config.get("idea_generation", {}).get("llm_model", "claude-opus-4-5")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Arxiv search
# ---------------------------------------------------------------------------

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arxiv via the public API. Returns list of {title, abstract, url, id}."""
    try:
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET
    except ImportError:
        return []

    base = "https://export.arxiv.org/api/query"
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"{base}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_data = resp.read()
    except Exception as e:
        print(f"[arxiv] search failed: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    results = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        results.append({
            "id": arxiv_id,
            "title": entry.find("atom:title", ns).text.strip(),
            "abstract": entry.find("atom:summary", ns).text.strip(),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return results


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def run_git(args: list[str], cwd: Path = None) -> tuple[int, str, str]:
    cwd = cwd or REPO_ROOT
    result = subprocess.run(
        ["git"] + args, cwd=str(cwd),
        capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def current_branch() -> str:
    _, out, _ = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return out


def create_experiment_branch(idea_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = f"experiment/{idea_id}-{ts}"
    code, _, err = run_git(["checkout", "-b", branch])
    if code != 0:
        raise RuntimeError(f"Failed to create branch {branch}: {err}")
    return branch


def checkout_main() -> None:
    code, _, err = run_git(["checkout", "main"])
    if code != 0:
        # Try 'master' as fallback
        run_git(["checkout", "master"])


def delete_branch(branch: str) -> None:
    run_git(["branch", "-D", branch])


def tag_branch(branch: str, tag: str) -> None:
    run_git(["tag", tag, branch])


def commit_all(message: str) -> None:
    run_git(["add", "-A"])
    run_git(["commit", "-m", message])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def load_experiment_history(max_recent: int = 10) -> list[dict]:
    files = sorted(EXPERIMENTS_DIR.glob("exp_*.json"))[-max_recent:]
    history = []
    for f in files:
        with open(f) as fp:
            history.append(json.load(fp))
    return history
