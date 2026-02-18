"""
Idea Generator — generates research ideas using LLM + arxiv search.

Usage:
    python agents/idea_generator.py
    python agents/idea_generator.py --count 5

Appends new ideas to ideas_backlog.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    call_llm, search_arxiv, load_config, load_backlog, save_backlog,
    load_experiment_history, timestamp, PROMPTS_DIR, REPO_ROOT
)


def get_codebase_summary() -> str:
    """Walk the research project directory and summarize file structure + key files."""
    config = load_config()
    project_dirs = [
        d for d in REPO_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name not in ("agents", "prompts", "baselines", "experiments", "automation-logs", "__pycache__")
    ]
    if not project_dirs:
        return "No research project directory found yet."

    lines = []
    for proj_dir in project_dirs:
        lines.append(f"## {proj_dir.name}/")
        for f in sorted(proj_dir.rglob("*.py"))[:20]:
            rel = f.relative_to(proj_dir)
            lines.append(f"  {rel}")
            # Include first 30 lines of key files
            if f.name in ("model.py", "train.py", "eval.py", "config.py", "dataset.py"):
                try:
                    content = f.read_text()[:1500]
                    lines.append(f"    ```python\n{content}\n    ```")
                except Exception:
                    pass
    return "\n".join(lines)


def format_experiment_history(history: list[dict]) -> str:
    if not history:
        return "No experiments run yet."
    lines = []
    for exp in history:
        status = exp.get("status", "?")
        outcome = exp.get("outcome", "?")
        idea = exp.get("idea_title", exp.get("idea_id", "?"))
        delta = exp.get("vs_original_baseline", {}).get("delta", "?")
        lines.append(f"- [{status}] {idea}: {outcome}, delta={delta}")
    return "\n".join(lines)


def generate_ideas(count: int = 5) -> list[dict]:
    config = load_config()
    backlog = load_backlog()
    history = load_experiment_history()

    already_tried = [
        idea.get("title", "") for idea in backlog.get("ideas", [])
        if idea.get("status") in ("done", "in_progress")
    ]

    codebase_summary = get_codebase_summary()
    history_text = format_experiment_history(history)

    # Search arxiv for relevant papers
    arxiv_context = ""
    if config.get("idea_generation", {}).get("use_arxiv", True):
        queries = config.get("arxiv_search_queries", [config.get("research_domain", "deep learning")])
        papers = []
        for q in queries[:2]:
            papers.extend(search_arxiv(q, max_results=3))
        if papers:
            paper_lines = []
            for p in papers[:6]:
                paper_lines.append(f"- [{p['id']}] {p['title']}\n  {p['abstract'][:200]}...")
            arxiv_context = "## Relevant recent papers:\n" + "\n".join(paper_lines)

    prompt_template = (PROMPTS_DIR / "idea_generation.md").read_text()
    user_prompt = prompt_template.format(
        research_goal=config.get("research_goal", "improve model performance"),
        research_domain=config.get("research_domain", "deep learning"),
        primary_metric=config.get("primary_metric", "accuracy"),
        codebase_summary=codebase_summary,
        experiment_history=history_text,
        already_tried="\n".join(f"- {t}" for t in already_tried) or "None yet",
        arxiv_context=arxiv_context,
        count=count,
    )

    system_prompt = (
        "You are an expert ML researcher. Generate creative, specific, and actionable "
        "research ideas. Return ONLY valid JSON — no markdown, no explanation."
    )

    response = call_llm(system_prompt, user_prompt)

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        ideas_raw = json.loads(text.strip())
        if isinstance(ideas_raw, dict) and "ideas" in ideas_raw:
            ideas_raw = ideas_raw["ideas"]
    except json.JSONDecodeError as e:
        print(f"[idea_generator] JSON parse error: {e}\nResponse:\n{response}")
        return []

    # Normalize and assign IDs
    existing_ids = {idea["id"] for idea in backlog.get("ideas", [])}
    new_ideas = []
    for i, raw in enumerate(ideas_raw[:count]):
        idea_id = f"idea_{len(backlog.get('ideas', [])) + i + 1:03d}"
        while idea_id in existing_ids:
            idea_id = idea_id[:-3] + f"{int(idea_id[-3:]) + 1:03d}"
        new_ideas.append({
            "id": idea_id,
            "title": raw.get("title", f"Idea {idea_id}"),
            "hypothesis": raw.get("hypothesis", ""),
            "implementation_notes": raw.get("implementation_notes", ""),
            "files_to_modify": raw.get("files_to_modify", []),
            "expected_impact": raw.get("expected_impact", "medium"),
            "source": raw.get("source", "LLM"),
            "status": "pending",
            "created_at": timestamp(),
        })

    return new_ideas


def main():
    parser = argparse.ArgumentParser(description="Generate research ideas")
    parser.add_argument("--count", type=int, default=5, help="Number of ideas to generate")
    args = parser.parse_args()

    print(f"[idea_generator] Generating {args.count} ideas...")
    ideas = generate_ideas(args.count)

    if not ideas:
        print("[idea_generator] No ideas generated.")
        sys.exit(1)

    backlog = load_backlog()
    backlog["ideas"].extend(ideas)
    save_backlog(backlog)

    print(f"[idea_generator] Added {len(ideas)} ideas to backlog:")
    for idea in ideas:
        print(f"  [{idea['id']}] {idea['title']}")

    # Output JSON for Claude to read
    print("\n--- GENERATED IDEAS JSON ---")
    print(json.dumps(ideas, indent=2))


if __name__ == "__main__":
    main()
