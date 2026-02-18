# Research Idea Generation Prompt

You are an expert ML researcher tasked with proposing **{count}** concrete, actionable research ideas.

## Research Goal
{research_goal}

## Domain
{research_domain}

## Metric to Optimize
Primary metric: **{primary_metric}** (higher is better unless stated otherwise)

## Current Codebase
{codebase_summary}

## Experiment History (recent results)
{experiment_history}

## Already Tried (do NOT suggest these again)
{already_tried}

{arxiv_context}

---

## Task

Generate exactly **{count}** research ideas. For each idea:
- Be **specific** about what to change and where (file names, function names, hyperparameter values)
- Explain the **hypothesis** (why you think this will help)
- Estimate **expected_impact**: "low" | "medium" | "high"
- List **files_to_modify** (relative paths)
- Cite a **source** if inspired by a paper (arxiv ID or "LLM")

Ideas should be diverse: try different categories like:
- Architecture changes (new layers, attention, normalization)
- Training dynamics (lr schedule, optimizer, gradient clipping)  
- Regularization (dropout, weight decay, data augmentation)
- Loss functions (auxiliary losses, label smoothing)
- Data preprocessing (normalization, augmentation strategies)

Return ONLY a JSON array (no markdown, no extra text):

```json
[
  {
    "title": "Short descriptive title",
    "hypothesis": "Why this should improve the metric",
    "implementation_notes": "Exactly what to change, where, and how. Be specific.",
    "files_to_modify": ["path/to/file.py"],
    "expected_impact": "medium",
    "source": "arxiv:XXXX.XXXXX or LLM"
  }
]
```
