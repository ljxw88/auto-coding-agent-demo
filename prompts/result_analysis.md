# Result Analysis Prompt

You are a research analyst. Interpret the experiment results and generate a concise conclusion.

## Experiment
**Idea**: {idea_title}
**Hypothesis**: {idea_hypothesis}

## Results
- Primary metric ({primary_metric}): **{new_value}**
- vs Original baseline: {vs_original}
- vs Rolling best: {vs_rolling}
- Outcome: **{outcome}**

## Training Info
{training_info}

---

## Task

Write a **2-3 sentence conclusion** that:
1. States clearly whether the idea worked
2. Quantifies the effect
3. Suggests a follow-up hypothesis or why it might have (not) worked

Keep it concise and factual. This will go into the research log.
