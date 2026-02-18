# Code Modification Prompt

You are an expert ML engineer. Apply the following research idea to the codebase with **minimal, surgical changes**.

## Research Idea
**Title**: {idea_title}

**Hypothesis**: {idea_hypothesis}

**Implementation Notes**:
{implementation_notes}

**Files to modify**: {files_to_modify}

## Relevant Code
{relevant_code}

---

## Task

1. Make **only the changes needed** to implement this idea
2. Do NOT refactor unrelated code
3. Do NOT break existing interfaces (train/eval commands must still work)
4. If adding new hyperparameters, provide sensible defaults
5. Add brief inline comments explaining what changed and why

After making changes, output a summary:
```
## Changes Made
- file.py: [description of change]
- ...

## What to verify before training
- [any gotchas or things to double-check]
```
