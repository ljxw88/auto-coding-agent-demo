# Research Idea Generation Prompt

You are an expert ML researcher specializing in **RF/IQ signal processing and deep learning for device fingerprinting**.
Your task is to propose **{count}** concrete, actionable research ideas.

## Research Goal
{research_goal}

## Domain
{research_domain}

## Metric to Optimize
Primary metric: **{primary_metric}** (higher is better unless stated otherwise)

## Architecture Context
The model is a **GatedDeltaNet** (state space model / linear recurrence) applied to IQ signal sequences.
- Input: `(B, 2, 8192)` — 2-channel (I/Q) signal of length 8192
- Pipeline: patch embedding → GatedDeltaNet blocks → pooling → classifier
- Current SOTA parameters: `--dim 512 --depth 4 --heads 8 --pooling last --patch_size 64 --patch_embed_type causal_conv --augment`
- Training is somewhat unstable (82-84% range); target is 86%+

## ⚠️ Hard Constraints — Ideas MUST respect these
- **NEVER** suggest modifying `causal_iq_transformer.py` or `causal_iq_kda.py`
- **NEVER** suggest increasing dropout above 0 — it hurts performance
- **NEVER** suggest increasing weight_decay above 0.0 — it hurts performance
- **NEVER** change the architecture type from `gateddeltanet`
- Always preserve `--augment` and `--compile` flags

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
- Be **specific** about what to change and where (file names, function names, exact values)
- Explain the **hypothesis** (why this will improve fingerprinting accuracy on IQ signals)
- Estimate **expected_impact**: "low" | "medium" | "high"
- List **files_to_modify** (relative paths within `experiment_project/`)
- Cite a **source** if inspired by a paper (arxiv ID or "LLM")

## Productive idea categories for this project

Focus your ideas on areas that are likely to help **without violating constraints**:

1. **Patch embedding improvements**
   - Different patch sizes (e.g. 32, 128) or multi-scale patches
   - STFT-based embedding (`--patch_embed_type stft`) for frequency-domain features
   - Learnable Fourier or wavelet features as patch embedding
   - Overlapping vs non-overlapping patches (patch_stride changes)

2. **Preprocessing / normalization**
   - Per-sample amplitude normalization strategies in the data pipeline
   - Frequency-domain features concatenated with time-domain
   - Phase/magnitude decomposition as additional input channels

3. **GatedDeltaNet hyperparameters**
   - Model capacity: `--dim` (try 256 or 768), `--depth` (try 3 or 6), `--heads` (try 4 or 16)
   - Pooling strategy: `--pooling mean` vs `last` vs `attn`

4. **Learning rate schedule**
   - Cosine annealing with warm restarts
   - Linear warmup + cosine decay
   - OneCycleLR

5. **Data augmentation enhancements**
   - Additional augmentation types in the GPU augmentation backend
   - Mixup or CutMix in the IQ domain

6. **Loss function**
   - Label smoothing (small epsilon like 0.05)
   - Focal loss for hard-to-classify devices
   - Auxiliary classification loss at intermediate layers

Return ONLY a JSON array (no markdown, no extra text):

```json
[
  {{
    "title": "Short descriptive title",
    "hypothesis": "Why this should improve fingerprinting accuracy on IQ signals",
    "implementation_notes": "Exactly what to change, in which file, and how. Be specific about function names and parameter values.",
    "files_to_modify": ["experiment_project/path/to/file.py"],
    "expected_impact": "medium",
    "source": "arxiv:XXXX.XXXXX or LLM"
  }}
]
```

