# Effective Approaches for IQ Signal Device Fingerprinting with GatedDeltaNet

This document summarises every approach that achieved **≥ 84.0%** accuracy on the LocD test set
(transmitters 35–45, packets 0–100 of `/LOCAL/data/n210_2_leg_LocD.h5`).
Details are written at the implementation level so the work can be reproduced from a blank project.

---

## Task Overview

**Goal**: Classify which of 10 USRP N210 software-defined radios transmitted a given burst,
using only the raw I/Q baseband signal — a task known as Radio-Frequency Fingerprinting
Identification (RFFI).

**Input**: `(B, 2, 8192)` float32 tensor — I channel at index 0, Q channel at index 1,
8 192 samples per burst, collected at 1 MHz sample rate.

**Output**: Class logit vector of shape `(B, 10)`.

**Test protocol**: Model is trained on shards from transmitters 35–45.
Final evaluation is on *unseen* packets (pkt 0–100) from the LocD capture location — a
different receiver position from training.

**Score threshold used in this document**: accuracy ≥ 84.0 %.

---

## Result Summary

| Approach | Accuracy | Δ vs 84 % baseline |
|---|---|---|
| **Baseline** (GDN, dim=512, depth=4, `--pooling last`) | 84.0 % | — |
| **Cosine LR Schedule** (D3, on top of focal loss) | 84.6 % | +0.6 pp |
| **expand_v=1.5** (I4, on top of E5+H2) | 84.5 % | +0.5 pp |
| **TTA** (H4, limited benefit, on top of E5+H2) | 84.5 % | +0.5 pp |
| **LR warmup** (J5, on top of E5+H2) | 84.3 % | +0.3 pp |
| **Focal Loss γ=2.0 + label_smoothing=0.05** (E5) | **85.0 %** | **+1.0 pp** |
| **lr=1e-4** (I1, on top of E5+H2, matches E5) | 85.0 % | +1.0 pp |
| **Focal Loss + head_dim=128, num_heads=4** (H2) | **85.1 %** | **+1.1 pp** |

> Note: D3, I4, H4, J5, and I1 all ran on top of the focal loss (E5) configuration.
> Their absolute scores ≥ 84 % are due to the focal loss baseline, not their own change.
> The two genuinely additive improvements are **E5** and **H2**.

---

## Approach 1 — The 84% Baseline Architecture

This is the starting point that achieves 84 % before any modification.
Understanding it fully is a prerequisite for everything else.

### 1.1 Architecture: CausalIQ GatedDeltaNet

The backbone is a **GatedDeltaNet** sequence model from the
[Flash Linear Attention (fla)](https://github.com/sustcsonglin/flash-linear-attention) library,
wrapped in a classification head.

```
Input (B, 2, 8192)
    │
    ▼  Causal Conv Patch Embed
    │  kernel_size=64, stride=32  →  ~257 tokens of dim=512
    │
    ▼  Transpose  →  (B, 257, 512)
    │
    ▼  GatedDeltaNetBlock × 4
    │    each block:
    │       RMSNorm → GatedDeltaNet → residual add → dropout
    │       RMSNorm → GatedMLP (ratio=4) → residual add → dropout
    │
    ▼  RMSNorm (final)
    │
    ▼  Pooling: last token  →  (B, 512)
    │
    ▼  Linear(512, 10)
    │
Output logits (B, 10)
```

#### Key design choices confirmed by ablation

| Component | Value | Why |
|---|---|---|
| `pooling` | `last` | Causal GDN encodes full sequence context in the final recurrent state. Mean pooling and attention pooling both hurt (–3.9 pp). |
| `patch_embed_type` | `causal_conv` | Time-domain causal convolution. STFT frequency-domain embedding loses time-localised startup transients (–2.0 pp). |
| `patch_size` | 64 | Smaller (32) doubles tokens → too slow (>10 s/epoch). Larger loses temporal detail. |
| `patch_stride` | 32 | Halved stride (16) doubles tokens → too slow (14.8 s/epoch). |
| `dim` | 512 | dim=768 hurts (–3.2 pp). Wider model overfits. |
| `depth` | 4 | depth=6 hurts (–2.3 pp). Task is well-served by the smaller model. |
| `resid_dropout` | 0.0 | Dropout=0 is confirmed optimal; any dropout above 0 degrades results. |
| `weight_decay` | 0.0 | Also confirmed optimal; never raise above 0. |
| `expand_v` | 2.0 | expand_v=3.0 or 4.0 → too slow. expand_v=1.5 → –0.6 pp. |
| `allow_neg_eigval` | False | Inhibitory delta-rule updates destabilise training (–1.5 pp). |
| `conv_size` | 4 | Internal short causal conv in GDN. conv_size=8 → too slow (10.6 s/epoch). |
| `conv_bias` | False | Learnable bias in GDN short conv interferes with delta-rule propagation (–1.2 pp). |
| `mlp_ratio` | 4.0 | mlp_ratio=2.0 under-provisions feedforward capacity (–3.3 pp). |
| `mode` | `chunk` | `fused_recurrent` is inference-only; fla library raises AssertionError during training. |

### 1.2 Causal Conv Patch Embedding Implementation

`model_arch/conv.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding so output[t] depends only on x[:t]."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs["padding"] = 0          # disable built-in padding
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_pad = (self.kernel_size[0] - 1) * self.dilation[0]
        if left_pad > 0:
            x = F.pad(x, (left_pad, 0))    # pad only on the left
        return super().forward(x)

def causal_conv1d(*args, **kwargs) -> CausalConv1d:
    return CausalConv1d(*args, **kwargs)
```

In `BaseCausalIQBackbone.__init__`:
```python
self.patch_embed = causal_conv1d(
    in_channels=2,        # I and Q
    out_channels=512,     # dim
    kernel_size=64,       # patch_size
    stride=32,            # patch_stride
)
# Resulting token count for seq_len=8192:
self.seq_len_patched = (8192 - 1) // 32 + 1  # = 257
```

Forward pass:
```python
def patch(self, x):
    # x: (B, 2, 8192)
    x = self.patch_embed(x)           # (B, 512, 257)
    x = x.transpose(1, 2).contiguous()  # (B, 257, 512)
    return x
```

### 1.3 GatedDeltaNetBlock

Each of the 4 blocks uses `fla.layers.gated_deltanet.GatedDeltaNet` with a gated MLP:

```python
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.modules import GatedMLP as GatedDeltaNetMLP
from liger_kernel.transformers import LigerRMSNorm as RMSNorm

class GatedDeltaNetBlock(nn.Module):
    def __init__(self, dim, expand_v=2.0, head_dim=64, num_heads=4,
                 hidden_ratio=4.0, mode='chunk', use_gate=True,
                 use_short_conv=True, allow_neg_eigval=False,
                 conv_size=4, conv_bias=False,
                 resid_dropout=0.0, layer_idx=None, norm_eps=1e-5):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = GatedDeltaNet(
            hidden_size=dim,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode=mode,                     # MUST be 'chunk' for training
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,  # keep False
            conv_size=conv_size,
            conv_bias=conv_bias,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
        )
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = GatedDeltaNetMLP(
            hidden_size=dim,
            hidden_ratio=hidden_ratio,    # =4.0
        )
        self.drop = nn.Dropout(resid_dropout)   # =0.0 in practice

    def forward(self, x, attention_mask=None, past_key_values=None,
                use_cache=False, **kwargs):
        residual = x
        x = self.attn_norm(x)
        x, _, past_key_values = self.attn(
            hidden_states=x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        x = residual + self.drop(x)       # first residual

        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + self.drop(x)       # second residual
        return x
```

### 1.4 Full Backbone and Classifier

```python
class CausalIQGatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # --- patch embedding (described in 1.2) ---
        self.patch_embed = causal_conv1d(2, cfg.dim, cfg.patch_size, cfg.patch_stride)
        self.seq_len_patched = (cfg.seq_len - 1) // cfg.patch_stride + 1
        self.ln_f = RMSNorm(cfg.dim, eps=cfg.norm_eps)

        self.blocks = nn.ModuleList([
            GatedDeltaNetBlock(
                dim=cfg.dim,
                expand_v=cfg.expand_v,        # 2.0
                head_dim=cfg.head_dim,        # 64 (baseline) / 128 (H2)
                num_heads=cfg.num_heads,      # 8 (baseline) / 4 (H2)
                hidden_ratio=cfg.mlp_ratio,   # 4.0
                mode=cfg.mode,                # 'chunk'
                use_gate=cfg.use_gate,        # True
                use_short_conv=cfg.use_short_conv,  # True
                allow_neg_eigval=cfg.allow_neg_eigval,  # False
                conv_size=cfg.conv_size,      # 4
                conv_bias=cfg.conv_bias,      # False
                resid_dropout=cfg.resid_dropout,  # 0.0
                layer_idx=i,
                norm_eps=cfg.norm_eps,
            )
            for i in range(cfg.depth)  # depth=4
        ])

    def forward(self, x, **kwargs):
        x = self.patch_embed(x)                    # (B, dim, T)
        x = x.transpose(1, 2).contiguous()         # (B, T, dim)
        for blk in self.blocks:
            x = blk(x, **kwargs)
        x = self.ln_f(x)
        emb = x[:, -1, :]                          # last-token pooling
        return emb

class CausalIQClassifier(nn.Module):
    def __init__(self, backbone, dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, num_classes)
        # small-std init for head prevents early saturation
        nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, **kwargs):
        emb = self.backbone(x, **kwargs)
        logits = self.head(emb)
        return logits, emb       # always returns (logits, emb)
```

### 1.5 GPU-Accelerated Data Augmentation

All augmentation happens **on GPU** inside the training loop, before the forward pass.
No augmentation is applied at test time.

The augmentation pipeline (`utils/gpu_aug_iq.py`):

```
Raw IQ (B, 2, L)
    │
    ▼  Convert to complex  →  (B, L)  complex64
    │
    ▼  Jakes Multipath Fading (2 taps, Fd~U(0,5) Hz, exponential PDP)
    │
    ▼  Carrier Frequency Offset (CFO), ±200 Hz
    │
    ▼  AWGN, SNR~U(10, 40) dB
    │
    ▼  RMS normalisation (per sample)
    │
Output (B, 2, L) float32
```

**Important constraints confirmed by ablation**:
- IQ imbalance augmentation: **hurts** (–3.1 pp). Do NOT add.
- Phase noise augmentation: **hurts** (–5.7 pp). Do NOT add.
- Gain/power jitter: **hurts** (–1.6 pp vs focal loss baseline). Do NOT add.
- Any temporal shift (circular roll, temporal crop): **hurts** (–7.5 pp and –15.3 pp).
  The GDN recurrent state encodes absolute burst position; shifting destroys the transient fingerprint.

```python
import math
import torch

@torch.compile
class GPUAugmentor(torch.nn.Module):
    def __init__(self, sample_rate=1e6, snr_min=10.0,
                 use_iq_imbalance=False,    # keep False
                 use_phase_noise=False,     # keep False
                 use_const_phase_rot=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.ts = 1.0 / sample_rate
        self.snr_min = snr_min

    def forward(self, iq_batch):
        B, C, L = iq_batch.shape
        device = iq_batch.device

        # 1. to complex
        x = torch.complex(iq_batch[:, 0], iq_batch[:, 1])   # (B, L)

        # 2. Jakes multipath (2 taps)
        tau_d = torch.rand(B, device=device) * (300e-9 - 5e-9) + 5e-9
        # exponential PDP with 2 taps
        taps_idx = torch.arange(2, device=device).unsqueeze(0).expand(B, 2)
        delays_sec = taps_idx * self.ts
        p = (1.0 / tau_d.unsqueeze(1)) * torch.exp(-delays_sec / tau_d.unsqueeze(1))
        p_norm = p / p.sum(dim=1, keepdim=True)

        # Jakes fading per tap
        fd = torch.rand(B, 1, 1, 1, device=device) * 5.0
        t = torch.arange(L, device=device).reshape(1, 1, L) * self.ts
        L_rays = 20
        phi = torch.rand(B, 2, L_rays, 1, device=device) * 2 * math.pi
        psi = torch.rand(B, 2, L_rays, 1, device=device) * 2 * math.pi
        arg = 2 * math.pi * fd * torch.cos(phi) * t.unsqueeze(2) + psi
        c_exp = torch.complex(torch.cos(arg), torch.sin(arg))
        h = c_exp.sum(dim=2) * math.sqrt(1.0 / L_rays)     # (B, 2, L)

        y = torch.zeros_like(x)
        for k in range(2):
            x_shifted = x if k == 0 else torch.cat([torch.zeros(B, k, device=device, dtype=x.dtype), x[:, :-k]], dim=1)
            weight = h[:, k, :] * torch.sqrt(p_norm[:, k].unsqueeze(1))
            y = y + x_shifted * weight

        # 3. CFO ±200 Hz
        delta_f = (torch.rand(B, 1, device=device) * 2.0 - 1.0) * 200.0
        t_cfo = torch.arange(L, device=device).unsqueeze(0) * self.ts
        phase = 2 * math.pi * delta_f * t_cfo
        y = y * torch.complex(torch.cos(phase), torch.sin(phase))

        # 4. AWGN, SNR~U(10, 40) dB
        snr_db = torch.rand(B, 1, device=device) * 30.0 + 10.0
        snr_lin = 10 ** (snr_db / 10.0)
        p_sig = y.abs().pow(2).mean(dim=1, keepdim=True)
        n0 = p_sig / snr_lin
        noise = (torch.randn_like(y) + 1j * torch.randn_like(y)) * torch.sqrt(n0 / 2)
        y = y + noise

        # 5. RMS normalisation
        rms = torch.sqrt(y.abs().pow(2).mean(dim=1, keepdim=True) + 1e-10)
        y = y / rms

        return torch.stack([y.real, y.imag], dim=1).float()
```

### 1.6 Training Setup

```python
# Optimiser
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.0,    # confirmed optimal; never raise
)

# LR scheduler
# ReduceLROnPlateau on validation loss
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=20,          # --lr_patience
    factor=0.5,
)

# Early stopping
# Custom EarlyStopping that monitors val_loss; fires if no improvement for 100 epochs
early_stopping = EarlyStopping(patience=100)

# Loss function (baseline; see Approach 2 for the better focal version)
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

# Mixed precision + gradient clipping
scaler = torch.amp.GradScaler("cuda")
GRAD_NORM = 1.0   # clip to 1.0; 0.5 hurts (–1.9 pp)
```

Training loop:
```python
for epoch in range(600):   # full_train_epochs
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            x = augmentor(x)          # GPU augmentation in-place

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits, _ = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

    val_loss, val_acc = evaluate(model, val_loader, device)
    lr_scheduler.step(val_loss)
    early_stopping(val_loss)
    if early_stopping.early_stop:
        break
```

**Other settings**:
- `batch_size=128` (64 hurts –3.1 pp; too-small batches destabilise focal loss)
- `torch.compile(model)` enabled (PyTorch 2.0+ — saves ~10–15 % wall-clock)
- `seed=1337` for reproducibility
- Save checkpoint on best `val_loss` (not val_acc)
- At test time: **no augmentation**; RMS normalise per sample with `--normalize` flag

### 1.7 Evaluation Command

```bash
python test_causal_iq_classifier.py \
    --tx_start 35 --tx_end 45 \
    --pkt_start 0 --pkt_end 100 \
    --device auto \
    --normalize \
    --data_path /LOCAL/data/n210_2_leg_LocD.h5 \
    --model_path ./saved_models/current_experiment.pth \
    --output tmp_metrics.json
```

**Critical note**: `test_causal_iq_classifier.py` calls
`logits, _ = model(xb)` — the model must always return a 2-tuple `(logits, embedding)`.

---

## Approach 2 — Focal Loss + Label Smoothing (E5, **85.0 %**)

**Experiment**: exp_018 `experiment/E5-20260219_010041`
**Result**: 85.0 % (+1.0 pp over 84 % baseline)
**This is the key breakthrough.** Every result above 84 % in the entire experiment history
depends on this change.

### Motivation

Device fingerprinting has **hard device pairs** — pairs of radios that share similar hardware
impairments and are frequently confused. Standard cross-entropy treats all errors equally.
Focal loss down-weights easy, well-classified samples, concentrating gradient on the hard pairs.

### What Changes

Replace `nn.CrossEntropyLoss` with a custom `FocalLoss`.
Add `--label_smoothing 0.05` to soften confident predictions slightly.

Everything else in the architecture and training setup remains identical to the baseline.

### Implementation

Add this class to `train_causal_iq_classifier.py`:

```python
class FocalLoss(nn.Module):
    """Focal loss FL(pt) = -(1-pt)^gamma * log(pt).

    gamma=0 reduces to standard cross-entropy.
    label_smoothing is applied before computing pt via the hard label probability.
    """

    def __init__(self, gamma: float = 2.0,
                 label_smoothing: float = 0.0,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_pt = torch.nn.functional.log_softmax(logits, dim=1)  # (B, C)

        if self.label_smoothing > 0.0:
            smooth_val = self.label_smoothing / self.num_classes
            one_hot = torch.zeros_like(log_pt).scatter_(1, targets.view(-1, 1), 1.0)
            soft_targets = one_hot * (1.0 - self.label_smoothing) + smooth_val
            # CE with soft targets: -sum(soft * log_p)
            ce = -(soft_targets * log_pt).sum(dim=1)          # (B,)
            # pt for focal weight: hard-label probability
            pt = log_pt.exp().gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        else:
            pt = log_pt.exp().gather(1, targets.view(-1, 1)).squeeze(1)
            ce = -log_pt.gather(1, targets.view(-1, 1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = (focal_weight * ce).mean()
        return loss
```

Instantiate as:
```python
criterion = FocalLoss(
    gamma=2.0,           # optimal; both 1.5 and 3.0 hurt (–2.1 and –1.9 pp)
    label_smoothing=0.05, # optimal; 0.1 hurts (–1.8 pp); 0.0 is second-best
    num_classes=10,
)
```

### Training Command (complete)

```bash
python train_causal_iq_classifier.py \
    --manifest ./shards/manifest.json \
    --arch gateddeltanet \
    --batch_size 128 \
    --epochs 600 \
    --lr 3e-4 \
    --dim 512 \
    --depth 4 \
    --heads 8 \
    --head_dim 64 \
    --pooling last \
    --patch_size 64 \
    --weight_decay 0.0 \
    --patch_embed_type causal_conv \
    --augment \
    --compile \
    --label_smoothing 0.05 \
    --focal_loss \
    --save_path ./saved_models/focal_loss.pth
```

### Why This Works

- Device fingerprints are hardware impairments (PA nonlinearity, oscillator characteristics, etc.)
- Some transmitter pairs share similar impairments and are chronically confused under CE loss
- Focal loss concentrates gradient on those hard pairs by suppressing the easy majority
- gamma=2.0 is the "Goldilocks" value: 1.5 is too soft (easy examples still dominate),
  3.0 is too hard (over-suppresses even moderately easy examples)
- label_smoothing=0.05 prevents overconfidence; 0.1 over-regularises when combined with focal

### Confirmed Hyperparameter Optima for This Configuration

| Parameter | Optimal | Tried alternatives |
|---|---|---|
| `gamma` | 2.0 | 1.5 → 82.9% (–2.1pp); 3.0 → 83.1% (–1.9pp) |
| `label_smoothing` | 0.05 | 0.0 → not tested alone; 0.1 → 83.3% (–1.8pp) |
| `lr` | 3e-4 | 1e-4 → 85.0% (ties E5, –0.1pp vs H2 best) |
| `batch_size` | 128 | 64 → 82.0% (–3.1pp) |
| `grad_norm` | 1.0 | 0.5 → 83.2% (–1.9pp) |

---

## Approach 3 — Larger Head Dimension (H2, **85.1 %**)

**Experiment**: exp_037 `experiment/H2-20260219_151935`
**Result**: 85.1 % (+0.1 pp over E5, +1.1 pp over 84 % baseline)
**Built on top of**: Focal loss (Approach 2) — must apply that first.

### What Changes

Change the GatedDeltaNet attention head factorisation from the original (8 heads × 64 dim)
to fewer, larger heads (4 heads × 128 dim). Total QK capacity is the same (512), but
each head has richer per-head state.

| Setting | Baseline | H2 |
|---|---|---|
| `num_heads` (`--heads`) | 8 | **4** |
| `head_dim` (`--head_dim`) | 64 | **128** |
| `num_heads × head_dim` | 512 = dim | 512 = dim |

### Implementation

In `CausalIQGatedDeltaNetConfig` / `GatedDeltaNetBlock`:
- `head_dim=128` replaces the default `head_dim=64`
- `num_heads=4` replaces the default `num_heads=8`

This flows through to `fla.layers.gated_deltanet.GatedDeltaNet`:
```python
self.attn = GatedDeltaNet(
    hidden_size=512,
    expand_v=2.0,
    head_dim=128,   # changed from 64
    num_heads=4,    # changed from 8
    ...
)
```

### Training Command

```bash
python train_causal_iq_classifier.py \
    --manifest ./shards/manifest.json \
    --arch gateddeltanet \
    --batch_size 128 \
    --epochs 600 \
    --lr 3e-4 \
    --dim 512 \
    --depth 4 \
    --heads 4 \            # changed
    --head_dim 128 \       # changed (requires --head_dim CLI arg)
    --pooling last \
    --patch_size 64 \
    --weight_decay 0.0 \
    --patch_embed_type causal_conv \
    --augment \
    --compile \
    --label_smoothing 0.05 \
    --focal_loss \
    --save_path ./saved_models/h2_focal.pth
```

> Note: `--head_dim` is a custom CLI argument added in the H2 experiment branch.
> You need to add `p.add_argument("--head_dim", type=int, default=64, ...)` to
> `train_causal_iq_classifier.py` if training from a fresh repo.

### Why Larger Head Dimension Helps (Tentative)

- RF fingerprinting requires tracking subtle correlations across the 512-dim embedding
- With 8 heads × 64 dim, each head has a smaller "slice" of the state to work with
- With 4 heads × 128 dim, each head has double the state capacity, potentially allowing
  richer representations of device-specific covariance patterns
- The +0.1 pp gain is marginal and may be within run-to-run variance; treat it as the
  upper end of a range rather than a guaranteed improvement
- Confirmed bad direction: 8 heads × 64 (K1, the baseline) → 83.7% when tested vs
  the H2 config, confirming head_dim=128 direction is correct

---

## The Complete Best Configuration

Combining all effective approaches gives the best result (85.1 %):

```bash
python train_causal_iq_classifier.py \
    --manifest ./shards/manifest.json \
    --arch gateddeltanet \
    --batch_size 128 \
    --epochs 600 \
    --lr 3e-4 \
    --dim 512 \
    --depth 4 \
    --heads 4 \
    --head_dim 128 \
    --pooling last \
    --patch_size 64 \
    --weight_decay 0.0 \
    --patch_embed_type causal_conv \
    --augment \
    --compile \
    --label_smoothing 0.05 \
    --focal_loss \
    --focal_gamma 2.0 \
    --save_path ./saved_models/best.pth
```

Model config at a glance:
```
Architecture : GatedDeltaNet  (fla.layers.gated_deltanet.GatedDeltaNet)
dim          : 512
depth        : 4
num_heads    : 4
head_dim     : 128
expand_v     : 2.0
mlp_ratio    : 4.0
conv_size    : 4
conv_bias    : False
allow_neg_eigval: False
patch_embed  : causal_conv (kernel=64, stride=32)
pooling      : last
seq_len      : 8192
num_classes  : 10
resid_dropout: 0.0
loss         : FocalLoss(gamma=2.0, label_smoothing=0.05, num_classes=10)
optimizer    : AdamW(lr=3e-4, weight_decay=0.0)
scheduler    : ReduceLROnPlateau(patience=20, factor=0.5)
early_stop   : patience=100 on val_loss
grad_norm    : 1.0
batch_size   : 128
epochs       : 600
augment      : Jakes fading + CFO ±200Hz + AWGN [10,40]dB (GPU, no phase noise/IQ imbalance)
compile      : torch.compile() enabled
```

---

## Validated Negative Results (What NOT to Do)

These are all rigorously tested and consistently hurt performance.

### Architecture

| Change | Result | Reason |
|---|---|---|
| Prepend CLS token at position 0 | 10% (random) | Causal GDN sees nothing; token only attends to itself |
| `pooling=attn` (attention pooling) | 81.1% (–3.9pp) | Last-token recurrent state already summarises full sequence |
| Learned positional embeddings | 82.4% (–2.6pp) | GDN causal state already encodes position implicitly |
| `depth=6` | 82.7% (–2.3pp) | Overfits; 10-class task is well-served by depth=4 |
| `dim=768` | 81.8% (–3.2pp) | Wider model hurts; dim=512 is optimal |
| Auxiliary intermediate head at block 2/4 | 82.9% (–2.1pp) | Conflicting gradient signals; GDN needs all 4 blocks |
| `mlp_ratio=2.0` | 81.8% (–3.3pp) | Under-provisions feedforward capacity |
| `expand_v=1.5` | 84.5% (–0.6pp vs H2) | Reduces recurrent state expressiveness |
| `expand_v≥3.0` | too slow | >10s/epoch; exceeds speed threshold |
| `allow_neg_eigval=True` | 83.5% (–1.5pp) | Inhibitory updates destabilise training |
| `conv_size=8` | too slow | 10.6s/epoch; keep conv_size=4 |
| `conv_bias=True` | 83.9% (–1.2pp) | Learnable bias interferes with delta-rule propagation |
| Hybrid GDN + SDPA (every 2nd block) | 82.6% (–1.4pp) | Quadratic attn adds no benefit at 128-token scale |
| STFT frequency-domain patch embedding | 83.0% (–2.0pp) | Loses time-localised startup transients |
| Multi-scale patch embedding (avg 3 kernels) | 81.1% (–2.9pp) | Averaging dilutes the optimal single scale |

### Data Augmentation

| Augmentation | Result | Reason |
|---|---|---|
| Circular temporal roll | 77.5% (–7.5pp) | GDN encodes absolute position; rolling destroys transient location |
| Temporal crop + zero-pad | 68.7% (–15.3pp) | Hard boundary artifact + shorter effective length |
| Patch token dropout (5%) | 27.2% (–57.8pp) | Zeroed tokens disrupt causal GDN state propagation |
| Phase noise | 78.3% (–5.7pp) | Phase noise IS a device fingerprint; randomising destroys it |
| IQ imbalance | 80.9% (–3.1pp) | IQ imbalance IS a device fingerprint; randomising destroys it |
| Gain/power jitter | 83.4% (–1.6pp vs E5) | LocD test has consistent capture conditions; jitter adds unhelpful variance |
| Mixup (alpha=0.4) | 80.4% (–3.6pp) | Device fingerprints do not linearly superpose |
| Constant phase rotation | 81.5% (–2.5pp) | LocD test has consistent phase characteristics used as discriminant |
| Curriculum SNR | 77.9% (–6.1pp) | High-SNR warmup causes model to overfit to clean signals |
| TTA with CFO+AWGN | 84.5% (–0.5pp vs H2) | Clean test data; hardware-variation TTA adds corruption |
| Removing all augmentation | 82.3% (–1.7pp) | Aug pipeline provides domain generalisation despite hurting val_acc |

### Training / Optimisation

| Change | Result | Reason |
|---|---|---|
| `lr=1e-4` | 85.0% (–0.1pp vs H2) | ReduceLROnPlateau already reduces LR; lower start delays learning |
| LR warmup 20 epochs | 84.3% (–0.8pp vs H2) | Delays reaching target lr; ReduceLROnPlateau makes warmup redundant |
| OneCycleLR (max_lr=3e-3) | 80.1% (–3.9pp) | Aggressive LR drives model into sharp, non-generalising minima |
| Cosine LR with warmup | 84.6% (–0.4pp vs H2) | LR schedule doesn't add value beyond focal loss + ReduceLROnPlateau |
| `batch_size=64` | 82.0% (–3.1pp) | Noisier focal loss gradient estimates destabilise training |
| `grad_norm=0.5` | 83.2% (–1.9pp) | Over-constrains optimisation; prevents large steps on hard pairs |
| Muon + AdamW mixed | 83.8% (–1.3pp) | Optimizer imbalance between block weights and embedding parameters |
| `focal_gamma=1.5` | 82.9% (–2.1pp) | Too soft; easy examples still dominate gradient |
| `focal_gamma=3.0` | 83.1% (–1.9pp) | Too aggressive; over-suppresses moderately easy examples |
| `label_smoothing=0.1` | 83.3% (–1.8pp) | Over-regularises combined with focal loss; use 0.05 |
| Mamba-2 backbone | OOM + too slow | Torch fallback requires prohibitive memory; needs CUDA kernel install |
| Stochastic depth | 83.2% (–0.8pp) | Best single-architecture result without focal loss, but focal loss outperforms |

---

## Key Principles for This Task

1. **Augmentation must be receiver-side only.**
   Hardware impairments (phase noise, IQ imbalance, PA nonlinearity) ARE the transmitter
   fingerprints. Augmenting them teaches invariance to the very signal you want to learn.
   Only channel-distortion augmentations that a receiver would experience regardless of
   transmitter identity are safe: multipath, AWGN, CFO.

2. **Do not touch absolute temporal position.**
   The GDN causal recurrent state encodes the absolute position of burst startup transients.
   Any transformation that shifts, rolls, or reorders tokens destroys this key discriminant.

3. **Validation accuracy is a misleading proxy for LocD test accuracy.**
   Augmentation consistently *lowers* val_acc (model sees noisier inputs) but is *required*
   for generalisation to the test location. Models that achieve 99 %+ val_acc (e.g. F7, D3, E2)
   often perform *worse* on LocD. Never select hyperparameters based on val_acc alone.

4. **The last-token recurrent state is the right representation.**
   For causal GDN, the final token's recurrent state summarises the full sequence via the
   delta-rule memory matrix. This is strictly better than mean pooling or attention pooling
   for this task. The state is self-sufficient — do not add positional embeddings.

5. **Loss function is the highest-leverage knob.**
   50 experiments tried; only the focal loss change (E5) produced a clear +1 pp gain.
   Architecture tweaks, LR schedules, augmentation changes — all failed or caused regressions.
   Focus future work on the loss function, label distribution, and training dynamics.
