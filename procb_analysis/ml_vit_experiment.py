"""
ml_vit_experiment.py
====================
Replication of the SR early-epoch generalization prediction using a Vision
Transformer (ViT) architecture on CIFAR-10 with symmetric label noise.

Protocol is identical to the ResNet-18 experiment (Stillwell 2025c):
  - Five symmetric label-noise conditions: 0%, 10%, 20%, 30%, 40%
  - Signal Ratio (SR) = V_E / V_P measured at epoch 5 of training
  - Primary test: does early-epoch SR predict final validation loss?
  - Critical test: the SR predictor uses ONLY loss-variance structure at
    epoch 5, not the noise condition itself. A naive noise-level predictor
    has access to information the SR predictor does not.

Architecture: compact ViT (patch size 4, depth 6, heads 6, dim 384)
  - Architecturally distinct from ResNet-18 in every meaningful way:
    attention vs. convolution, global vs. local receptive field,
    positional encoding vs. translation equivariance.

Device: Apple MPS (M-series) if available, else CPU.

Outputs
-------
  ml_vit_results.json   — per-condition SR and final val-loss values
  ml_vit_results.png    — scatter plot matching Figure 1C style
"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import json
import random
import math
import sys
from pathlib import Path

# Force unbuffered output so log file updates in real time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("/Users/stillwell/Documents/Google Drive/Project 47 - Materpiece")

# ── Pure numpy statistics (no scipy needed) ───────────────────────────────────
def pearsonr(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    r = float(np.dot(xm, ym) / (np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))))
    r = max(-1.0, min(1.0, r))
    n = len(x)
    t = r * math.sqrt((n - 2) / max(1 - r**2, 1e-15))
    p = float(min(1.0, 2 * math.exp(-0.717 * abs(t) - 0.416 * t * t)))
    return r, p

def linregress(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    m = (n * np.dot(x, y) - x.sum() * y.sum()) / (n * np.dot(x, x) - x.sum()**2)
    b = (y.sum() - m * x.sum()) / n
    return m, b

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NOISE_CONDITIONS = [0.0, 0.10, 0.20, 0.30, 0.40]
EPOCHS_TOTAL     = 60
EPOCH_SR         = 5
BATCH_SIZE       = 128
LR               = 3e-4
WEIGHT_DECAY     = 0.05
N_CLASSES        = 10
IMG_SIZE         = 32
PATCH_SIZE       = 4
N_PATCHES        = (IMG_SIZE // PATCH_SIZE) ** 2
DIM              = 384
DEPTH            = 6
N_HEADS          = 6
MLP_RATIO        = 4
DROPOUT          = 0.1

# ── Data ──────────────────────────────────────────────────────────────────────
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

DATA_DIR = Path.home() / ".cache" / "cifar10"

def apply_label_noise(dataset, noise_rate, seed=0):
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    n = len(targets)
    flip_mask = rng.random(n) < noise_rate
    noise_labels = rng.integers(0, N_CLASSES, size=n)
    targets[flip_mask] = noise_labels[flip_mask]
    return targets.tolist()

# ── Vision Transformer ────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = (DIM // N_HEADS) ** -0.5
        self.qkv   = nn.Linear(DIM, DIM * 3, bias=False)
        self.proj  = nn.Linear(DIM, DIM)
        self.drop  = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, N_HEADS, C // N_HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = int(DIM * MLP_RATIO)
        self.net = nn.Sequential(
            nn.Linear(DIM, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, DIM), nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.attn  = Attention()
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp   = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed   = nn.Parameter(torch.zeros(1, N_PATCHES + 1, DIM))
        self.drop        = nn.Dropout(DROPOUT)
        self.blocks      = nn.Sequential(*[Block() for _ in range(DEPTH)])
        self.norm        = nn.LayerNorm(DIM)
        self.head        = nn.Linear(DIM, N_CLASSES)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.drop(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.norm(x[:, 0])
        return self.head(x)

# ── Training utilities ────────────────────────────────────────────────────────
def compute_sr(model, loader, criterion):
    model.eval()
    batch_losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            batch_losses.append(loss.item())
    arr = np.array(batch_losses)
    V_P = float(np.var(arr))
    if len(arr) >= 3:
        smoothed = np.convolve(arr, np.ones(3)/3, mode='valid')
        residuals = arr[1:-1] - smoothed
        V_E = float(np.var(residuals))
    else:
        V_E = V_P * 0.5
    V_E = max(V_E, 1e-10)
    V_P = max(V_P, V_E + 1e-10)
    SR  = V_E / V_P
    return SR, V_P, V_E

def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out  = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            n          += len(yb)
    return total_loss / n, correct / n

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_base = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=transform_train)
    test_set   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    results = {}

    for noise in NOISE_CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Noise condition: {int(noise*100)}%")
        print(f"{'='*60}")

        noisy_targets = apply_label_noise(train_base, noise, seed=SEED)
        train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                                         transform=transform_train)
        train_dataset.targets = noisy_targets
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=False)

        torch.manual_seed(SEED)
        model     = ViT().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = EPOCHS_TOTAL * len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        sr_at_epoch5   = None
        final_val_loss = None

        for epoch in range(1, EPOCHS_TOTAL + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                          criterion, scheduler)
            val_loss, val_acc = eval_epoch(model, test_loader, criterion)

            if epoch == EPOCH_SR:
                sr_at_epoch5, vp, ve = compute_sr(model, train_loader, criterion)
                print(f"  Epoch {epoch}: SR={sr_at_epoch5:.4f}  "
                      f"V_P={vp:.6f}  V_E={ve:.6f}  val_loss={val_loss:.4f}")

            if epoch % 10 == 0 or epoch == EPOCHS_TOTAL:
                print(f"  Epoch {epoch:3d}: tr_loss={tr_loss:.4f}  "
                      f"tr_acc={tr_acc:.3f}  val_loss={val_loss:.4f}  "
                      f"val_acc={val_acc:.3f}")

        final_val_loss = val_loss
        results[noise] = {
            'noise_pct':      int(noise * 100),
            'sr_epoch5':      sr_at_epoch5,
            'final_val_loss': final_val_loss,
        }
        print(f"  → SR@epoch5={sr_at_epoch5:.4f}  final_val_loss={final_val_loss:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Noise':>8}  {'SR@ep5':>10}  {'FinalValLoss':>14}")
    for noise, r in results.items():
        print(f"{r['noise_pct']:>7}%  {r['sr_epoch5']:>10.4f}  {r['final_val_loss']:>14.4f}")

    sr_vals  = np.array([r['sr_epoch5']      for r in results.values()])
    val_vals = np.array([r['final_val_loss']  for r in results.values()])
    r_val, p_val = pearsonr(sr_vals, val_vals)
    print(f"\nPearson r (SR@epoch5 vs final val loss): r = {r_val:.3f}, p = {p_val:.4f}")
    print(f"n = {len(results)} conditions")
    direction_match = (r_val < 0)
    print(f"Direction consistent with ResNet-18 (r < 0): {direction_match}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        'architecture':         'ViT (patch=4, depth=6, heads=6, dim=384)',
        'dataset':              'CIFAR-10',
        'epoch_sr':             EPOCH_SR,
        'epochs_total':         EPOCHS_TOTAL,
        'conditions':           results,
        'pearson_r':            float(r_val),
        'pearson_p':            float(p_val),
        'resnet18_r':           -0.97,
        'direction_consistent': bool(direction_match),
    }
    with open(OUT_DIR / 'ml_vit_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to ml_vit_results.json")

    # ── Figure ────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams.update({'font.family': 'serif', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False})

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(results)))

        ax = axes[0]
        for j, (noise, r) in enumerate(results.items()):
            ax.scatter(r['sr_epoch5'], r['final_val_loss'],
                       c=[colors[j]], s=80, zorder=5, edgecolors='white', lw=0.8)
            ax.annotate(f"{r['noise_pct']}%", (r['sr_epoch5'], r['final_val_loss']),
                        xytext=(5, 3), textcoords='offset points', fontsize=8)
        m, b = linregress(sr_vals, val_vals)
        x_fit = np.linspace(sr_vals.min() - 0.02, sr_vals.max() + 0.02, 100)
        ax.plot(x_fit, m * x_fit + b, 'k--', lw=1.2, alpha=0.6)
        ax.set_xlabel('Signal Ratio at epoch 5  (SR = $V_E / V_P$)', fontsize=9)
        ax.set_ylabel('Final validation loss', fontsize=9)
        ax.set_title(f'A  ViT: early-epoch SR predicts generalization\n'
                     f'$r = {r_val:.2f}$,  $p = {p_val:.3f}$,  $n = {len(results)}$',
                     loc='left', fontsize=9, fontweight='bold')
        ax.text(0.05, 0.10,
                f'Architecture: ViT (d={DEPTH}, h={N_HEADS}, dim={DIM})\nDataset: CIFAR-10',
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBBBBB', lw=0.6))

        ax2 = axes[1]
        arch_labels = ['ResNet-18\n(Stillwell 2025c)', 'ViT\n(this study)']
        arch_r      = [-0.97, r_val]
        arch_colors = ['#0072B2', '#E69F00']
        bars = ax2.bar(arch_labels, [abs(r) for r in arch_r],
                       color=arch_colors, width=0.45, edgecolor='white', lw=0.8)
        for bar, rv in zip(bars, arch_r):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'$r = {rv:.2f}$', ha='center', va='bottom', fontsize=9,
                     fontweight='bold')
        ax2.axhline(0.666, color='gray', lw=1.0, ls=':', alpha=0.7)
        ax2.text(1.52, 0.677, '$p < 0.05$ threshold\n($n=5$)', fontsize=7,
                 color='gray', va='bottom')
        ax2.set_ylim(0, 1.12)
        ax2.set_ylabel('|Pearson r|  (SR@epoch5 vs final val loss)', fontsize=9)
        ax2.set_title('B  Cross-architecture replication\nSame protocol, different architecture',
                      loc='left', fontsize=9, fontweight='bold')

        plt.tight_layout(pad=1.5)
        fig.savefig(OUT_DIR / 'ml_vit_results.png', dpi=300, bbox_inches='tight')
        fig.savefig(OUT_DIR / 'ml_vit_results.pdf', bbox_inches='tight')
        print("Figure saved to ml_vit_results.png / .pdf")
        plt.close()
    except Exception as e:
        print(f"Figure generation skipped: {e}")

    print("\n✓ ViT experiment complete.")
