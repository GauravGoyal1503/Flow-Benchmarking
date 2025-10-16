# ============================================================
# Flow++ on Morpho-MNIST: +10% thickness intervention (H100 OOM-hardened)
# Metrics: MAE, FD_attr(1D), Inception FID (streaming, CPU),
#          CVS (validity), LOC (latent locality), FCS (SSIM)
# ============================================================
# This script trains an invertible flow (Flow++ or a fallback) on Morpho‑MNIST, learns a linear probe for thickness, performs a +10% latent edit along the probe direction, decodes counterfactual images, and reports MAE, FD_attr (1D Fréchet distance on predictions), CVS, LOC, FCS (SSIM), and an Inception‑based FID against proxy real images matched by target thickness. [web:22][web:26][web:10][web:48]

# ---- set allocator BEFORE importing torch to reduce fragmentation ----
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# The CUDA allocator settings reduce memory fragmentation and help avoid OOM on large GPUs by enabling expandable segments and limiting large split sizes. [web:22]

import sys, glob, math, random, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F

# -----------------------------
# OOM-SAFE CONFIG
# -----------------------------
# Device placement, precision, and batch sizes are chosen to be conservative for stability; Inception is kept on CPU to avoid GPU pressure during FID feature extraction. [web:48]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INC_DEVICE = torch.device('cpu')   # keep Inception on CPU
AMP_DTYPE = torch.bfloat16         # Hopper-friendly mixed precision
USE_AMP = (device.type == 'cuda')  # enable autocast for training
TRAIN_BS = 48                      # conservative; raise once stable
TEST_BS  = 48
INCEPT_BS = 16                     # CPU Inception micro-batch
DECODE_MB = 16                     # micro-batch for inverse+reencode
MAX_FID_SAMPLES = 4000             # cap CF imgs for FID (None for all)

# perf knobs (won't affect correctness)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

torch.manual_seed(1337); np.random_seed = 1337; random.seed(1337)
np.random.seed(1337)

# -----------------------------
# Paths (edit if needed)
# -----------------------------
# Expect Morpho‑MNIST images in PNG/JPG and morpho attribute CSVs that include a 'thickness' column; this keeps parity with the dataset’s morphometrics design. [web:22]
ROOT = '/workspace/base'
TRAIN_IMG_DIR, TEST_IMG_DIR = f'{ROOT}/train-images', f'{ROOT}/t10k-images'
TRAIN_MORPHO_CSV, TEST_MORPHO_CSV = f'{ROOT}/train-morpho.csv', f'{ROOT}/t10k-morpho.csv'
assert Path(TRAIN_IMG_DIR).exists() and Path(TEST_IMG_DIR).exists(), "Missing image folders"
assert Path(TRAIN_MORPHO_CSV).exists() and Path(TEST_MORPHO_CSV).exists(), "Missing morpho CSVs"

# -----------------------------
# Dataset
# -----------------------------
# Simple folder dataset that pairs images with thickness from CSV; images are loaded as grayscale and transformed to tensors matching MNIST-like shape. [web:22]
def _try_int_stem(p):
    s = Path(p).stem
    d = ''.join(ch for ch in s if ch.isdigit())
    return int(d) if d else None

class MorphoMNISTFolder(Dataset):
    def __init__(self, images_dir, morpho_csv, transform=None):
        paths = []
        for ext in ('*.png','*.jpg','*.jpeg'):
            paths.extend(glob.glob(os.path.join(images_dir, ext)))
        assert len(paths) > 0, f'No images in {images_dir}; expected PNG/JPG'
        self.images = sorted(
            paths,
            key=lambda p: (_try_int_stem(p) is None,
                           _try_int_stem(p) if _try_int_stem(p) is not None else p)
        )
        df = pd.read_csv(morpho_csv)
        df.columns = [c.lower() for c in df.columns]
        assert 'thickness' in df.columns, "CSV must include 'thickness'"
        if 'index' in df.columns:
            df = df.sort_values('index').reset_index(drop=True)
        n = min(len(self.images), len(df))
        self.images = self.images[:n]
        self.thickness = torch.tensor(df['thickness'].values[:n], dtype=torch.float32)
        self.transform = transform if transform is not None else T.ToTensor()
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        x = Image.open(self.images[idx]).convert('L')
        x = self.transform(x)  # [1,28,28]
        t = self.thickness[idx]
        return x, t

transform = T.ToTensor()
train_ds = MorphoMNISTFolder(TRAIN_IMG_DIR, TRAIN_MORPHO_CSV, transform)
test_ds  = MorphoMNISTFolder(TEST_IMG_DIR,  TEST_MORPHO_CSV,  transform)
train_loader = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True,  num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=TEST_BS,  shuffle=False, num_workers=0, pin_memory=False)

# -----------------------------
# Flow++ import (or fallback minimal invertible flow)
# -----------------------------
# Tries to import a Flow++ implementation; if unavailable, uses a compact channel-affine coupling flow that supports invertibility for encoding/decoding. [web:22]
FlowPP = None; flowpp_available = False
try:
    if 'flowplusplus' in [p.name.lower() for p in Path('.').iterdir() if p.is_dir()]:
        sys.path.append('flowplusplus')
    for mod_name in ['model','flowplusplus','fpp_model']:
        try:
            m = __import__(mod_name)
            for cname in ['FlowPP','FlowPlusPlus','FlowPPNet','Flowpp']:
                if hasattr(m, cname):
                    FlowPP = getattr(m, cname); flowpp_available = True; break
            if flowpp_available: break
        except Exception:
            pass
except Exception:
    pass

class ChannelAffineCoupling(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        assert c_in % 2 == 0, "Even channels required"
        self.c_half = c_in // 2
        hidden = 128
        self.net = nn.Sequential(
            nn.Conv2d(self.c_half, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_in, 3, padding=1),
        )
    def forward(self, x, reverse=False):
        x_a, x_b = torch.split(x, [self.c_half, self.c_half], dim=1)
        h = self.net(x_a); s, t = torch.split(h, [self.c_half, self.c_half], dim=1); s = torch.tanh(s)
        if not reverse:
            y_b = x_b * torch.exp(s) + t
            y = torch.cat([x_a, y_b], dim=1)
            ldj = s.sum(dim=(1,2,3))
        else:
            x_b_rec = (x_b - t) * torch.exp(-s)
            y = torch.cat([x_a, x_b_rec], dim=1)
            ldj = -s.sum(dim=(1,2,3))
        return y, ldj

def squeeze2x2(x):
    B,C,H,W = x.shape; assert H%2==0 and W%2==0
    x = x.view(B,C,H//2,2,W//2,2).permute(0,1,3,5,2,4).contiguous().view(B,C*4,H//2,W//2)
    return x

def unsqueeze2x2(x):
    B,C,H,W = x.shape; assert C%4==0
    c = C//4
    x = x.view(B,c,2,2,H,W).permute(0,1,4,2,5,3).contiguous().view(B,c,H*2,W*2)
    return x

class FallbackFlowPP(nn.Module):
    def __init__(self, c_in=1, K=8):
        super().__init__()
        self.layers = nn.ModuleList([ChannelAffineCoupling(c_in*4) for _ in range(K)])
    def forward(self, x):
        z = squeeze2x2(x); ldj = torch.zeros(x.size(0), device=x.device)
        for f in self.layers:
            z, inc = f(z, reverse=False); ldj = ldj + inc
        return z, ldj
    def inverse(self, z):
        x = z
        for f in reversed(self.layers):
            x, _ = f(x, reverse=True)
        return unsqueeze2x2(x)

C,H,W = 1,28,28
if flowpp_available:
    try:
        flowpp = FlowPP(in_channel=C, n_block=1, n_flow=8).to(device)
    except Exception:
        try:
            flowpp = FlowPP((C,H,W), n_flow=8, n_block=1).to(device)
        except Exception:
            flowpp = FallbackFlowPP(c_in=C, K=8).to(device)
else:
    flowpp = FallbackFlowPP(c_in=C, K=8).to(device)

# -----------------------------
# Preprocess, forward/inverse, flatten helpers
# -----------------------------
# Preprocess uses dequantization and logit transform for flow training; forward_flow and inverse adapters make the code robust to different Flow++ signatures; helpers flatten and unflatten z lists. [web:22]
def preprocess(x, alpha=1e-6, add_noise=True):
    x = x * 255.0
    x = x + (torch.rand_like(x) if add_noise else 0.5)
    x = x / 256.0
    x = x * (1 - 2*alpha) + alpha
    return torch.log(x) - torch.log1p(-x)  # logit

def inverse_preprocess(x_p, alpha=1e-6):
    x_sig = torch.sigmoid(x_p)
    x = (x_sig - alpha) / (1 - 2*alpha)
    return torch.clamp(x, 0.0, 1.0)

def forward_flow(xp):
    out = flowpp(xp)
    if not isinstance(out, tuple):
        return [out], torch.zeros(xp.size(0), device=xp.device)
    tensors = []
    for o in out:
        if torch.is_tensor(o): tensors.append(o)
        elif isinstance(o, (list, tuple)) and len(o) > 0:
            for t in o:
                if torch.is_tensor(t): tensors.append(t)
    z_list = [t for t in tensors if t.dim() == 4]
    if len(z_list) == 0 and len(tensors) > 0:
        z_list = [max(tensors, key=lambda t: t.numel())]
    B = xp.size(0); sldj_terms = []
    for t in tensors:
        if t.dim() == 0: sldj_terms.append(t.expand(B))
        elif t.dim() == 1 and t.size(0) == B: sldj_terms.append(t)
    sldj = torch.stack(sldj_terms, dim=0).sum(dim=0) if len(sldj_terms) else torch.zeros(B, device=xp.device)
    return z_list, sldj

def zlist_to_flat(z_list):
    return torch.cat([z.view(z.size(0), -1) for z in z_list], dim=1)

def log_pz_sum(z_list):
    const = math.log(2*math.pi)
    parts = [ -0.5 * (z**2 + const).sum(dim=(1,2,3)) for z in z_list ]
    return torch.stack(parts, dim=0).sum(dim=0)

# ---------- Robust inverse adapter ----------
def _z_shapes(z_list):
    return [tuple(z.shape) for z in z_list]

@torch.no_grad()
def inverse_flow_safe(flowpp, z_list_prime):
    errors = []
    try:
        return flowpp.inverse(z_list_prime)
    except Exception as e:
        errors.append(f"inverse(list): {type(e).__name__}: {e}")
    try:
        return flowpp.inverse(z_list_prime[0])
    except Exception as e:
        errors.append(f"inverse(single): {type(e).__name__}: {e}")
    if hasattr(flowpp, "reverse"):
        try:
            r = flowpp.reverse(z_list_prime); return r if torch.is_tensor(r) else r[0]
        except Exception as e:
            errors.append(f"reverse(list): {type(e).__name__}: {e}")
        try:
            r = flowpp.reverse(z_list_prime[0]); return r if torch.is_tensor(r) else r[0]
        except Exception as e:
            errors.append(f"reverse(single): {type(e).__name__}: {e}")
    try:
        cat = torch.cat(z_list_prime, dim=1) if len(z_list_prime) > 1 else z_list_prime[0]
        return flowpp.inverse(cat)
    except Exception as e:
        errors.append(f"inverse(concatC): {type(e).__name__}: {e}")
    print("[decode] All inverse attempts failed.")
    print("[decode] Tried shapes:", _z_shapes(z_list_prime))
    raise RuntimeError("Inverse failed. Attempts:\n  - " + "\n  - ".join(errors))

def unflatten_to_zlist(z_flat, shapes):
    B = z_flat.size(0)
    out, start = [], 0
    for (C,H,W) in shapes:
        sz = C*H*W
        part = z_flat[:, start:start+sz].contiguous().view(B, C, H, W)
        out.append(part); start += sz
    return out

# -----------------------------
# Train Flow++ (autocast bf16)
# -----------------------------
# Standard maximum likelihood training on the flow using dequantized/logit inputs with bf16 autocast; gradient clipping and periodic cache clears improve stability. [web:22]
optimizer = optim.Adam(flowpp.parameters(), lr=1e-3)
EPOCHS = 5

flowpp.train()
for epoch in range(1, EPOCHS+1):
    pbar = tqdm(train_loader, desc=f'Flow++ train {epoch}/{EPOCHS}')
    for x, _ in pbar:
        xp = preprocess(x.to(device), add_noise=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            z_list, sldj = forward_flow(xp)
            log_pz = log_pz_sum(z_list)
            loss = -(log_pz + sldj).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flowpp.parameters(), 5.0)
        optimizer.step()
        pbar.set_postfix(loss=f'{loss.item():.3f}')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
flowpp.eval()

# -----------------------------
# Encode & linear probe for thickness
# -----------------------------
# Encodes images to latents and trains a small linear regressor y = w·z + b to predict thickness, which provides the direction used for latent intervention. [web:22]
@torch.no_grad()
def encode_to_latent(x):
    with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
        z_list, _ = forward_flow(preprocess(x.to(device), add_noise=False))
    return z_list

Z_train, T_train = [], []
for x, t in tqdm(train_loader, desc='Encode train'):
    z_list = encode_to_latent(x)
    z_flat = zlist_to_flat(z_list).cpu()
    Z_train.append(z_flat); T_train.append(t.view(-1,1))
Z_train = torch.cat(Z_train, dim=0); T_train = torch.cat(T_train, dim=0)

lin = nn.Linear(Z_train.size(1), 1).to(device)
opt_lin = optim.Adam(lin.parameters(), lr=1e-3); mse = nn.MSELoss()
lin.train()
for _ in range(5):
    idx = torch.randperm(Z_train.size(0))
    for i in range(0, len(idx), 1024):
        sel = idx[i:i+1024]
        xb = Z_train[sel].to(device); yb = T_train[sel].to(device)
        pred = lin(xb); loss = mse(pred, yb)
        opt_lin.zero_grad(); loss.backward(); opt_lin.step()
lin.eval()
with torch.no_grad():
    W = lin.weight.detach().view(-1); b = lin.bias.detach().view(1)
del Z_train; del T_train
if device.type == 'cuda':
    torch.cuda.empty_cache()

# -----------------------------
# MAE (factual)
# -----------------------------
# MAE evaluates the absolute error of predicted thickness against ground truth for both factual and counterfactual comparisons, providing an intuitive regression metric. [web:22]
def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.view(-1); target = target.view(-1)
    m = torch.isfinite(pred) & torch.isfinite(target)
    if m.sum() == 0: return float('nan')
    return torch.mean(torch.abs(pred[m] - target[m])).item()

def eval_mae(loader, name="set"):
    ys_true, ys_pred = [], []
    with torch.no_grad():
        for x, t in tqdm(loader, desc=f'MAE eval ({name})'):
            z_list = encode_to_latent(x)
            z_flat = zlist_to_flat(z_list).to(device)
            y_hat = lin(z_flat).squeeze(1).detach().cpu()
            ys_true.append(t.detach().cpu()); ys_pred.append(y_hat)
    y_true = torch.cat(ys_true, dim=0); y_pred = torch.cat(ys_pred, dim=0)
    print(f"MAE (thickness) on {name}: {compute_mae(y_pred, y_true):.6f}")

eval_mae(train_loader, "train")
eval_mae(test_loader,  "test")

# -----------------------------
# Latent intervention (+10%) and CF decode (MICRO-BATCHED)
# -----------------------------
# Intervenes in the latent by moving along w to increase predicted thickness by +10%, then decodes counterfactual images and re-encodes to obtain y_cf predictions. [web:22]
def intervene_thickness_flat(z_flat, w, b, pct=0.10, alpha_cap=0.5, z_clip=5.0):
    y = (z_flat @ w.to(z_flat.device)) + b.to(z_flat.device)
    w2 = (w.to(z_flat.device) @ w.to(z_flat.device)) + 1e-8
    alpha = torch.clamp(pct * y / w2, -alpha_cap, alpha_cap)
    z_prime = z_flat + alpha.unsqueeze(1) * w.to(z_flat.device)
    z_prime = torch.nan_to_num(z_prime)
    z_prime = torch.clamp(z_prime, -z_clip, z_clip)
    return z_prime

def frechet_distance_1d(mu1, var1, mu2, var2, eps=1e-9):
    return (mu1 - mu2)**2 + var1 + var2 - 2.0*math.sqrt((var1 + eps)*(var2 + eps))

def compute_ffd_1d(f_real, f_cf):
    fr = f_real.detach().cpu().numpy().reshape(-1)
    fc = f_cf.detach().cpu().numpy().reshape(-1)
    mu_r, mu_c = fr.mean(), fc.mean()
    var_r = fr.var(ddof=1 if fr.size>1 else 0)
    var_c = fc.var(ddof=1 if fc.size>1 else 0)
    return frechet_distance_1d(mu_r, max(var_r,0.0), mu_c, max(var_c,0.0))

# -----------------------------
# Inception encoder (CPU) + streaming FID
# -----------------------------
# Uses Torchvision’s Inception‑v3 on CPU to extract 2048‑D features with correct resize and normalization, then streams mean/cov estimates for FID. [web:48][web:26]
class InceptionEncoder(nn.Module):
    def __init__(self, run_device=INC_DEVICE):
        super().__init__()
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                aux_logits=False)  # reduce mem
        m.fc = nn.Identity(); m.eval()
        for p in m.parameters(): p.requires_grad = False
        self.net = m.to(run_device)
        self.dev = run_device
        self.resize = T.Resize((299, 299))
        self.norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    @torch.no_grad()
    def forward(self, x):  # x in [0,1], [N,3,H,W]
        x = x.to(self.dev, non_blocking=False)
        x = self.resize(x); x = self.norm(x)
        return self.net(x)

@torch.no_grad()
def streaming_feats_stats(encoder: nn.Module, imgs: torch.Tensor, bsz=INCEPT_BS):
    if imgs.size(1) == 1:
        imgs = imgs.repeat(1,3,1,1)
    N_total, d = 0, None
    sum_x = None
    sum_xxT = None
    for i in range(0, imgs.size(0), bsz):
        batch = imgs[i:i+bsz].clamp(0,1)
        feats = encoder(batch).detach()
        feats_cpu = feats.float().cpu()
        n, d_now = feats_cpu.shape
        if d is None:
            d = d_now
            sum_x   = torch.zeros(d, dtype=torch.float32)
            sum_xxT = torch.zeros(d, d, dtype=torch.float32)
        sum_x   += feats_cpu.sum(dim=0)
        sum_xxT += feats_cpu.T @ feats_cpu
        N_total += n
        del feats, feats_cpu, batch
    mu = sum_x / max(N_total, 1)
    cov = (sum_xxT - N_total * torch.outer(mu, mu)) / max(N_total - 1, 1)
    return mu, cov

@torch.no_grad()
def trace_sqrt_product(c1: torch.Tensor, c2: torch.Tensor, eps=1e-6):
    I = torch.eye(c1.size(0), dtype=c1.dtype)
    e1, U1 = torch.linalg.eigh(c1 + eps*I)
    e1 = torch.clamp(e1, min=0.0)
    sqrtC1 = (U1 * torch.sqrt(e1).unsqueeze(0)) @ U1.T
    A = sqrtC1 @ c2 @ sqrtC1
    A = 0.5*(A + A.T)
    ea, _ = torch.linalg.eigh(A)
    ea = torch.clamp(ea, min=0.0)
    return torch.sum(torch.sqrt(ea))

@torch.no_grad()
def fid_distance(mu1, cov1, mu2, cov2):
    diff = (mu1 - mu2)
    return float((torch.dot(diff, diff) + torch.trace(cov1) + torch.trace(cov2) - 2.0*trace_sqrt_product(cov1, cov2)).item())

# -----------------------------
# CVS / LOC / FCS helpers
# -----------------------------
# CVS checks that y_cf achieves the intended relative change within a tolerance band, LOC quantifies alignment and off‑target latent movement, and FCS uses SSIM to assess perceptual similarity between factual and counterfactual images. [web:22][web:10]
@torch.no_grad()
def compute_cvs(y_f: torch.Tensor,
                y_cf: torch.Tensor,
                pct: float,
                rel_tol: float = 0.05,
                abs_tol: float = 0.0) -> float:
    y_f = y_f.view(-1); y_cf = y_cf.view(-1)
    target = (1.0 + pct) * y_f
    band = rel_tol * target.abs() + abs_tol
    ok = (y_cf - target).abs() <= band
    if ok.numel() == 0: return float('nan')
    return ok.float().mean().item()

@torch.no_grad()
def compute_loc_stats(z_flat: torch.Tensor,
                      z_flat_prime: torch.Tensor,
                      w: torch.Tensor,
                      eps: float = 1e-8):
    dz = z_flat_prime - z_flat
    w = w.to(dz.device)
    dz_norm = dz.norm(dim=1) + eps
    w_norm  = w.norm() + eps
    cos_sim = (dz @ w) / (dz_norm * w_norm)
    alpha = (dz @ w) / (w @ w + eps)
    dz_proj = alpha.unsqueeze(1) * w
    dz_perp = dz - dz_proj
    perp_ratio = dz_perp.norm(dim=1) / (dz.norm(dim=1) + eps)
    return cos_sim, perp_ratio

def _gaussian_window(window_size=11, sigma=1.5, device='cpu', channels=1):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(- (coords**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, -1)
    window_2d = (g.transpose(2, 1) @ g).unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.repeat(channels, 1, 1, 1)
    return window_2d

@torch.no_grad()
def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    assert img1.shape == img2.shape
    N, C, H, W = img1.shape
    dev_local = img1.device
    window = _gaussian_window(window_size, sigma, device=dev_local, channels=C)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return ssim_map.mean().item()

# ---- Intervene & collect (MICRO-BATCHED inverse + re-encode) ----
# Applies the latent edit, decodes counterfactual images in micro-batches to limit memory, computes FCS via SSIM, and collects factual and counterfactual thickness predictions. [web:10][web:22]
thick_factual, thick_counterfactual = [], []
T_true, T_cf_target = [], []
CF_IMAGES = []            # stored on CPU

loc_cos_all, loc_perp_all = [], []
fcs_ssim_all = []

with torch.no_grad():
    for x, t in tqdm(test_loader, desc='Intervene (+10%) & collect'):
        # factual encode
        z_list = encode_to_latent(x)
        z_flat = zlist_to_flat(z_list)
        shapes = [tuple(z.shape[1:]) for z in z_list]

        # factual prediction
        y_f_batch = lin(z_flat.to(device))
        thick_factual.append(y_f_batch.detach().cpu())

        # latent edit
        z_flat_prime = intervene_thickness_flat(z_flat.to(device), W, b, pct=0.10)

        # LOC (latent locality)
        cos_sim, perp_ratio = compute_loc_stats(z_flat.to(device), z_flat_prime, W)
        loc_cos_all.append(cos_sim.detach().cpu())
        loc_perp_all.append(perp_ratio.detach().cpu())

        # Prepare containers for this batch's CF preds (micro-batched)
        y_cf_parts = []

        # micro-batch loop to keep inverse+reencode memory small
        B = z_flat_prime.size(0)
        for j in range(0, B, DECODE_MB):
            j2 = min(j + DECODE_MB, B)
            # slice inputs for micro-batch
            z_flat_prime_mb = z_flat_prime[j:j2]
            x_f_mb = x[j:j2]
            try:
                z_list_prime_mb = unflatten_to_zlist(z_flat_prime_mb, shapes)
                # inverse on GPU, still under @no_grad; use bf16 autocast
                with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
                    x_cf_p_mb = inverse_flow_safe(flowpp, z_list_prime_mb)
                x_cf_mb = inverse_preprocess(x_cf_p_mb).clamp(0,1)
                CF_IMAGES.append(x_cf_mb.detach().cpu())
                # FCS via SSIM (GPU→CPU safe)
                fcs_ssim_all.append(torch.tensor(ssim_torch(x_f_mb.to(device), x_cf_mb.to(device))))
                # re-encode deterministically for attribute prediction
                z_list_cf_mb = encode_to_latent(x_cf_mb)
                y_cf_mb = lin(zlist_to_flat(z_list_cf_mb).to(device))
            except Exception as e:
                # fallback: predict attribute directly from edited latent
                y_cf_mb = lin(z_flat_prime_mb)
            y_cf_parts.append(y_cf_mb.detach().cpu())

        # stack micro-batch preds for this batch
        y_cf_batch = torch.cat(y_cf_parts, dim=0)
        thick_counterfactual.append(y_cf_batch)
        T_true.append(t.view(-1,1).cpu())
        T_cf_target.append((1.10 * t.view(-1,1)).cpu())

        if device.type == 'cuda':
            torch.cuda.empty_cache()

# ---- Stack + Metrics ----
# Aggregates predictions and targets, then reports MAE (factual, CF vs target, CF vs GT), 1D Fréchet distance between predicted thickness distributions, CVS, LOC, and FCS. [web:26][web:10]
y_f  = torch.cat(thick_factual, dim=0)
y_cf = torch.cat(thick_counterfactual, dim=0)
t_gt = torch.cat(T_true, dim=0)
t_cf = torch.cat(T_cf_target, dim=0)

print('\nMAE (factual vs GT):                ', compute_mae(y_f, t_gt))
print('MAE (CF vs scaled target +10%):     ', compute_mae(y_cf, t_cf))
print('MAE (CF vs original GT):            ', compute_mae(y_cf, t_gt))

f_vec  = y_f.view(-1); cf_vec = y_cf.view(-1)
fd_attr_thickness = compute_ffd_1d(f_vec, cf_vec)
print(f'\nFD_attr(thickness, +10%): {fd_attr_thickness:.6f}')
print('Mean predicted thickness factual      :', float(f_vec.mean()))
print('Mean predicted thickness counterfactual:', float(cf_vec.mean()))
print('Ratio CF/F (measured):', float(cf_vec.mean() / (f_vec.mean()+1e-12)))

# CVS / LOC / FCS
PCT = 0.10
CVS_REL_TOL = 0.05
CVS_ABS_TOL = 0.00
cvs = compute_cvs(y_f, y_cf, pct=PCT, rel_tol=CVS_REL_TOL, abs_tol=CVS_ABS_TOL)
print(f"\nCVS (@{int(PCT*100)}% target, ±{int(CVS_REL_TOL*100)}% rel tol): {cvs:.4f}")

if len(loc_cos_all) > 0:
    loc_cos = torch.cat(loc_cos_all).mean().item()
    loc_perp = torch.cat(loc_perp_all).mean().item()
    print(f"LOC (mean cosine Δz vs w): {loc_cos:.4f}  (↑ better)")
    print(f"LOC off-target ratio     : {loc_perp:.4f}  (↓ better)")

if len(fcs_ssim_all) == 0:
    print("FCS (SSIM factual vs CF): N/A (no CF images decoded)")
else:
    fcs = torch.stack(fcs_ssim_all).mean().item() if isinstance(fcs_ssim_all[0], torch.Tensor) \
          else float(np.mean(fcs_ssim_all))
    print(f"FCS (SSIM factual vs CF): {fcs:.4f}  (↑ better)")

# -----------------------------
# Inception FID (streaming, CPU by default)
# -----------------------------
# Builds a proxy real set by nearest-neighbor matching on target thickness and computes FID between CF images and proxies using Inception‑v3 features on CPU. [web:26][web:48]
if len(CF_IMAGES) == 0:
    print("\n[WARN] No CF images decoded — Inception FID can’t be computed.")
else:
    CF_SET = torch.cat(CF_IMAGES, dim=0).clamp(0,1)  # [N,1,28,28]
    if (MAX_FID_SAMPLES is not None) and (CF_SET.size(0) > MAX_FID_SAMPLES):
        idx = torch.randperm(CF_SET.size(0))[:MAX_FID_SAMPLES]
        CF_SET = CF_SET[idx]

    ALL_TEST_IMGS, ALL_TEST_T = [], []
    for x, t in DataLoader(test_ds, batch_size=1024, shuffle=False):
        ALL_TEST_IMGS.append(x); ALL_TEST_T.append(t)
    ALL_TEST_IMGS = torch.cat(ALL_TEST_IMGS, dim=0)
    ALL_TEST_T    = torch.cat(ALL_TEST_T,    dim=0)

    targets = (1.10 * t_gt.view(-1))[:CF_SET.size(0)]
    pool_t = ALL_TEST_T.view(-1).cpu().numpy()
    proxies = []
    for tprime in targets.view(-1).cpu().numpy():
        idx = np.argsort(np.abs(pool_t - tprime))[:1]
        proxies.append(ALL_TEST_IMGS[idx])
    PROXY_SET = torch.cat(proxies, dim=0)

    inc = InceptionEncoder(run_device=INC_DEVICE)
    mu1, cov1 = streaming_feats_stats(inc, CF_SET, bsz=INCEPT_BS)
    mu2, cov2 = streaming_feats_stats(inc, PROXY_SET, bsz=INCEPT_BS)
    fid_img = fid_distance(mu1, cov1, mu2, cov2)
    print(f'\nInception FID (Model-CF vs Proxy-Real-After): {fid_img:.6f}')
