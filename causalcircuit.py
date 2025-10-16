# causal_circuit_all_interventions.py
# CausalFlow-style training and evaluation on a 4-node causal circuit
# Interventions: arm (label=1, node=0), blue (label=2, node=1),
#                green (label=3, node=2), red (label=4, node=3)
# Metrics per intervention: MAE, FFD (Gaussian Fréchet feature distance),
#                           DAG-aware LOC, CVS, FCS, with CSV logging.  # [web:82][web:27][web:26]

import os, ast, csv, glob, random
import numpy as np, pandas as pd
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# DAG: arm → blue → green → red with extra direct edges arm→green, arm→red, blue→red
# -----------------------------
# Nodes: 0=arm, 1=blue, 2=green, 3=red; acyclic directed graph encodes causal relations used by LOC.  # [web:82][web:90]
DAG_STRUCTURE = {
    0: [1, 2, 3],  # arm affects blue, green, red directly  # [web:82]
    1: [2, 3],     # blue affects green, red  # [web:82]
    2: [3],        # green affects red       # [web:82]
    3: []          # red has no children     # [web:82]
}

def get_descendants(node: int) -> List[int]:
    """All unique nodes reachable by directed paths from node (children, grandchildren, ...)."""  # [web:82][web:90]
    descendants, to_visit, visited = [], [node], set()
    while to_visit:
        cur = to_visit.pop()
        if cur in visited:
            continue
        visited.add(cur)
        kids = DAG_STRUCTURE.get(cur, [])
        descendants.extend(kids)
        to_visit.extend(kids)
    return list(set(descendants))  # [web:82]

def get_ancestors(node: int) -> List[int]:
    """All nodes that can reach 'node' via directed paths (parents, grandparents, ...)."""  # [web:82][web:90]
    anc = []
    for src in DAG_STRUCTURE.keys():
        if node in get_descendants(src):
            anc.append(src)
    return anc  # [web:82]

# -----------------------------
# Config
# -----------------------------
class Config:
    root = '/workspace/base/datasetCC'      # dataset root with train/validate/test subfolders  # [web:82]
    epochs = 10                             # training epochs  # [web:27]
    batch_size = 128                        # dataloader batch size  # [web:27]
    lr = 1e-3                               # Adam learning rate  # [web:27]
    hidden_dim = 512                        # MLP width  # [web:27]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # compute device  # [web:27]
    seed = 42                               # RNG seed  # [web:27]
    log_dir = './logs_causalflow_all'       # where logs and checkpoints are stored  # [web:27]

config = Config()  # [web:27]

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    """Set seeds for Python, NumPy, and PyTorch for reproducibility (best‑effort)."""  # [web:27]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # [web:27]

set_seed(config.seed)  # [web:27]

# -----------------------------
# Dataset utilities
# -----------------------------
def find_metadata_csv(split_dir: str, split_name: str) -> str:
    """Find a metadata CSV for split by preferred names or fallback to any CSV in directory."""  # [web:82]
    pref = os.path.join(split_dir, f"{split_name}_metadata.csv")
    if os.path.isfile(pref):
        return pref  # [web:82]
    for name in ["metadata.csv", "cc_metadata.csv", "data.csv"]:
        p = os.path.join(split_dir, name)
        if os.path.isfile(p):
            return p  # [web:82]
    cands = [f for f in glob.glob(os.path.join(split_dir, "*.csv")) if "img" not in f.lower()]
    if not cands:
        raise FileNotFoundError(f"No metadata CSV found in {split_dir}")  # [web:82]
    return cands[0]  # [web:82]

def safe_literal_eval(s):
    """Safely parse Python literal from string; replaces nan/inf tokens to parseable floats when needed."""  # [web:82]
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        s = str(s).replace('nan', 'float("nan")').replace('inf', 'float("inf")')
        return ast.literal_eval(s)  # [web:82]

# -----------------------------
# Dataset
# -----------------------------
class CausalCircuitDataset(Dataset):
    """Loads z_before/z_after pairs and intervention labels for the causal circuit."""  # [web:82]
    INTERVENTION_NAMES = {0: "none", 1: "arm", 2: "blue", 3: "green", 4: "red"}  # [web:82]
    def __init__(self, root_dir: str, split: str):
        assert split in ["train", "test", "validate"]  # [web:82]
        self.root_dir, self.split = root_dir, split
        self.split_dir = os.path.join(root_dir, split)
        self.csv_path = find_metadata_csv(self.split_dir, split)
        print(f"Loading {split} data from: {self.csv_path}")  # [web:82]
        self.df = pd.read_csv(self.csv_path).reset_index(drop=True)
        print(f"Loaded {len(self.df)} samples for {split}")  # [web:82]
        self._validate_dataset()
        if 'intervention_labels' in self.df.columns:
            counts = self.df['intervention_labels'].value_counts().sort_index()
            print(f"Intervention distribution: {dict(counts)}")  # [web:82]

    def _validate_dataset(self):
        """Check required columns and latent vector shapes [4]."""  # [web:82]
        req = ['original_latents', 'intervention_labels', 'intervention_masks']
        miss = [c for c in req if c not in self.df.columns]
        if miss:
            raise ValueError(f"Missing required columns: {miss}")  # [web:82]
        sample_latents = safe_literal_eval(self.df.iloc[0]['original_latents'])
        if not (isinstance(sample_latents, list) and len(sample_latents) == 2):
            raise ValueError("original_latents should be [[z_before],[z_after]]")  # [web:82]
        if len(sample_latents[0]) != 4 or len(sample_latents[1]) != 4:
            raise ValueError("Latent vectors should have 4 dimensions")  # [web:82]
        print("✓ Dataset validation passed")  # [web:82]

    def __len__(self): return len(self.df)  # [web:82]

    def __getitem__(self, idx):
        """Return a dict with z_before, z_after, intervention label/onehot, and mask."""  # [web:82]
        row = self.df.iloc[idx]
        orig = safe_literal_eval(row["original_latents"])
        z_before = torch.tensor(orig[0], dtype=torch.float32)
        z_after  = torch.tensor(orig[1], dtype=torch.float32)
        intv_label = int(row["intervention_labels"])
        intv_mask = torch.tensor([float(b) for b in safe_literal_eval(row["intervention_masks"])], dtype=torch.float32)
        onehot = torch.zeros(5, dtype=torch.float32); onehot[intv_label] = 1.0
        return {"z_before": z_before, "z_after": z_after,
                "intv_label": intv_label, "intv_onehot": onehot, "intv_mask": intv_mask}  # [web:82]

# -----------------------------
# Model
# -----------------------------
class CausalFlowModel(nn.Module):
    """Residual MLP: z_after ≈ z_before + α·MLP([z_before, onehot(intervention)])."""  # [web:82]
    def __init__(self, latent_dim=4, hidden_dim=512, num_interventions=5):
        super().__init__()
        self.latent_dim, self.hidden_dim, self.num_interventions = latent_dim, hidden_dim, num_interventions
        inp = latent_dim + num_interventions
        self.predictor = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self._init_weights()
        self.initial_param_norm = self._get_param_norm()

    def _init_weights(self):
        """Xavier init for Linear layers with zero bias for stable start."""  # [web:82]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _get_param_norm(self):
        """Sum of parameter norms as a coarse training sanity check."""  # [web:82]
        return sum(p.norm().item() for p in self.parameters())

    def forward(self, z_before, intv_onehot):
        """Concatenate inputs, predict Δz, and form residual output."""  # [web:82]
        x = torch.cat([z_before, intv_onehot], dim=-1)
        dz = self.predictor(x)
        return z_before + self.residual_weight * dz  # [web:82]

# -----------------------------
# Metrics: MAE, FFD, DAG-aware LOC, CVS, FCS
# -----------------------------
@torch.no_grad()
def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error across dims/samples (lower is better)."""  # [web:27]
    return torch.mean(torch.abs(pred - target)).item()  # [web:27]

@torch.no_grad()
def _trace_sqrt_product(c1: torch.Tensor, c2: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute tr(sqrt(c1 c2)) via eigenvalues of A = sqrt(c1) c2 sqrt(c1) for stability."""  # [web:26][web:34]
    d = c1.size(0)
    I = torch.eye(d, dtype=c1.dtype, device=c1.device)
    e1, U1 = torch.linalg.eigh(c1 + eps * I)
    e1 = torch.clamp(e1, min=0.0)
    sqrtC1 = (U1 * torch.sqrt(e1).unsqueeze(0)) @ U1.T
    A = sqrtC1 @ c2 @ sqrtC1
    A = 0.5 * (A + A.T)
    ea, _ = torch.linalg.eigh(A)
    ea = torch.clamp(ea, min=0.0)
    return float(torch.sum(torch.sqrt(ea)).item())  # [web:26][web:34]

@torch.no_grad()
def compute_ffd(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """Fréchet Feature Distance between Gaussian embeddings (lower is better)."""  # [web:26][web:34]
    if pred.shape[0] < 2 or target.shape[0] < 2:
        return float('nan')  # [web:26]
    mu_p, mu_t = pred.mean(dim=0), target.mean(dim=0)
    P, T = pred - mu_p, target - mu_t
    cP = (P.T @ P) / (pred.shape[0] - 1)
    cT = (T.T @ T) / (target.shape[0] - 1)
    dmu2 = torch.sum((mu_p - mu_t) ** 2)
    tr_sum = torch.trace(cP) + torch.trace(cT)
    ts = _trace_sqrt_product(cP + eps * torch.eye(cP.size(0), device=cP.device),
                             cT + eps * torch.eye(cT.size(0), device=cT.device), eps=eps)
    return float((dmu2 + tr_sum - 2.0 * ts))  # [web:26][web:34]

@torch.no_grad()
def compute_loc_dag_corrected(pred: torch.Tensor, target: torch.Tensor, z_before: torch.Tensor, intv_node=3) -> float:
    """DAG-aware locality: penalize unintended change in ancestors of the intervened node (lower is better)."""  # [web:82]
    anc = get_ancestors(intv_node)
    if not anc:
        return 0.0  # [web:82]
    pred_change = torch.abs(pred[:, anc] - z_before[:, anc])
    targ_change = torch.abs(target[:, anc] - z_before[:, anc])
    excess = torch.clamp(pred_change - targ_change, min=0.0)
    return float(excess.mean().item())  # [web:82]

@torch.no_grad()
def compute_cvs(pred: torch.Tensor, target: torch.Tensor, z_before: torch.Tensor, intv_dim=3) -> float:
    """Validity proxy: correlation between intended and predicted changes at intervened dim (higher is better)."""  # [web:27]
    p = pred[:, intv_dim] - z_before[:, intv_dim]
    t = target[:, intv_dim] - z_before[:, intv_dim]
    p = p - p.mean(); t = t - t.mean()
    num = (p * t).sum()
    den = torch.sqrt((p ** 2).sum() * (t ** 2).sum() + 1e-8)
    return float((num / den).item())  # [web:27]

@torch.no_grad()
def compute_fcs_fixed(pred: torch.Tensor, target: torch.Tensor, z_before: torch.Tensor) -> float:
    """Feature Causal Score: mean absolute normalized |Δz_pred − Δz_true| (lower is better)."""  # [web:27]
    dp = pred - z_before
    dt = target - z_before
    mag = torch.abs(dt).mean(dim=0)
    scale = torch.clamp(mag, min=0.01)
    if mag.mean() < 1e-3:
        scale = torch.clamp(dt.std(dim=0), min=0.01)
    return float((torch.abs(dp - dt) / scale).mean().item())  # [web:27]

# -----------------------------
# Training and eval helpers
# -----------------------------
def train_epoch(model, dataloader, optimizer, device):
    """One pass of MSE training on z_after with Adam optimizer."""  # [web:27]
    model.train(); total, n, b = 0.0, 0, 0
    for batch in dataloader:
        z_before = batch["z_before"].to(device)
        z_after  = batch["z_after"].to(device)
        onehot   = batch["intv_onehot"].to(device)
        optimizer.zero_grad()
        pred = model(z_before, onehot)
        loss = F.mse_loss(pred, z_after)
        loss.backward(); optimizer.step()
        bs = z_before.size(0)
        total += loss.item() * bs; n += bs; b += 1
    return total / max(n, 1), b  # [web:27]

@torch.no_grad()
def evaluate_intervention(model, dataloader, device, target_intervention: int, intv_dim: int):
    """Compute MAE, FFD, LOC, CVS, FCS on samples with the given intervention label and node index."""  # [web:27][web:82]
    model.eval()
    preds, targs, befs = [], [], []
    for batch in dataloader:
        mask = (batch["intv_label"] == target_intervention)
        if mask.sum() == 0:
            continue
        z_before = batch["z_before"][mask].to(device)
        z_after  = batch["z_after"][mask].to(device)
        onehot   = batch["intv_onehot"][mask].to(device)
        pred = model(z_before, onehot)
        preds.append(pred.cpu()); targs.append(z_after.cpu()); befs.append(z_before.cpu())
    if not preds:
        return {"MAE": float('nan'), "FFD": float('nan'), "LOC": float('nan'),
                "CVS": float('nan'), "FCS": float('nan'), "num_samples": 0}  # [web:27]
    P = torch.cat(preds, dim=0)
    T = torch.cat(targs, dim=0)
    B = torch.cat(befs,  dim=0)
    return {
        "MAE": compute_mae(P, T),
        "FFD": compute_ffd(P, T),
        "LOC": compute_loc_dag_corrected(P, T, B, intv_node=intv_dim),
        "CVS": compute_cvs(P, T, B, intv_dim=intv_dim),
        "FCS": compute_fcs_fixed(P, T, B),
        "num_samples": P.shape[0],
    }  # [web:27][web:26][web:82]

# Unified intervention spec and evaluators
INTERVENTION_SPECS = {
    "arm":   {"label": 1, "node": 0},
    "blue":  {"label": 2, "node": 1},
    "green": {"label": 3, "node": 2},
    "red":   {"label": 4, "node": 3},
}  # [web:82]

@torch.no_grad()
def evaluate_all_interventions(model, dataloader, device):
    """Run evaluation for arm, blue, green, and red; print and return a dict of metrics per name."""  # [web:82]
    out = {}
    for name, spec in INTERVENTION_SPECS.items():
        m = evaluate_intervention(model, dataloader, device, spec["label"], spec["node"])
        print(f"\n{name.upper()} (label={spec['label']}, node={spec['node']}):")  # [web:82]
        print(f"  MAE: {m['MAE']:.4f}  | FFD: {m['FFD']:.4f}  | LOC: {m['LOC']:.4f}  | CVS: {m['CVS']:.4f}  | FCS: {m['FCS']:.4f}  | N={m['num_samples']}")  # [web:27][web:26]
        out[name] = m
    return out  # [web:82]

def log_all_interventions_csv(csv_path, split_tag, results_by_name):
    """Append per‑intervention rows to a CSV: split,name,MAE,FFD,LOC,CVS,FCS,num_samples."""  # [web:27]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    flds = ["split","name","MAE","FFD","LOC","CVS","FCS","num_samples"]
    exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=flds)
        if not exists: w.writeheader()
        for name, m in results_by_name.items():
            w.writerow({"split": split_tag, "name": name, "MAE": m["MAE"], "FFD": m["FFD"],
                        "LOC": m["LOC"], "CVS": m["CVS"], "FCS": m["FCS"], "num_samples": m["num_samples"]})  # [web:27]

# -----------------------------
# Main execution
# -----------------------------
print(f"Using device: {config.device}")  # [web:27]
print("Loading datasets...")  # [web:82]
train_ds = CausalCircuitDataset(config.root, 'train')
val_ds   = CausalCircuitDataset(config.root, 'validate')
test_ds  = CausalCircuitDataset(config.root, 'test')
print(f"\nDataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")  # [web:82]

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=2)

model = CausalFlowModel(latent_dim=4, hidden_dim=config.hidden_dim, num_interventions=5).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")  # [web:27]

os.makedirs(config.log_dir, exist_ok=True)
log_file = os.path.join(config.log_dir, 'training_log_all.csv')
best_model_path = os.path.join(config.log_dir, 'best_model_all.pt')

best_val_mae = float('inf')
print("\n" + "="*60); print("STARTING TRAINING - ALL INTERVENTIONS (DAG-AWARE LOC)"); print("="*60)  # [web:82]
for epoch in range(1, config.epochs + 1):
    tr_loss, n_batches = train_epoch(model, train_loader, optimizer, config.device)
    cur_norm = model._get_param_norm()
    delta_norm = abs(cur_norm - model.initial_param_norm)
    # Validate on RED by default for early stopping (can choose another policy)
    val_red = evaluate_intervention(model, val_loader, config.device, target_intervention=4, intv_dim=3)
    if epoch % 10 == 0 or epoch == 1:
        print(f"\nEpoch {epoch}/{config.epochs} | Train Loss: {tr_loss:.4f} | Δ‖θ‖: {delta_norm:.4f}")  # [web:27]
        print(f"Val (RED): MAE={val_red['MAE']:.4f} FFD={val_red['FFD']:.4f} LOC={val_red['LOC']:.4f} CVS={val_red['CVS']:.4f} FCS={val_red['FCS']:.4f} N={val_red['num_samples']}")  # [web:27][web:26]
    if val_red['MAE'] < best_val_mae:
        best_val_mae = val_red['MAE']
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}, best_model_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ✓ New best checkpoint (val RED MAE={best_val_mae:.4f}) saved.")  # [web:27]
    # Log epoch with only red val for brevity
    exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        fields = ['epoch','train_loss','val_red_MAE','val_red_FFD','val_red_LOC','val_red_CVS','val_red_FCS','val_red_N']
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow({'epoch': epoch, 'train_loss': tr_loss,
                    'val_red_MAE': val_red['MAE'], 'val_red_FFD': val_red['FFD'], 'val_red_LOC': val_red['LOC'],
                    'val_red_CVS': val_red['CVS'], 'val_red_FCS': val_red['FCS'], 'val_red_N': val_red['num_samples']})  # [web:27]

print("\n" + "="*60); print("FINAL EVALUATION ON TEST SET - ALL INTERVENTIONS"); print("="*60)  # [web:27]
ckpt = torch.load(best_model_path, map_location=config.device)
model.load_state_dict(ckpt['model_state_dict'])
test_results = evaluate_all_interventions(model, test_loader, config.device)
val_results  = evaluate_all_interventions(model, val_loader,  config.device)
log_all_interventions_csv(os.path.join(config.log_dir, 'intervention_metrics_test.csv'), "test", test_results)
log_all_interventions_csv(os.path.join(config.log_dir, 'intervention_metrics_val.csv'),  "val",  val_results)
print("\nPer‑intervention metrics written to logs_causalflow_all/intervention_metrics_{test,val}.csv")  # [web:27]

# Informal range checks (can be customized per dataset scaling)
print("\nRange checks (informational):")  # [web:27]
for name, m in test_results.items():
    print(f"  {name:>5}: MAE OK? {0 <= m['MAE'] <= 10} | FFD OK? {0 <= m['FFD'] <= 100} | LOC OK? {0 <= m['LOC'] <= 1} | CVS OK? {-1 <= m['CVS'] <= 1} | FCS OK? {0 <= m['FCS'] <= 10}")  # [web:27][web:26]

# DAG inspection summary
print("\n" + "="*60); print("DAG STRUCTURE SUMMARY"); print("="*60)  # [web:82]
for node, kids in DAG_STRUCTURE.items():
    print(f"Node {node} -> children {kids}")  # [web:82]
print(f"Ancestors(red=3): {get_ancestors(3)} | Descendants(arm=0): {get_descendants(0)}")  # [web:82]
