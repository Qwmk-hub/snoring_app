import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

OUTDIR = "backend/model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== MLP (학습 때와 동일) ======
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(256,128,64,32), dropout=0.2,
                 norm='batch', residual_every=0):
        super().__init__()
        assert len(hidden) >= 1
        self.n_classes = n_classes
        self.residual_every = int(residual_every) if residual_every else 0

        def make_norm(d):
            if norm == 'batch': return nn.BatchNorm1d(d)
            if norm == 'layer': return nn.LayerNorm(d)
            return nn.Identity()

        drops = list(dropout) if isinstance(dropout, (list, tuple)) else [float(dropout)]
        while len(drops) < len(hidden): drops.append(drops[-1])

        layers = []; prev = in_dim
        self.proj_for_res = nn.ModuleDict()
        for i, h in enumerate(hidden, start=1):
            block = [nn.Linear(prev, h), make_norm(h), nn.GELU(), nn.Dropout(drops[i-1])]
            layers.append(nn.Sequential(*block))
            if self.residual_every and (i % self.residual_every == 0):
                key = f"{i}_proj"
                self.proj_for_res[key] = nn.Linear(prev, h) if prev != h else nn.Identity()
            prev = h

        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(prev, 1 if n_classes == 2 else n_classes)

    def forward(self, x):
        z = x
        for i, blk in enumerate(self.blocks, start=1):
            h = blk(z)
            if self.residual_every and (i % self.residual_every == 0):
                z = h + self.proj_for_res[f"{i}_proj"](z)
            else:
                z = h
        return self.head(z)

# ====== 로드 & 유틸 ======
def load_ckpt():
    path = os.path.join(OUTDIR, "best_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    return torch.load(path, map_location="cpu")

def build_model(ckpt):
    model = MLP(
        in_dim=ckpt["in_dim"],
        n_classes=ckpt["n_classes"],
        hidden=tuple(ckpt["hidden"]),
        dropout=ckpt["dropout"],
        norm=ckpt.get("norm", "batch"),
        residual_every=ckpt.get("residual_every", 0),
    )
    model.load_state_dict({k: v for k, v in ckpt["model_state"].items()})
    model.to(device).eval()
    return model

def align_features(df, feature_cols):
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

@torch.no_grad()
def infer(model, X, n_classes, batch_size=512):
    probs_list, preds_list = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        logits = model(xb)
        if n_classes == 2:
            prob = torch.sigmoid(logits.view(-1)).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
        else:
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = prob.argmax(axis=1)
        probs_list.append(prob); preds_list.append(pred)
    probs = np.concatenate(probs_list) if probs_list else np.array([])
    preds = np.concatenate(preds_list) if preds_list else np.array([])
    return probs, preds

def result(mfcc_features):
    ckpt = load_ckpt()
    model = build_model(ckpt)
    scaler = joblib.load(os.path.join(OUTDIR, "scaler.joblib"))
    
    # 메타
    n_classes = int(ckpt["n_classes"])
    feature_cols = ckpt["feature_cols"]
    label_map = ckpt["label_map"]
    name_of = lambda i: label_map[str(i)] if isinstance(label_map, dict) and str(i) in label_map else label_map[i]
    
    # 특징 정렬 & 스케일링
    X = align_features(mfcc_features, feature_cols)
    X = scaler.transform(X)
    
    probs, y_pred = infer(model, X, n_classes, batch_size=512)
    unique, counts = np.unique(y_pred, return_counts=True)
    
    return unique[0], counts[0]