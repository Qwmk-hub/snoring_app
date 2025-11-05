import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

OUTDIR = "./dl_outputs"
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="mfcc_features.csv")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    # 체크포인트/스케일러/분할 인덱스 로드
    ckpt = load_ckpt()
    model = build_model(ckpt)
    scaler = joblib.load(os.path.join(OUTDIR, "scaler.joblib"))
    split = json.load(open(os.path.join(OUTDIR, "split_indices.json"), encoding="utf-8"))

    # 메타
    n_classes = int(ckpt["n_classes"])
    feature_cols = ckpt["feature_cols"]
    label_map = ckpt["label_map"]              # {0:'non_snore', 1:'snore'} 같은 형태
    name_of = lambda i: label_map[str(i)] if isinstance(label_map, dict) and str(i) in label_map else label_map[i]

    # 데이터 로드 & 정렬
    df = pd.read_csv(args.csv_path)
    if "class" not in df.columns:
        raise ValueError("test 평가에는 입력 CSV에 'class' 컬럼이 필요합니다.")
    X_all = align_features(df, feature_cols)
    X_all = scaler.transform(X_all)

    idx_te = split["test"]
    X_te = X_all[idx_te]
    y_true_raw = df.loc[idx_te, "class"].astype(str).str.strip().values

    # 라벨 이름->인덱스 역매핑 생성
    inv_label = {v: int(k) for k, v in (label_map.items() if isinstance(label_map, dict) else enumerate(label_map))}
    # 미등록 라벨 체크
    unknown = [lbl for lbl in y_true_raw if lbl not in inv_label]
    if unknown:
        raise ValueError(f"라벨맵에 없는 클래스 발견: {sorted(set(unknown))}")
    y_true = np.array([inv_label[lbl] for lbl in y_true_raw], dtype=int)

    # 추론
    probs, y_pred = infer(model, X_te, n_classes, batch_size=args.batch_size)

    # 메트릭
    metrics = {
        "F1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "BalAcc": float(balanced_accuracy_score(y_true, y_pred)),
        "ROC_AUC": None,
    }
    if n_classes == 2:
        try:
            metrics["ROC_AUC"] = float(roc_auc_score(y_true, probs))
        except Exception:
            metrics["ROC_AUC"] = None

    # 리포트 콘솔 출력
    print("=== TEST METRICS (best_model.pt) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if v is not None else f"{k}: None")
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=[name_of(i) for i in range(n_classes)], digits=4))
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_true, y_pred))

    # CSV 저장: 정답/예측/정오
    fn_col = "filename" if "filename" in df.columns else None
    out_rows = {
        "index": idx_te,
        **({"filename": df.loc[idx_te, fn_col].values} if fn_col else {}),
        "true": [name_of(i) for i in y_true],
        "pred": [name_of(i) for i in y_pred],
        "correct": (y_true == y_pred).astype(int),
    }
    if n_classes == 2:
        # 보통 label 1이 'snore'일 가능성이 높음 → 확률 컬럼 이름 예시
        out_rows["prob_of_class_1"] = probs
        # 사람이 보기 좋게 한글 열 추가(예: 코골이/아님)
        # label 1 이름이 'snore'라면 '코골이', 0이면 '코골이 아님'으로 해석 가능
        name1 = name_of(1)
        

    save_path = os.path.join(OUTDIR, "test_eval_predictions.csv")
    pd.DataFrame(out_rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] {save_path}")
    
if __name__ == "__main__":
    main()
