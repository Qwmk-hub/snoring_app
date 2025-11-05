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

def main():
    ap = argparse.ArgumentParser(description="추론 전용 스크립트 (라벨 없는 새 데이터 예측)")
    ap.add_argument("--csv_path", type=str, required=True, help="예측할 CSV 파일 경로")
    ap.add_argument("--output", type=str, default=None, help="결과 저장 경로 (기본: dl_outputs/predictions.csv)")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    print("="*60)
    print("🔮 추론 모드 시작")
    print("="*60)
    
    # 체크포인트/스케일러 로드
    print("\n📦 모델 로딩 중...")
    ckpt = load_ckpt()
    model = build_model(ckpt)
    scaler = joblib.load(os.path.join(OUTDIR, "scaler.joblib"))
    
    # 메타
    n_classes = int(ckpt["n_classes"])
    feature_cols = ckpt["feature_cols"]
    label_map = ckpt["label_map"]
    name_of = lambda i: label_map[str(i)] if isinstance(label_map, dict) and str(i) in label_map else label_map[i]
    
    print(f"   클래스 이름: {[name_of(i) for i in range(n_classes)]}")
    
    # 데이터 로드
    print(f"\n📂 데이터 로딩: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"   총 {len(df)}개 샘플")
    
    # 특징 정렬 & 스케일링
    X = align_features(df, feature_cols)
    X = scaler.transform(X)
    
    # 추론
    print(f"\n🚀 추론 중...")
    probs, y_pred = infer(model, X, n_classes, batch_size=args.batch_size)
    
    print(f"✅ 추론 완료!")
    
    # 예측 분포 출력
    unique, counts = np.unique(y_pred, return_counts=True)
    print(unique[0])
    print(counts[0])
    print(f"\n📊 예측 결과 분포:")
    for cls, cnt in zip(unique, counts):
        print(f"   {name_of(cls)}: {cnt}개 ({cnt/len(y_pred)*100:.1f}%)")
    
    # CSV 저장
    chunk_col = "chunk_name" if "chunk_name" in df.columns else None
    fn_col = "filename" if "filename" in df.columns else None
    
    out_rows = {
        "index": list(range(len(y_pred))),
        **({"chunk_name": df[chunk_col].values} if chunk_col else {}),
        **({"filename": df[fn_col].values} if fn_col else {}),
        "predicted_class": [name_of(i) for i in y_pred],
    }
    
    if n_classes == 2:
        out_rows["probability"] = probs
    else:
        # 다중 클래스는 각 클래스별 확률 저장
        for i in range(n_classes):
            out_rows[f"prob_{name_of(i)}"] = probs[:, i]
    
    save_path = args.output if args.output else os.path.join(OUTDIR, "predictions.csv")
    pd.DataFrame(out_rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    
    print(f"\n💾 결과 저장: {save_path}")
    print("="*60)
    
if __name__ == "__main__": main()

# python backend/predict.py --csv_path /Users/parksung-cheol/Desktop/snoring/backend/mfcc_features.csv