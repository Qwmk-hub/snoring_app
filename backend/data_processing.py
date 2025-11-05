import os
import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

# MLP 모델 클래스 (한 번만 정의)
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(256,128,64,32), dropout=0.2,
                 norm='batch', residual_every=0):
        super().__init__()
        self.n_classes = n_classes
        self.residual_every = int(residual_every) if residual_every else 0

        def make_norm(d):
            if norm == 'batch': return nn.BatchNorm1d(d)
            if norm == 'layer': return nn.LayerNorm(d)
            return nn.Identity()

        if isinstance(dropout, (list, tuple)):
            drop_list = list(dropout)
        else:
            drop_list = [float(dropout)]
        while len(drop_list) < len(hidden):
            drop_list.append(drop_list[-1])

        layers = []
        prev = in_dim
        self.proj_for_res = nn.ModuleDict()

        for i, h in enumerate(hidden, start=1):
            block = [nn.Linear(prev, h), make_norm(h), nn.GELU(), nn.Dropout(drop_list[i-1])]
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
                key = f"{i}_proj"
                z = h + self.proj_for_res[key](z)
            else:
                z = h
        return self.head(z)

# 모델을 전역으로 로드 (한 번만)
_model = None
_checkpoint = None

def load_model():
    global _model, _checkpoint
    if _model is None:
        model_path = "backend/model/best_model.pt"
        _checkpoint = torch.load(model_path, map_location='cpu')
        
        _model = MLP(
            in_dim=_checkpoint["in_dim"],
            n_classes=_checkpoint["n_classes"],
            hidden=tuple(_checkpoint["hidden"]),
            dropout=_checkpoint["dropout"],
            norm=_checkpoint.get("norm", "batch"),
            residual_every=_checkpoint.get("residual_every", 0)
        )
        
        _model.load_state_dict(_checkpoint["model_state"])
        _model.eval()
    
    return _model, _checkpoint

def split_audio_files(input, chunk_duration=1, target_sr=16000):
    """
    오디오 파일을 1초씩 분할하여 리스트로 반환합니다.

    :param input: 오디오 파일 경로
    :param chunk_duration: 잘라낼 시간 (초 단위)
    :param target_sr: 목표 샘플링 레이트
    :return: 1초씩 분할된 오디오 데이터 리스트
    """
    chunks = []
    y, sr = librosa.load(input, sr=target_sr)

    chunk_length = chunk_duration * sr

    num_chunks = int(np.ceil(len(y) / chunk_length))

    for i in range(num_chunks):
        start_sample = i * chunk_length
        end_sample = start_sample + chunk_length
        y_chunk = y[start_sample:end_sample]

        if len(y_chunk) < chunk_length:
            y_chunk = librosa.util.fix_length(y_chunk, size=chunk_length)

        chunks.append(y_chunk)

    return chunks

def MFCC(stack):
    """
    1초 오디오 청크들로부터 MFCC 특징을 추출하여 데이터프레임으로 반환합니다.
    
    :param stack: 1초씩 분할된 오디오 데이터 리스트 (numpy 배열들)
    :return: MFCC 특징이 담긴 pandas DataFrame
    """
    results = []
    
    for i, chunk in enumerate(stack):
        mfcc = librosa.feature.mfcc(y=chunk, sr=16000, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        feat = np.concatenate([mfcc, delta, delta2], axis=0)
        feat_mean = np.mean(feat, axis=1)
        
        results.append([f"chunk_{i}"] + feat_mean.tolist())
    
    columns = ["chunk_name"] + [f"feature_{j}" for j in range(39)]
    df = pd.DataFrame(results, columns=columns)
    
    return df

def GRID(dataframe):
    """
    MFCC 특징 데이터프레임을 받아서 각 행별로 모델 예측하고 1의 개수를 반환합니다.
    
    :param dataframe: MFCC 특징이 담긴 DataFrame (chunk_name, feature_0~38)
    :return: 코골이로 예측된 청크의 개수 (int)
    """
    model, checkpoint = load_model()
    
    snore_count = 0
    for _, row in dataframe.iterrows():
        features = row[[col for col in dataframe.columns if col.startswith('feature_')]].values
        features_tensor = torch.tensor(features.astype(np.float32), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(features_tensor)
            prob = torch.sigmoid(logits.view(-1))
            prediction = 1 if prob >= 0.5 else 0
        
        if prediction == 1:
            snore_count += 1
    
    return snore_count