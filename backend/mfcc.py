import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 설정 ---
# 스크립트 파일의 현재 위치를 기준으로 절대 경로를 생성합니다.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 대화형 환경 (예: Jupyter Notebook)에서 실행될 경우를 대비
    script_dir = os.getcwd()

# 1. 원본 오디오 파일이 있는 폴더 경로 
INPUT_DIR = r"C:\Users\almon\OneDrive\바탕 화면\snoring\Total_dataset"

# 2. 결과 CSV 파일을 저장할 경로와 "파일 이름"까지 명확하게 지정
# 'Output' 폴더 안에 'mfcc_features_final.csv' 라는 이름으로 저장됩니다.
SAVE_CSV_PATH = os.path.join(script_dir, "Output", "mfcc_features.csv")

# --- 코드 실행 ---
results = []
print("MFCC + Delta + Delta-Delta 특징 추출을 시작합니다...")
print(f"입력 폴더: {INPUT_DIR}")
print(f"출력 파일: {SAVE_CSV_PATH}")


# 'snore', 'nonsnore' 레이블 순서로 폴더 처리
for label in ["snore", "nonsnore"]:
    folder_path = os.path.join(INPUT_DIR, label)
    
    if not os.path.isdir(folder_path):
        print(f"\n경고: '{folder_path}' 폴더를 찾을 수 없어 건너뜁니다.")
        print("폴더 구조가 올바른지 확인해주세요. (스크립트 파일 옆에 'data/processed_1s/snore' 와 'data/processed_1s/nonsnore' 폴더가 있어야 합니다.)")
        continue
        
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    
    if not files:
        print(f"경고: '{folder_path}' 폴더에 .wav 파일이 없습니다.")
        continue

    for file in tqdm(files, desc=f"Processing {label} files"):
        file_path = os.path.join(folder_path, file)

        try:
            # 오디오 로드 (16kHz, 모노로 로드)
            y, sr = librosa.load(file_path, sr=16000, mono=True)

            # MFCC (13) + Delta (13) + Delta-Delta (13) = 39 특징 추출
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            # 특징을 하나로 합치고 시간 축에 대해 평균
            feat = np.concatenate([mfcc, delta, delta2], axis=0)
            feat_mean = np.mean(feat, axis=1)

            # 결과 리스트에 추가 (파일명, 클래스, 특징 벡터)
            results.append([file, label] + feat_mean.tolist())
        except Exception as e:
            print(f"오류 발생: {file} 처리 중 문제 발생 - {e}")

# DataFrame으로 변환
if not results:
    print("\n오류: 처리된 오디오 파일이 없습니다. 입력 폴더 경로와 파일들을 다시 확인해주세요.")
else:
    columns = ["filename", "class"] + [f"feature_{i}" for i in range(39)]
    df = pd.DataFrame(results, columns=columns)

    # CSV 파일로 저장하기 전, 저장할 폴더가 있는지 확인하고 없으면 생성
    save_dir = os.path.dirname(SAVE_CSV_PATH)
    os.makedirs(save_dir, exist_ok=True)
        
    df.to_csv(SAVE_CSV_PATH, index=False)

    print("\n-------------------------------------------")
    print(f"✅ 특징 추출 완료!")
    print(f"총 {len(df)}개의 샘플이 처리되었습니다.")
    print(f"결과가 '{SAVE_CSV_PATH}'에 저장되었습니다.")
    print("-------------------------------------------")

