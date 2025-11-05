import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

def split_audio_files(input_folder, output_folder, chunk_duration=1, target_sr=16000):
    """
    지정된 폴더의 오디오 파일들을 일정한 길이의 조각으로 자릅니다.

    :param input_folder: 원본 오디오 파일이 있는 폴더 경로
    :param output_folder: 잘린 오디오 파일을 저장할 폴더 경로
    :param chunk_duration: 잘라낼 시간 (초 단위)
    :param target_sr: 목표 샘플링 레이트
    """
    # 결과 저장 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)
    print(f"결과가 '{output_folder}' 폴더에 저장됩니다.")

    # 입력 폴더 내의 wav 파일 목록 가져오기
    files_to_process = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    
    if not files_to_process:
        print(f"'{input_folder}'에 처리할 .wav 파일이 없습니다.")
        return

    # 각 파일 처리
    for filename in tqdm(files_to_process, desc="오디오 파일 분할 중"):
        input_path = os.path.join(input_folder, filename)
        
        try:
            # 오디오 로드 및 샘플링 레이트 통일
            y, sr = librosa.load(input_path, sr=target_sr)

            # 조각의 길이 계산 (샘플 수 기준)
            chunk_length = chunk_duration * sr

            # 전체 오디오를 몇 개의 조각으로 나눌지 계산
            num_chunks = int(np.ceil(len(y) / chunk_length))

            # 각 조각을 파일로 저장
            for i in range(num_chunks):
                start_sample = i * chunk_length
                end_sample = start_sample + chunk_length
                y_chunk = y[start_sample:end_sample]

                # 마지막 조각이 1초보다 짧을 경우, 길이를 맞춰주기 (padding)
                if len(y_chunk) < chunk_length:
                    y_chunk = librosa.util.fix_length(y_chunk, size=chunk_length)

                # 새로운 파일 이름 생성 (예: original.wav -> original_chunk_0.wav)
                base_filename = os.path.splitext(filename)[0]
                output_filename = f"{base_filename}_chunk_{i}.wav"
                output_path = os.path.join(output_folder, output_filename)

                # 조각난 오디오 파일 저장
                sf.write(output_path, y_chunk, sr)

        except Exception as e:
            print(f"'{filename}' 처리 중 오류 발생: {e}")

    print("✅ 모든 파일 처리가 완료되었습니다.")

# --- 설정 (이 부분만 수정하세요) ---
# 5초짜리 오디오 파일들이 있는 폴더 (사용자가 지정)
INPUT_DATA_FOLDER = "your_5sec_audio_folder" 

# 1초로 잘린 파일들을 저장할 새로운 폴더
OUTPUT_DATA_FOLDER = "output_1sec_chunks" 

# --- 코드 실행 ---
split_audio_files(INPUT_DATA_FOLDER, OUTPUT_DATA_FOLDER)