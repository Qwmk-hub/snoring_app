from data_processing import split_audio_files, MFCC
from real_predict import result

def processing(file):
    stack = split_audio_files(file, chunk_duration=1, target_sr=16000)
    mfcc_features = MFCC(stack)
    snore_count, non_snore_count = result(mfcc_features)
    return snore_count

K = processing("/Users/parksung-cheol/Desktop/snoring/1-17295-A-29.wav")
print(K)