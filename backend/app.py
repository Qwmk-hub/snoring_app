from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from data_processing import split_audio_files, MFCC
from real_predict import result
from main import processing
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

DATA_FILE = 'database/data.txt'

def read_count():
    """data.txt에서 현재 카운트 읽기"""
    try:
        with open(DATA_FILE, 'r') as f:
            return int(f.read().strip())
    except:
        return 0

def update_count(new_count):
    """data.txt에 새로운 카운트 저장"""
    try:
        with open(DATA_FILE, 'w') as f:
            f.write(str(new_count))
        return True
    except:
        return False

def convert_webm_to_wav(input_path, output_path):
    """WebM 파일을 WAV로 변환"""
    try:
        print(f"🔄 오디오 형식 변환 시작: {input_path} → {output_path}")
        
        # WebM 파일 로드
        audio = AudioSegment.from_file(input_path, format="webm")
        
        # WAV로 변환하여 저장 (16kHz, mono)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        
        print(f"✅ 오디오 변환 완료: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 오디오 변환 실패: {str(e)}")
        return False

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """5초 오디오 파일을 처리하고 결과를 data.txt에 추가"""
    try:
        # 업로드된 파일 받기
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # processing 함수로 처리
            snore_count = processing(temp_path)
            
            # 현재 카운트 읽고 더하기
            current_count = read_count()
            new_count = current_count + snore_count
            
            # 새로운 카운트 저장
            if update_count(new_count):
                return jsonify({
                    'success': True,
                    'snore_count': snore_count,
                    'total_count': new_count,
                    'message': f'Processed successfully. Added {snore_count} to total.'
                })
            else:
                return jsonify({'error': 'Failed to update count'}), 500
                
        finally:
            # 임시 파일 삭제
            os.unlink(temp_path)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-count', methods=['GET'])
def get_count():
    """현재 총 카운트 가져오기"""
    try:
        count = read_count()
        return jsonify({'total_count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset-count', methods=['POST'])
def reset_count():
    """카운트 초기화"""
    try:
        if update_count(0):
            return jsonify({'success': True, 'total_count': 0})
        else:
            return jsonify({'error': 'Failed to reset count'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-session-audio', methods=['POST'])
def save_session_audio():
    """전체 세션 오디오를 database/data 폴더에 저장"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # database/data 폴더 생성 (없으면)
        save_directory = 'database/data'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 임시 파일명 생성 (WebM)
        from datetime import datetime
        now = datetime.now()
        temp_webm_filename = f"temp_session_{now.strftime('%Y%m%d_%H%M%S')}.webm"
        final_wav_filename = f"session_{now.strftime('%Y%m%d_%H%M%S')}.wav"
        
        # 임시 WebM 파일 저장
        temp_webm_path = os.path.join(save_directory, temp_webm_filename)
        audio_file.save(temp_webm_path)
        print(f"📁 임시 WebM 파일 저장: {temp_webm_path}")
        
        # WAV 파일 경로
        final_wav_path = os.path.join(save_directory, final_wav_filename)
        
        # WebM을 WAV로 변환
        conversion_success = convert_webm_to_wav(temp_webm_path, final_wav_path)
        
        if not conversion_success:
            # 변환 실패 시 임시 파일 정리하고 오류 반환
            if os.path.exists(temp_webm_path):
                os.remove(temp_webm_path)
            return jsonify({'error': 'Audio format conversion failed'}), 500
        
        # 임시 WebM 파일 삭제
        if os.path.exists(temp_webm_path):
            os.remove(temp_webm_path)
            print(f"🗑️ 임시 WebM 파일 삭제: {temp_webm_path}")
        
        print(f"✅ 세션 오디오 저장 완료: {final_wav_path}")
        
        # 저장된 파일을 main.py의 processing 함수로 처리
        try:
            session_snore_count = processing(final_wav_path)
            print(f"🔍 세션 코골이 감지 결과: {session_snore_count}")
            
            # 현재 총 카운트에 추가
            current_count = read_count()
            new_total_count = current_count + session_snore_count
            update_success = update_count(new_total_count)
            
            if update_success:
                print(f"📊 data.txt 업데이트 완료: {current_count} + {session_snore_count} = {new_total_count}")
                
                return jsonify({
                    'message': 'Session audio processed and saved successfully',
                    'filename': final_wav_filename,
                    'path': final_wav_path,
                    'session_snore_count': session_snore_count,
                    'total_count': new_total_count
                }), 200
            else:
                return jsonify({
                    'error': 'Failed to update count file',
                    'filename': final_wav_filename,
                    'path': final_wav_path,
                    'session_snore_count': session_snore_count
                }), 500
                
        except Exception as processing_error:
            print(f"❌ 오디오 처리 실패: {str(processing_error)}")
            return jsonify({
                'message': 'Session audio saved but processing failed',
                'filename': final_wav_filename,
                'path': final_wav_path,
                'error': str(processing_error)
            }), 200
        
    except Exception as e:
        print(f"❌ 세션 오디오 저장 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

def processing(file):
    """오디오 파일 처리 함수"""
    stack = split_audio_files(file, chunk_duration=1, target_sr=16000)
    mfcc_features = MFCC(stack)
    snore_count = result(mfcc_features)
    return snore_count

if __name__ == '__main__': app.run(debug=True, port=5001, host='0.0.0.0')