import React, { useState, useRef } from 'react';
import p3hImage from './assets/images/p3h.png';
import alarm from './assets/images/alarm.png';
import home_color from './assets/images/home_color.png';
import polygon from './assets/images/polygon.png';
import graph from './assets/images/graph.png';
import setting from './assets/images/settings.png';

interface HomeProps {
  onLogout: () => void;
  onGoToReport: () => void;
  onGoToGraph: () => void;
  onGoToSetting: () => void;
}

const Home: React.FC<HomeProps> = ({ onLogout, onGoToReport, onGoToGraph, onGoToSetting }) => {
  const [selectedHour, setSelectedHour] = useState(9);
  const [selectedMinute, setSelectedMinute] = useState(0);
  const [selectedPeriod, setSelectedPeriod] = useState<'AM' | 'PM'>('AM');
  const [isAlarmOn, setIsAlarmOn] = useState(true);
  const [isActivated, setIsActivated] = useState(false);

  // 녹음 관련 상태
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const sessionAudioChunksRef = useRef<Blob[]>([]); // 전체 세션 오디오 청크
  const [totalSnoreCount, setTotalSnoreCount] = useState(0);

  const hours = Array.from({ length: 12 }, (_, i) => i + 1); // 1-12
  const minutes = Array.from({ length: 60 }, (_, i) => i);   // 0-59

  const formatTime = (hour: number, minute: number) => {
    return `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
  };

  const toggleAlarm = () => {
    setIsAlarmOn(!isAlarmOn);
  };

  // 전체 세션 오디오를 database/data에 저장하는 함수

  // 전체 세션 오디오를 백엔드에 저장하는 함수
  const saveSessionAudioToBackend = async (audioBlob: Blob) => {
    try {
      const now = new Date();
      const fileName = `session_${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}${now.getSeconds().toString().padStart(2, '0')}.wav`;
      
      const formData = new FormData();
      formData.append('audio', audioBlob, fileName);

      const response = await fetch('http://localhost:5001/save-session-audio', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ 세션 음성 저장 및 처리 완료:', result);
        
        // 세션에서 감지된 코골이가 있으면 총 카운트 업데이트
        if (result.total_count !== undefined) {
          setTotalSnoreCount(result.total_count);
          console.log(`📊 총 코골이 카운트 업데이트: ${result.total_count} (세션에서 +${result.session_snore_count || 0})`);
        }
        
        return result;
      } else {
        console.error('❌ 세션 음성 저장 실패:', response.statusText);
        return null;
      }
    } catch (error) {
      console.error('❌ 세션 음성 저장 통신 실패:', error);
      return null;
    }
  };

  // 연속 녹음 함수 (5초 단위 처리 제거)

  // 연속 녹음 시작 함수 (deactivate까지 계속 녹음)
  const startContinuousRecording = async (): Promise<boolean> => {
    try {
      if (!audioStreamRef.current) {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 16000
          } 
        });
        audioStreamRef.current = stream;
      }

      // 세션 오디오 청크 초기화
      sessionAudioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(audioStreamRef.current, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        console.log(`📊 데이터 수신됨: ${event.data.size} bytes`);
        if (event.data.size > 0) {
          sessionAudioChunksRef.current.push(event.data); // 전체 세션에 추가
          console.log(`📦 현재 총 청크 개수: ${sessionAudioChunksRef.current.length}`);
        }
      };

      mediaRecorder.onstop = () => {
        console.log('🔴 MediaRecorder 중지됨 - 최종 데이터 처리 준비');
        console.log(`📋 최종 청크 개수: ${sessionAudioChunksRef.current.length}`);
      };

      // 연속 녹음 시작 (1초마다 데이터 수집)
      mediaRecorder.start(1000); // 1초마다 ondataavailable 이벤트 발생
      console.log('🎤 연속 녹음 시작 - deactivate까지 계속 녹음됩니다 (1초 간격으로 데이터 수집)');
      return true;
    } catch (error) {
      console.error('❌ 녹음 시작 실패:', error);
      return false;
    }
  };

  // 녹음 시작 함수 (최초 시작)
  const startRecording = async (): Promise<boolean> => {
    try {
      // 총 카운트 초기화
      const response = await fetch('http://localhost:5001/get-count');
      if (response.ok) {
        const result = await response.json();
        setTotalSnoreCount(result.total_count);
      }

      // 연속 녹음 시작 (deactivate까지 계속)
      const success = await startContinuousRecording();
      
      if (success) {
        console.log('✅ 연속 녹음 시작 완료');
        return true;
      }
      return false;
    } catch (error) {
      console.error('❌ 녹음 시작 실패:', error);
      return false;
    }
  };

  // 녹음 중지 함수
  const stopRecording = async () => {
    console.log('🔄 녹음 중지 프로세스 시작...');
    
    // 녹음 중지
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      console.log('📱 MediaRecorder 중지 중...');
      mediaRecorderRef.current.stop();
    }

    // 전체 세션 오디오 저장
    console.log(`🔍 세션 오디오 청크 개수: ${sessionAudioChunksRef.current.length}`);
    if (sessionAudioChunksRef.current.length > 0) {
      const sessionAudioBlob = new Blob(sessionAudioChunksRef.current, { type: 'audio/webm' });
      console.log(`📦 생성된 오디오 Blob 크기: ${sessionAudioBlob.size} bytes`);
      console.log('💾 전체 세션 오디오를 database/data에 저장 중...');
      
      try {
        const result = await saveSessionAudioToBackend(sessionAudioBlob);
        if (result) {
          console.log('✅ 파일 저장 성공:', result);
        } else {
          console.error('❌ 파일 저장 실패');
        }
      } catch (error) {
        console.error('❌ 파일 저장 오류:', error);
      }
    } else {
      console.log('⚠️ 저장할 오디오 데이터가 없습니다 - sessionAudioChunksRef.current가 비어있음');
    }

    // 스트림 정리
    if (audioStreamRef.current) {
      console.log('🧹 오디오 스트림 정리 중...');
      audioStreamRef.current.getTracks().forEach(track => track.stop());
      audioStreamRef.current = null;
    }

    console.log('⏹️ 녹음 중지 완료');
  };

  // 로컬 다운로드용 저장 함수 (필요시 사용)
  const saveRecording = () => {
    if (sessionAudioChunksRef.current.length === 0) return;

    const audioBlob = new Blob(sessionAudioChunksRef.current, { type: 'audio/webm' });
    const now = new Date();
    const fileName = `recording_${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}${now.getSeconds().toString().padStart(2, '0')}.wav`;
    
    // 다운로드 링크 생성
    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.style.display = 'none';
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // URL 객체 해제
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    
    console.log(`📁 녹음 파일 로컬 저장: ${fileName}`);
  };

  const toggleActivation = async () => {
    if (!isActivated) {
      // Activate: 녹음 시작
      console.log('▶️ Activate 버튼 클릭 - 녹음 시작 시도');
      const success = await startRecording();
      if (success) {
        setIsActivated(true);
        console.log('✅ 녹음 활성화 완료');
      } else {
        console.error('❌ 녹음 시작 실패');
      }
    } else {
      // Deactivate: 녹음 중지
      console.log('⏹️ Deactivate 버튼 클릭 - 녹음 중지 시도');
      await stopRecording();
      setIsActivated(false);
      console.log('✅ 녹음 비활성화 완료');
    }
  };
  return (
    <div className={`home-container ${isActivated ? 'activated' : ''}`}>
      {/* 메인 배경 */}
      <div className={`home-background ${isActivated ? 'activated' : ''}`}></div>
      
      {/* Activate/Deactivate 버튼 */}
      <div className={`activate-button-container ${isActivated ? 'activated' : ''}`} onClick={toggleActivation}>
        <div className="activate-button">{isActivated ? 'Deactivate' : 'Activate'}</div>
      </div>
      
      {/* 메인 이미지 */}
      <img 
        className="main-image" 
        src={p3hImage} 
        alt="Main"
      />
      
      {/* 하단 네비게이션 바 */}
      <div className="bottom-nav">
        {/* 홈 아이콘 (활성화된 상태) */}
        <div className="nav-item nav-home active">
          <div className="nav-background-left"></div>
          <img 
            className="nav-icon-home" 
            src={home_color} 
            alt="Home_color"
          />
        </div>
        
        {/* 중간 아이콘 - Report로 이동 */}
        <div className="nav-item nav-middle" onClick={onGoToReport}>
          <div className="nav-background-middle"></div>
          <img 
            className="nav-icon-middle" 
            src={polygon} 
            alt="Polygon"
          />
        </div>

        {/* 중간 아이콘 - Graph로 이동 */}
        <div className="nav-item nav-middle" onClick={onGoToGraph}>
          <div className="nav-background-middle"></div>
          <img 
            className="nav-icon-middle" 
            src={graph} 
            alt="Graph"
          />
        </div>
        
        {/* 오른쪽 아이콘 - Setting으로 이동 */}
        <div className="nav-item nav-right" onClick={onGoToSetting}>
          <div className="nav-background-right"></div>
          <img 
            className="nav-icon-right" 
            src={setting} 
            alt="Settings"
          />
        </div>
      </div>
      
      {/* 시간 표시 - 활성화된 상태에서만 보임 */}
      {!isActivated && (
        <div className="time-display">
          <img src={alarm} alt="Alarm" />
          
          {/* 알람 상태에 따른 표시 */}
          {isAlarmOn ? (
            /* 시간 선택 영역 */
            <div className="time-selector">
            {/* 시 선택 */}
            <select 
              className="time-select hour-select"
              value={selectedHour}
              onChange={(e) => setSelectedHour(Number(e.target.value))}
            >
              {hours.map(hour => (
                <option key={hour} value={hour}>{hour.toString().padStart(2, '0')}</option>
              ))}
            </select>
            
            <span className="time-separator">:</span>
            
            {/* 분 선택 */}
            <select 
              className="time-select minute-select"
              value={selectedMinute}
              onChange={(e) => setSelectedMinute(Number(e.target.value))}
            >
              {minutes.map(minute => (
                <option key={minute} value={minute}>{minute.toString().padStart(2, '0')}</option>
              ))}
            </select>
            
            {/* AM/PM 선택 */}
            <select 
              className="time-select period-select"
              value={selectedPeriod}
              onChange={(e) => setSelectedPeriod(e.target.value as 'AM' | 'PM')}
            >
              <option value="AM">AM</option>
              <option value="PM">PM</option>
            </select>
          </div>
        ) : (
          /* No alarm 텍스트 */
          <div className="no-alarm-text">No alarm</div>
        )}
        
          <div className="toggle-container" onClick={toggleAlarm}>
            <div className={`toggle-background ${isAlarmOn ? 'on' : 'off'}`}></div>
            <div className={`toggle-circle ${isAlarmOn ? 'on' : 'off'}`}></div>
          </div>
        </div>
      )}

      {/* 활성화 상태에서 간단한 시간 표시 */}
      {isActivated && (
        <div className="activated-time-display">
          <img src={alarm} alt="Alarm" />
          <span className="simple-time">{formatTime(selectedHour, selectedMinute)} {selectedPeriod}</span>
          <div className="toggle-container" onClick={toggleAlarm}>
            <div className={`toggle-background ${isAlarmOn ? 'on' : 'off'}`}></div>
            <div className={`toggle-circle ${isAlarmOn ? 'on' : 'off'}`}></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;