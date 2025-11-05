import React, { useState, useEffect } from 'react';
import './assets/css/Design.css';
import home from './assets/images/home.png';
import polygon from './assets/images/polygon.png';
import graph from './assets/images/graph.png';
import setting from './assets/images/settings.png';
import arrow from './assets/images/arrow.png';
import x from './assets/images/delete.png';
import trash from './assets/images/trash.png';

interface TodayProps {
  onBack: () => void;
}

const Today: React.FC<TodayProps> = ({ onBack }) => {
  const [snoreCount, setSnoreCount] = useState(0);
  const [timeDisplay, setTimeDisplay] = useState('0 minutes 0 seconds');

  // 초를 분:초 형태로 변환하는 함수
  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes === 0) {
      return `${remainingSeconds} seconds`;
    } else if (remainingSeconds === 0) {
      return `${minutes} minutes`;
    } else {
      return `${minutes} minutes ${remainingSeconds} seconds`;
    }
  };

  // API에서 데이터 가져오기
  const fetchSnoreCount = async () => {
    try {
      const response = await fetch('http://localhost:5001/get-count');
      if (response.ok) {
        const result = await response.json();
        setSnoreCount(result.total_count);
        setTimeDisplay(formatTime(result.total_count));
      } else {
        console.error('❌ 데이터 가져오기 실패:', response.statusText);
      }
    } catch (error) {
      console.error('❌ API 통신 실패:', error);
    }
  };

  // 컴포넌트 마운트 시 데이터 가져오기
  useEffect(() => {
    fetchSnoreCount();
    
    // 5초마다 데이터 갱신
    const interval = setInterval(fetchSnoreCount, 5000);
    
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="today-container">
      {/* 메인 배경 */}
      <div className="today-main-background"></div>
      
      {/* 상단 헤더 배경 */}
      <div className="today-header-background"></div>
      
      {/* 닫기 버튼 */}
      <div className="today-close-button" onClick={onBack}>
        <img 
        className="right-trash" 
        src={trash} 
        alt="Trash"
        />
      </div>
      
      {/* 뒤로가기 버튼 */}
      <div className="today-back-button" onClick={onBack}>
        <img 
        className="nav-icon-home" 
        src={arrow} 
        alt="Arrow"
        />
      </div>
      
      {/* Today 제목 */}
      <div className="today-title">Today</div>
      
      {/* 시간 표시 */}
      <div className="today-time-display">{timeDisplay}</div>
      
      {/* 하단 네비게이션 바 */}
      <div className="bottom-nav">
        {/* 홈 아이콘 */}
        <div className="nav-item nav-home" onClick={onBack}>
          <div className="nav-background-middle"></div>
          <img 
            className="nav-icon-home" 
            src={home} 
            alt="Home"
          />
        </div>
        
        {/* 중간 아이콘 (활성화) */}
        <div className="nav-item nav-middle active">
          <div className="nav-background-middle-active"></div>
          <img 
            className="nav-icon-middle" 
            src={polygon} 
            alt="Polygon"
          />
        </div>
        
        {/* 중간 아이콘 */}
        <div className="nav-item nav-middle">
          <div className="nav-background-middle"></div>
          <img 
            className="nav-icon-middle" 
            src={graph} 
            alt="Graph"
          />
        </div>
        
        {/* 오른쪽 아이콘 */}
        <div className="nav-item nav-right">
          <div className="nav-background-right"></div>
          <img 
            className="nav-icon-right" 
            src={setting} 
            alt="Settings"
          />
        </div>
      </div>
    </div>
  );
};

export default Today;
