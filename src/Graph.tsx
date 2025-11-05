import React from 'react';
import './assets/css/Design.css';
import p3hImage from './assets/images/p3h.png';
import alarm from './assets/images/alarm.png';
import home from './assets/images/home.png';
import polygon from './assets/images/polygon.png';
import graph from './assets/images/graph.png';
import setting from './assets/images/settings.png';

interface GraphProps {
  onBack: () => void;
  onGoToReport: () => void;
  onGoToSetting: () => void;
}

const Graph: React.FC<GraphProps> = ({ onBack, onGoToReport, onGoToSetting }) => {
  return (
    <div className="graph-container">
      {/* 메인 배경 */}
      <div className="graph-main-background"></div>
      
      {/* 제목 */}
      <div className="graph-title">Record for the past week</div>
      
      {/* 요일 표시 */}
      <div className="graph-day-label graph-day-s1">S</div>
      <div className="graph-day-label graph-day-m">M</div>
      <div className="graph-day-label graph-day-t1">T</div>
      <div className="graph-day-label graph-day-w">W</div>
      <div className="graph-day-label graph-day-t2">T</div>
      <div className="graph-day-label graph-day-f">F</div>
      <div className="graph-day-label graph-day-s2">S</div>
      
      {/* 일요일 차트 */}
      <div className="graph-bar-background graph-sun-bg"></div>
      <div className="graph-bar-data graph-sun-data"></div>
      <div className="graph-time-label graph-sun-time">07:43</div>
      
      {/* 월요일 차트 */}
      <div className="graph-bar-background graph-mon-bg"></div>
      <div className="graph-time-label graph-mon-time">00:00</div>
      
      {/* 화요일 차트 */}
      <div className="graph-bar-background graph-tue-bg"></div>
      <div className="graph-bar-data graph-tue-data"></div>
      <div className="graph-time-label graph-tue-time">10:29</div>
      
      {/* 수요일 차트 */}
      <div className="graph-bar-background graph-wed-bg"></div>
      <div className="graph-time-label graph-wed-time">00:00</div>
      
      {/* 목요일 차트 */}
      <div className="graph-bar-background graph-thu-bg"></div>
      <div className="graph-time-label graph-thu-time">00:00</div>
      
      {/* 금요일 차트 */}
      <div className="graph-bar-background graph-fri-bg"></div>
      <div className="graph-bar-data graph-fri-data"></div>
      <div className="graph-time-label graph-fri-time">15:17</div>
      
      {/* 토요일 차트 */}
      <div className="graph-bar-background graph-sat-bg"></div>
      <div className="graph-time-label graph-sat-time">00:00</div>
      
      {/* 하단 네비게이션 바 */}
      <div className="bottom-nav">
        {/* 홈 아이콘 - Home으로 이동 */}
        <div className="nav-item nav-home" onClick={onBack}>
          <div className="nav-background-middle"></div>
          <img 
            className="nav-icon-home" 
            src={home} 
            alt="Home"
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

        {/* 중간 아이콘 - Graph (활성화된 상태) */}
        <div className="nav-item nav-middle active">
          <div className="nav-background-middle-active"></div>
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
    </div>
  );
};

export default Graph;