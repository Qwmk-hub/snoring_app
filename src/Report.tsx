import React from 'react';
import home_color from './assets/images/home_color.png';
import home from './assets/images/home.png';
import polygon from './assets/images/polygon.png';
import graph from './assets/images/graph.png';
import setting from './assets/images/settings.png';
import x from './assets/images/delete.png';
import trash from './assets/images/trash.png';

interface ReportProps {
  onBack: () => void;
  onTodayClick?: () => void;
  onGoToGraph?: () => void;
  onGoToSetting?: () => void;
}

const Report: React.FC<ReportProps> = ({ onBack, onTodayClick, onGoToGraph, onGoToSetting }) => {
  const reportData = [
    { date: 'Today', duration: 'Today', clips: 0 },
    { date: '2025-10-17', duration: '6h 12m', clips: 50 },
    { date: '2025-10-9', duration: '7h 00m', clips: 21 }
  ];

  return (
    <div className="report-container">
      {/* 메인 배경 */}
      <div className="report-background"></div>
      
      {/* 상단 헤더 */}
      <div className="report-header">
        <div className="report-title">Library</div>
      </div>
      
      {/* October 2025 섹션 */}
      <div className="month-section october">
        <div className="month-header">
          <img 
            className="left-icon" 
            src={polygon} 
            alt="october-polygon"
          />
          <div className="month-text">October 2025</div>
          <img 
            className="right-icon" 
            src={x} 
            alt="october-x"
          />
        </div>
      </div>
      
      {/* 리포트 아이템들 */}
      {reportData.map((item, index) => (
        <div 
          key={index} 
          className="report-item"
          onClick={item.date === 'Today' ? onTodayClick : undefined}
          style={item.date === 'Today' ? { cursor: 'pointer' } : {}}
        >
          <div className="report-date">{item.date}</div>
          <div className="report-stats">Active {item.duration} | {item.clips} Clips</div>
          <img 
            className="right-icon-trash" 
            src={trash} 
            alt="october-trash"
          />
        </div>
      ))}
      
      {/* September 2025 섹션 */}
      <div className="month-section september">
        <div className="month-header">
          <img 
            className="left-icon" 
            src={polygon} 
            alt="october-polygon"
          />
          <div className="month-text">September 2025</div>
          <img 
            className="right-icon" 
            src={x} 
            alt="october-x"
          />
        </div>
      </div>
      
      {/* August 2025 섹션 */}
      <div className="month-section august">
        <div className="month-header">
          <img 
            className="left-icon" 
            src={polygon} 
            alt="october-polygon"
          />
          <div className="month-text">August 2025</div>
          <img 
            className="right-icon" 
            src={x} 
            alt="october-x"
          />
        </div>
      </div>
      
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
    </div>
  );
};

export default Report;