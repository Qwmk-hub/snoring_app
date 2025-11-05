import React from 'react';
import home from './assets/images/home.png';
import polygon from './assets/images/polygon.png';
import graph from './assets/images/graph.png';
import setting from './assets/images/settings.png';
import './assets/css/Design.css';

interface SettingProps {
  onBack: () => void;
  onGoToReport: () => void;
  onGoToGraph: () => void;
}

const Setting: React.FC<SettingProps> = ({ onBack, onGoToReport, onGoToGraph }) => {
  return (
    <div className="setting-container">
      {/* 메인 배경 */}
      <div className="setting-background"></div>
      
      {/* 상단 헤더 */}
      <div className="setting-header">
        <div className="setting-title">Settings</div>
      </div>
      
      {/* 설정 메뉴 섹션 */}
      <div className="setting-menu-section">
        
        {/* 계정 설정 메뉴 */}
        <div className="setting-menu-item">
          <div className="setting-menu-icon-container">
            <div className="setting-menu-icon account-icon"></div>
          </div>
          <div className="setting-menu-content">
            <div className="setting-menu-title">계정 설정</div>
            <div className="setting-menu-subtitle">Manage your account information</div>
          </div>
          <div className="setting-menu-arrow">→</div>
        </div>
        
        {/* 로그아웃 메뉴 */}
        <div className="setting-menu-item logout-item">
          <div className="setting-menu-icon-container">
            <div className="setting-menu-icon logout-icon"></div>
          </div>
          <div className="setting-menu-content">
            <div className="setting-menu-title">로그아웃</div>
            <div className="setting-menu-subtitle">Sign out of your account</div>
          </div>
          <div className="setting-menu-arrow">→</div>
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
        
        {/* 중간 아이콘 - Report */}
        <div className="nav-item nav-middle" onClick={onGoToReport}>
          <div className="nav-background-middle"></div>
          <img 
            className="nav-icon-middle" 
            src={polygon} 
            alt="Polygon"
          />
        </div>
        
        {/* 중간 아이콘 - Graph */}
        <div className="nav-item nav-middle" onClick={onGoToGraph}>
          <div className="nav-background-middle"></div>
          <img 
            className="nav-icon-middle" 
            src={graph} 
            alt="Graph"
          />
        </div>
        
        {/* 오른쪽 아이콘 (활성화된 상태) */}
        <div className="nav-item nav-right active">
          <div className="nav-background-right-active"></div>
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

export default Setting;
