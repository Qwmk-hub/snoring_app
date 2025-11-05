import React from 'react';

interface LoginProps {
  onLogin: () => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  return (
    <div className="login-container">
      {/* 헤더 */}
      <div className="login-header">
        <div className="login-title">로그인</div>
      </div>
    
      {/* 아이디 라벨 */}
      <div className="id-label">아이디</div>
      
      {/* 아이디 입력 필드 */}
      <div className="id-input-container">
        <input 
          type="text" 
          placeholder="아이디를 입력하세요"
          className="id-input"
        />
      </div>
      
      {/* 비밀번호 라벨 */}
      <div className="password-label">비밀번호</div>
      
      {/* 비밀번호 입력 필드 */}
      <div className="password-input-container">
        <input 
          type="password" 
          placeholder="비밀번호를 입력하세요"
          className="password-input"
        />
      </div>
      
      {/* 로그인 버튼 */}
      <div className="login-button-container">
        <button className="login-button" onClick={onLogin}>로그인</button>
      </div>
      
      {/* 하단 링크 */}
      <div className="bottom-links">
        아이디 찾기  |  비밀번호 찾기  | 회원가입
      </div>
    </div>
  );
};

export default Login;