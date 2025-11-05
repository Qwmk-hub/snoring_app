import React, { useState } from 'react';
import Login from './Login';
import Home from './Home';
import Report from './Report';
import Today from './Today';
import Graph from './Graph';
import Setting from './Setting';

function App() {
  const [currentScreen, setCurrentScreen] = useState<'login' | 'home' | 'report' | 'today' | 'graph' | 'setting'>('login');

  const handleLogin = () => {
    setCurrentScreen('home');
  };

  const handleLogout = () => {
    setCurrentScreen('login');
  };

  const handleGoToReport = () => {
    setCurrentScreen('report');
  };

  const handleBackToHome = () => {
    setCurrentScreen('home');
  };

  const handleGoToToday = () => {
    setCurrentScreen('today');
  };

  const handleBackToReport = () => {
    setCurrentScreen('report');
  };

  const handleGoToGraph = () => {
    setCurrentScreen('graph');
  };

  const handleBackToHomeFromGraph = () => {
    setCurrentScreen('home');
  };

  const handleGoToSetting = () => {
    setCurrentScreen('setting');
  };

  const handleBackToHomeFromSetting = () => {
    setCurrentScreen('home');
  };

  return (
    <div className="App">
      {currentScreen === 'login' ? (
        <Login onLogin={handleLogin} />
      ) : currentScreen === 'home' ? (
        <Home onLogout={handleLogout} onGoToReport={handleGoToReport} onGoToGraph={handleGoToGraph} onGoToSetting={handleGoToSetting} />
      ) : currentScreen === 'report' ? (
        <Report onBack={handleBackToHome} onTodayClick={handleGoToToday} onGoToGraph={handleGoToGraph} onGoToSetting={handleGoToSetting} />
      ) : currentScreen === 'today' ? (
        <Today onBack={handleBackToReport} />
      ) : currentScreen === 'graph' ? (
        <Graph onBack={handleBackToHomeFromGraph} onGoToReport={handleGoToReport} onGoToSetting={handleGoToSetting} />
      ) : currentScreen === 'setting' ? (
        <Setting onBack={handleBackToHomeFromSetting} onGoToReport={handleGoToReport} onGoToGraph={handleGoToGraph} />
      ) : null}
    </div>
  );
}

export default App;