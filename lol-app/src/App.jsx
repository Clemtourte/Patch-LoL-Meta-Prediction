import React, { useState } from 'react';
import UserInfo from './components/UserInfo';
import ChampionPerformance from './components/ChampionPerformance';

export default function App() {
  const [currentPage, setCurrentPage] = useState('userInfo');

  return (
    <div className="container">
      <nav>
        <ul>
          <li>
            <button onClick={() => setCurrentPage('userInfo')}>User Info</button>
          </li>
          <li>
            <button onClick={() => setCurrentPage('championPerformance')}>Champion Performance</button>
          </li>
        </ul>
      </nav>

      <main>
        {currentPage === 'userInfo' && <UserInfo />}
        {currentPage === 'championPerformance' && <ChampionPerformance />}
      </main>
    </div>
  );
}