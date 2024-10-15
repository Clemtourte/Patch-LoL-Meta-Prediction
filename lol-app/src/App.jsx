import React, { useState } from "react";
import "./App.css";
import UserInfo from "./components/UserInfo";
import ChampionPerformance from "./components/ChampionPerformance";
import PatchAnalysis from "./components/PatchAnalysis";

export default function App() {
  const [currentPage, setCurrentPage] = useState("userInfo");

  return (
    <div className="container">
      <nav>
        <ul>
          <li>
            <button onClick={() => setCurrentPage("userInfo")}>
              User Info
            </button>
          </li>
          <li>
            <button onClick={() => setCurrentPage("championPerformance")}>
              Champion Performance
            </button>
          </li>
          <li>
            <button onClick={() => setCurrentPage("patchAnalysis")}>
              Patch Analysis
            </button>
          </li>
        </ul>
      </nav>

      <main>
        {currentPage === "userInfo" && <UserInfo />}
        {currentPage === "championPerformance" && <ChampionPerformance />}
        {currentPage === "patchAnalysis" && <PatchAnalysis />}
      </main>
    </div>
  );
}
