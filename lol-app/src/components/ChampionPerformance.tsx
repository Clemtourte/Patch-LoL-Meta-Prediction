import React, { useState, useEffect } from "react";
import axios from "axios";

interface ChampionData {
  champion_name: string;
  position: string;
  games: number;
  win_rate: number;
  rating: number;
  avg_score: number;
  median_score: number;
  min_score: number;
  max_score: number;
}

const positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];

const ChampionPerformance: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<ChampionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const response = await axios.get<ChampionData[]>(
          "http://localhost:5000/api/champion_performance"
        );
        setPerformanceData(response.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch champion performance data");
        setLoading(false);
      }
    };

    fetchPerformanceData();
  }, []);

  if (loading) return <div>Loading champion performance data...</div>;
  if (error) return <div className="error">Error: {error}</div>;
  if (!performanceData || performanceData.length === 0) {
    return (
      <div>No performance data available. Try playing more games first.</div>
    );
  }

  const groupedData = performanceData.reduce((acc, champion) => {
    if (!acc[champion.position]) {
      acc[champion.position] = [];
    }
    acc[champion.position].push(champion);
    return acc;
  }, {} as Record<string, ChampionData[]>);

  Object.keys(groupedData).forEach((position) => {
    groupedData[position].sort((a, b) => b.avg_score - a.avg_score);
  });

  return (
    <div className="champion-performance">
      <h2>Champion Performance by Role</h2>
      <div className="role-columns">
        {positions.map((position) => (
          <div key={position} className="role-column">
            <h3>{position}</h3>
            <div className="champions-list">
              {groupedData[position]?.map((champ) => (
                <div key={champ.champion_name} className="champion-card">
                  <h4>{champ.champion_name}</h4>
                  <p>Games: {champ.games}</p>
                  <p>Win Rate: {champ.win_rate.toFixed(2)}%</p>
                  <p>Avg Score: {champ.avg_score.toFixed(2)}</p>
                  <p>Median Score: {champ.median_score.toFixed(2)}</p>
                  <p>
                    Score Range: {champ.min_score.toFixed(2)} to{" "}
                    {champ.max_score.toFixed(2)}
                  </p>
                  <p>Original Rating: {champ.rating.toFixed(2)}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChampionPerformance;
