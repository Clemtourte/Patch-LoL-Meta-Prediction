import React, { useState, useEffect } from "react";
import axios from "axios";

interface ChampionData {
  champion_name: string;
  win_rate: number;
  games: number;
  rating: number;
  position: string;
}

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

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div>
      <h2>Champion Performance</h2>
      <ul>
        {performanceData.map((champ, index) => (
          <li key={index}>
            {champ.champion_name}: Win Rate - {champ.win_rate.toFixed(2)}%,
            Games Played - {champ.games}, Position - {champ.position}, Rating -{" "}
            {champ.rating.toFixed(2)}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ChampionPerformance;
