import React, { useState, useEffect } from "react";
import axios from "axios";

interface ChampionData {
  champion_name: string;
  position: string;
  games: number;
  win_rate: number;
  rating: number;
  avg_score: number;
  kill_participation: number;
  damage_share: number;
  gold_share: number;
  vision_score: number;
  cs_per_min: number;
}

const positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];

const ChampionPerformance: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<ChampionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [selectedMetric, setSelectedMetric] = useState<string>("rating");

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const response = await axios.get<ChampionData[]>(
          "http://localhost:5000/api/champion_performance"
        );
        setPerformanceData(response.data);
      } catch (err) {
        setError("Failed to fetch champion performance data");
      } finally {
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

  const groupedData = positions.reduce((acc, position) => {
    acc[position] = performanceData
      .filter((champ) => champ.position === position)
      .sort((a, b) => (b[selectedMetric] || 0) - (a[selectedMetric] || 0));
    return acc;
  }, {} as Record<string, ChampionData[]>);

  return (
    <div className="champion-performance">
      <h2>Champion Performance by Role</h2>
      <div className="metric-selector">
        <label>Sort by: </label>
        <select
          value={selectedMetric}
          onChange={(e) => setSelectedMetric(e.target.value)}
          className="metric-select"
        >
          <option value="rating">Rating</option>
          <option value="win_rate">Win Rate</option>
          <option value="kill_participation">Kill Participation</option>
          <option value="damage_share">Damage Share</option>
          <option value="gold_share">Gold Share</option>
          <option value="vision_score">Vision Score</option>
          <option value="cs_per_min">CS per Minute</option>
        </select>
      </div>
      <div className="role-columns">
        {positions.map((position) => (
          <div key={position} className="role-column">
            <h3>{position}</h3>
            <div className="champions-list">
              {(groupedData[position] || []).map((champ) => (
                <div key={champ.champion_name} className="champion-card">
                  <h4>{champ.champion_name}</h4>
                  <p>Games: {champ.games}</p>
                  <p>Win Rate: {champ.win_rate?.toFixed(2)}%</p>
                  <p>Performance Rating: {champ.rating?.toFixed(2)}</p>
                  <div className="additional-stats">
                    <p>
                      Kill Participation:{" "}
                      {(champ.kill_participation * 100)?.toFixed(1)}%
                    </p>
                    <p>
                      Damage Share: {(champ.damage_share * 100)?.toFixed(1)}%
                    </p>
                    <p>Gold Share: {(champ.gold_share * 100)?.toFixed(1)}%</p>
                    <p>Vision Score: {champ.vision_score?.toFixed(1)}</p>
                    <p>CS/min: {champ.cs_per_min?.toFixed(1)}</p>
                  </div>
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
