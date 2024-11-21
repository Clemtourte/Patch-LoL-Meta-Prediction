import React, { useState, useEffect } from "react";

const ChampionPerformance = () => {
  const [performanceData, setPerformanceData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedRole, setSelectedRole] = useState("TOP");
  const [selectedMetric, setSelectedMetric] = useState("rating");

  const roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];
  const metrics = [
    { value: "rating", label: "Rating" },
    { value: "win_rate", label: "Win Rate" },
    { value: "kill_participation", label: "Kill Participation" },
    { value: "damage_share", label: "Damage Share" },
    { value: "gold_share", label: "Gold Share" },
    { value: "vision_score", label: "Vision Score" },
    { value: "cs_per_min", label: "CS per Minute" },
  ];

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const response = await fetch(
          "http://localhost:5000/api/champion_performance"
        );
        const data = await response.json();
        setPerformanceData(data);
      } catch (err) {
        setError("Failed to fetch champion performance data");
      } finally {
        setLoading(false);
      }
    };

    fetchPerformanceData();
  }, []);

  if (loading) return <div>Loading champion performance data...</div>;
  if (error) return <div className="error">{error}</div>;
  if (!performanceData || Object.keys(performanceData).length === 0) {
    return <div>No performance data available</div>;
  }

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Champion Performance by Role</h2>

      <div className="flex gap-4 mb-6">
        <div>
          <label className="mr-2">Role: </label>
          <select
            value={selectedRole}
            onChange={(e) => setSelectedRole(e.target.value)}
            className="p-2 border rounded"
          >
            {roles.map((role) => (
              <option key={role} value={role}>
                {role}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mr-2">Sort by: </label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="p-2 border rounded"
          >
            {metrics.map((metric) => (
              <option key={metric.value} value={metric.value}>
                {metric.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {performanceData[selectedRole]
          ?.sort((a, b) => b[selectedMetric] - a[selectedMetric])
          .map((champion) => (
            <div
              key={champion.champion_name}
              className="bg-white shadow rounded-lg p-4"
            >
              <h3 className="text-xl font-semibold mb-2">
                {champion.champion_name}
              </h3>
              <div className="space-y-2">
                <p>Games: {champion.games}</p>
                <p>Win Rate: {champion.win_rate.toFixed(1)}%</p>
                <p>Rating: {champion.rating.toFixed(2)}</p>
                <div className="text-sm text-gray-600">
                  <p>
                    Kill Participation:{" "}
                    {(champion.kill_participation * 100).toFixed(1)}%
                  </p>
                  <p>
                    Damage Share: {(champion.damage_share * 100).toFixed(1)}%
                  </p>
                  <p>Gold Share: {(champion.gold_share * 100).toFixed(1)}%</p>
                  <p>Vision Score: {champion.vision_score.toFixed(1)}</p>
                  <p>CS/min: {champion.cs_per_min.toFixed(1)}</p>
                </div>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
};

export default ChampionPerformance;
