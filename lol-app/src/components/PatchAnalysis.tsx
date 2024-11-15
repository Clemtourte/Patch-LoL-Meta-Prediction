import React, { useState, useEffect } from "react";

interface PatchData {
  match_count: number;
  match_stats: {
    avg_duration: number;
  };
  champion_stats: {
    [champion: string]: {
      games_played: number;
      win_rate: number;
      pick_rate: number;
    };
  };
  role_analysis: {
    [role: string]: {
      [champion: string]: {
        games_played: number;
        win_rate: number;
        pick_rate: number;
      };
    };
  };
}

const PatchAnalysis: React.FC = () => {
  const [selectedPatch, setSelectedPatch] = useState<string>("");
  const [patchData, setPatchData] = useState<PatchData | null>(null);
  const [patches, setPatches] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/patches")
      .then((response) => response.json())
      .then((data) => {
        const sortedPatches = data.sort((a, b) => {
          const [aMajor, aMinor] = a.split(".").map(Number);
          const [bMajor, bMinor] = b.split(".").map(Number);
          return bMajor - aMajor || bMinor - aMinor;
        });
        setPatches(sortedPatches);
      })
      .catch((error) => {
        console.error("Error fetching patches:", error);
        setError("Failed to load patches");
      });
  }, []);

  const handlePatchChange = (event) => {
    setSelectedPatch(event.target.value);
  };

  useEffect(() => {
    if (selectedPatch) {
      setLoading(true);
      setError(null);
      fetch(`/api/patch_analysis/${selectedPatch}`)
        .then((response) => response.json())
        .then((data) => {
          setPatchData(data);
          setLoading(false);
        })
        .catch((error) => {
          console.error("Error fetching patch data:", error);
          setError("Failed to load patch data");
          setLoading(false);
        });
    }
  }, [selectedPatch]);

  const roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];

  return (
    <div className="patch-analysis">
      <h2>Patch Analysis</h2>
      <select onChange={handlePatchChange} value={selectedPatch}>
        <option value="">Select a patch</option>
        {patches.map((patch) => (
          <option key={patch} value={patch}>
            Patch {patch}
          </option>
        ))}
      </select>

      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}

      {patchData && (
        <div className="patch-data">
          <h3>Patch {selectedPatch}</h3>
          <p>Total Matches: {patchData.match_count}</p>

          <h4>Match Statistics</h4>
          <ul>
            <li>
              Average Duration:{" "}
              {Math.round(patchData.match_stats.avg_duration / 60)} minutes
            </li>
          </ul>

          <h4>Top Champions Overall</h4>
          <ul>
            {Object.entries(patchData.champion_stats)
              .filter(([_, stats]) => stats.games_played >= 5)
              .sort((a, b) => b[1].games_played - a[1].games_played)
              .slice(0, 5)
              .map(([champion, stats]) => {
                const wins = Math.round(
                  (stats.games_played * stats.win_rate) / 100
                );
                return (
                  <li key={champion}>
                    {champion}: {wins} wins / {stats.games_played} games (Win
                    Rate: {stats.win_rate.toFixed(1)}%), Pick Rate{" "}
                    {stats.pick_rate.toFixed(2)}%
                  </li>
                );
              })}
          </ul>

          <h4>Role Analysis</h4>
          <div className="role-analysis">
            {roles.map((role) => (
              <div key={role} className="role-column">
                <h5>{role}</h5>
                <ul>
                  {Object.entries(patchData.role_analysis[role] || {})
                    .filter(([_, stats]) => stats.games_played >= 5)
                    .sort((a, b) => b[1].games_played - a[1].games_played)
                    .slice(0, 5)
                    .map(([champion, stats]) => {
                      const wins = Math.round(
                        (stats.games_played * stats.win_rate) / 100
                      );
                      return (
                        <li key={champion}>
                          {champion}: {wins} wins / {stats.games_played} games
                          (Win Rate: {stats.win_rate.toFixed(1)}%), Pick Rate{" "}
                          {stats.pick_rate.toFixed(2)}%
                        </li>
                      );
                    })}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PatchAnalysis;
