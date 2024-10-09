import React, { useState } from "react";
import axios from "axios";

interface UserInfoData {
  ranked?: string;
  flex?: string;
  recentMatches?: {
    gameId: string;
    date: string;
    duration: string;
    patch: string;
    winningTeam: string;
    participants: {
      summonerName: string;
      champion: string;
      team: string;
      result: string;
      kda: string;
      kdaRaw: string;
      cs: number;
    }[];
  }[];
  championStats?: {
    championName: string;
    games: number;
    winRate: number;
    kda: number;
    avgCs: number;
  }[];
}

const UserInfo: React.FC = () => {
  const [userInfo, setUserInfo] = useState<UserInfoData | null>(null);
  const [region, setRegion] = useState("");
  const [username, setUsername] = useState("");
  const [tag, setTag] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [expandedMatches, setExpandedMatches] = useState<{
    [key: string]: boolean;
  }>({});
  const [numMatches, setNumMatches] = useState(10);
  const [displayedMatches, setDisplayedMatches] = useState(5);

  const regions = [
    "EUW1",
    "NA1",
    "KR",
    "JP1",
    "EUN1",
    "BR1",
    "LA1",
    "LA2",
    "OC1",
    "RU",
    "TR1",
  ];

  const fetchUserInfo = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await axios.get(
        `http://localhost:5000/api/user_info/${region}/${username}/${tag}?num_matches=${numMatches}`
      );
      setUserInfo(response.data);
      setExpandedMatches({});
      setDisplayedMatches(5);
    } catch (error) {
      console.error("Error fetching user info:", error);
      setError("Failed to fetch user info. Please try again.");
      setUserInfo(null);
    } finally {
      setLoading(false);
    }
  };

  const toggleMatchExpansion = (gameId: string) => {
    setExpandedMatches((prev) => ({
      ...prev,
      [gameId]: !prev[gameId],
    }));
  };

  const loadMoreMatches = () => {
    setDisplayedMatches((prev) =>
      Math.min(prev + 5, userInfo?.recentMatches?.length || 0)
    );
  };

  return (
    <div className="user-info-container">
      <h2>User Information</h2>
      <div className="input-group">
        <div className="input-field">
          <label htmlFor="region">Region</label>
          <select
            id="region"
            value={region}
            onChange={(e) => setRegion(e.target.value)}
          >
            <option value="">Select Region</option>
            {regions.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <div className="input-field">
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
        </div>
        <div className="input-field">
          <label htmlFor="tag">Tag</label>
          <input
            id="tag"
            type="text"
            value={tag}
            onChange={(e) => setTag(e.target.value)}
          />
        </div>
        <div className="input-field">
          <label htmlFor="numMatches">Number of Matches</label>
          <input
            id="numMatches"
            type="number"
            min="1"
            max="100"
            value={numMatches}
            onChange={(e) => setNumMatches(Number(e.target.value))}
          />
        </div>
        <button id="fetchUserInfo" onClick={fetchUserInfo} disabled={loading}>
          {loading ? "Loading..." : "Fetch User Info"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {userInfo && (
        <div className="user-data">
          <h3>Ranked Solo: {userInfo.ranked || "Not Available"}</h3>
          <h3>Ranked Flex: {userInfo.flex || "Not Available"}</h3>

          <div className="stats-and-matches">
            {userInfo.championStats && (
              <div className="champion-stats">
                <h3>Top 5 Champion Statistics</h3>
                <ul>
                  {userInfo.championStats.map((champ, index) => (
                    <li key={index}>
                      {champ.championName} - Games: {champ.games}, Win Rate:{" "}
                      {champ.winRate.toFixed(2)}%, KDA: {champ.kda.toFixed(2)},
                      Avg CS: {champ.avgCs.toFixed(1)}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {userInfo.recentMatches && (
              <div className="recent-matches">
                <h3>Recent Matches</h3>
                <div className="matches-grid">
                  {userInfo.recentMatches
                    .slice(0, displayedMatches)
                    .map((match, index) => (
                      <div key={index} className="match">
                        <h4
                          onClick={() => toggleMatchExpansion(match.gameId)}
                          className="match-header"
                        >
                          {match.date} - Duration: {match.duration} - Patch:{" "}
                          {match.patch}{" "}
                          {expandedMatches[match.gameId] ? "▼" : "▶"}
                        </h4>
                        {expandedMatches[match.gameId] && (
                          <div className="teams">
                            <div
                              className={`team blue ${
                                match.winningTeam === "Blue"
                                  ? "winning-team"
                                  : "losing-team"
                              }`}
                            >
                              <h5>
                                Blue Team{" "}
                                {match.winningTeam === "Blue"
                                  ? "(Win)"
                                  : "(Lose)"}
                              </h5>
                              {match.participants
                                .filter((p) => p.team === "Blue")
                                .map((p, i) => (
                                  <div
                                    key={i}
                                    className={`player ${
                                      p.summonerName === username
                                        ? "highlight"
                                        : ""
                                    }`}
                                  >
                                    {p.summonerName} - {p.champion} - KDA:{" "}
                                    {p.kda} ({p.kdaRaw}) - CS: {p.cs}
                                  </div>
                                ))}
                            </div>
                            <div
                              className={`team red ${
                                match.winningTeam === "Red"
                                  ? "winning-team"
                                  : "losing-team"
                              }`}
                            >
                              <h5>
                                Red Team{" "}
                                {match.winningTeam === "Red"
                                  ? "(Win)"
                                  : "(Lose)"}
                              </h5>
                              {match.participants
                                .filter((p) => p.team === "Red")
                                .map((p, i) => (
                                  <div
                                    key={i}
                                    className={`player ${
                                      p.summonerName === username
                                        ? "highlight"
                                        : ""
                                    }`}
                                  >
                                    {p.summonerName} - {p.champion} - KDA:{" "}
                                    {p.kda} ({p.kdaRaw}) - CS: {p.cs}
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                </div>
                {displayedMatches < (userInfo.recentMatches?.length || 0) && (
                  <button
                    onClick={loadMoreMatches}
                    className="load-more-button"
                  >
                    Load More Matches
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default UserInfo;
