import React, { useState } from "react";
import axios from "axios";

interface UserInfoData {
  ranked?: string;
  flex?: string;
}

const UserInfo: React.FC = () => {
  const [userInfo, setUserInfo] = useState<UserInfoData | null>(null);
  const [region, setRegion] = useState("");
  const [username, setUsername] = useState("");
  const [tag, setTag] = useState("");
  const [error, setError] = useState("");

  const fetchUserInfo = async () => {
    try {
      const response = await axios.get<UserInfoData>(
        `http://localhost:5000/api/user_info/${region}/${username}/${tag}`
      );
      setUserInfo(response.data);
      setError("");
    } catch (error) {
      console.error("Error fetching user info:", error);
      setError("Failed to fetch user info. Please try again.");
      setUserInfo(null);
    }
  };

  return (
    <div className="container mt-5">
      <h2>User Information</h2>
      <div className="mb-3">
        <input
          type="text"
          className="form-control mb-2"
          placeholder="Region"
          value={region}
          onChange={(e) => setRegion(e.target.value)}
        />
        <input
          type="text"
          className="form-control mb-2"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="text"
          className="form-control mb-2"
          placeholder="Tag"
          value={tag}
          onChange={(e) => setTag(e.target.value)}
        />
        <button className="btn btn-primary" onClick={fetchUserInfo}>
          Fetch User Info
        </button>
      </div>
      {error && <p className="text-danger">{error}</p>}
      {userInfo && (
        <div>
          <h3>Ranked Solo: {userInfo.ranked || "Not Available"}</h3>
          <h3>Ranked Flex: {userInfo.flex || "Not Available"}</h3>
        </div>
      )}
    </div>
  );
};

export default UserInfo;
