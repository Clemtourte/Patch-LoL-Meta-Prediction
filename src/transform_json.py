import pandas as pd
import json
import streamlit as st


with open("../datasets/match_data.json", "r") as f:
    data = json.load(f)

matches = []
teams = []
participants = []

for match_key, match_data in data.items():
    match_id = int(match_key.split("_")[1])

    # Match info
    matches.append({
        "match_id": match_id,
        "game_id": match_data["general_info"]["game_id"],
        "game_duration": match_data["general_info"]["game_duration"],
        "patch": match_data["general_info"]["patch"],
        "timestamp": match_data["general_info"]["timestamp"],
        "mode": match_data["general_info"]["mode"],
    })

    # Teams info
    for team_name, players in match_data["participant_info"].items():
        team_id = len(teams) + 1
        win = next(iter(players.values()))["win"]
        teams.append({
            "team_id": team_id,
            "match_id": match_id,
            "team_name": team_name,
            "win": win,
        })

        # Participants info
        for summoner_key, player_data in players.items():
            participants.append({
                "participant_id": len(participants) + 1,
                "team_id": team_id,
                "summoner_id": player_data["summoner_id"],
                "summoner_name": player_data["summoner_name"],
                "champion_name": player_data["champ_name"],
                "champion_id": player_data["champ_id"],
                "champ_level": player_data["champ_level"],
                "kills": player_data["kills"],
                "deaths": player_data["deaths"],
                "assists": player_data["assists"],
                "kda": player_data["kda"],
                "gold_earned": player_data["gold_earned"],
                "total_damage_dealt": player_data["total_damage_dealt"],
                "cs": player_data["cs"]
            })

df_matches = pd.DataFrame(matches)
df_teams = pd.DataFrame(teams)
df_participants = pd.DataFrame(participants)

st.title("League of Legends DataFrames")

st.header("Matches")
st.dataframe(df_matches)

st.header("Teams")
st.dataframe(df_teams)

st.header("Participants")
st.dataframe(df_participants)