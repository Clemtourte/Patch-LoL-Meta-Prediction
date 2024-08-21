import streamlit as st
from main import UserData, UserDisplay, save_match_data

st.title("League of Legends User Information")

region = st.selectbox("Select your region:", ["EUW1"]) 
username = st.text_input("Enter your username:")
tag = st.text_input("Enter your tag (without #):")

if st.button("Search"):
    if not username or not tag:
        st.error("Please enter both username and tag.")
    else:
        user_data = UserData(username, tag, region)
        user_display = UserDisplay(user_data)

        rank_info = user_display.display_user_info()

        if rank_info:
            st.header("Rank Information")
            st.write(f"**Tier**: {rank_info[0]} {rank_info[1]}")
            st.write(f"**League Points**: {rank_info[2]}")
        else:
            st.error("User might not be ranked yet.")

        match_ids = user_data.get_matches('ranked', 1)

        if match_ids:
            st.header("Recent Matches")
            for i, match_id in enumerate(match_ids, 1):
                match_info = user_display.display_match_info(match_id)
                
                if match_info:  # match_info is now a dictionary
                    st.subheader(f"Match {i}")
                    st.write(f"**Game Duration**: {match_info['game_duration']}")
                    st.write(f"**Patch**: {match_info['patch']}")
                    st.write(f"**Timestamp**: {match_info['timestamp']}")
                    st.write("**Participants Info:**")
                    for team, participants in match_info['teams'].items():
                        st.write(f"**{team}**")
                        for summoner, details in participants.items():
                            st.write(f"{summoner}: {details}")
                else:
                    st.error(f"Match {i} information could not be retrieved or is not in the expected format.")
            save_match_data(user_data, match_ids)
        else:
            st.error("No recent matches found.")