import streamlit as st
from main import User, save_match_data  # Ensure that the file containing the User class is named main.py

st.title("League of Legends User Information")

# Selection of region and input for username and tag
region = st.selectbox("Select your region:", ["EUW1"])  # You can add other regions if necessary
username = st.text_input("Enter your username:")
tag = st.text_input("Enter your tag (without #):")

if st.button("Search"):
    if not username or not tag:
        st.error("Please enter both username and tag.")
    else:
        # Create an instance of User
        user = User(username, tag, region)

        # Retrieve rank information
        rank_info = user.display_user_info()

        if rank_info:
            st.header("Rank Information")
            st.write(f"**Tier**: {rank_info[0]} {rank_info[1]}")
            st.write(f"**League Points**: {rank_info[2]}")
        else:
            st.error("User might not be ranked yet.")

        # Retrieve recent matches
        match_ids = user.get_matches('ranked', 1)

        if match_ids:
            st.header("Recent Matches")
            for i, match_id in enumerate(match_ids, 1):
                general_info, participant_info = user.get_match_info(match_id)
                if general_info and participant_info:
                    st.subheader(f"Match {i}")
                    st.write(f"**Game Duration**: {general_info['game_duration']}")
                    st.write(f"**Patch**: {general_info['patch']}")
                    st.write(f"**Timestamp**: {general_info['timestamp']}")
                    st.write("**Participants Info:**")
                    for team, participants in participant_info.items():
                        st.write(f"**{team}**")
                        for summoner, details in participants.items():
                            st.write(f"{summoner}: {details}")
            save_match_data(user, match_ids)
        else:
            st.error("No recent matches found.")
