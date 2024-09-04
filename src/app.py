import streamlit as st
from main import UserData, UserDisplay, save_match_data, Session

st.title("League of Legends User Information")

region = st.selectbox("Select your region:", ["EUW1"]) 
username = st.text_input("Enter your username:")
tag = st.text_input("Enter your tag (without #):")
number_of_games = st.number_input("Number of recent games to display:", min_value=1, max_value=20, value=1)

if st.button("Search"):
    if not username or not tag:
        st.error("Please enter both username and tag.")
    else:
        user_data = UserData(username, tag, region)
        user_display = UserDisplay(user_data)

        rank_info = user_display.display_user_info()

        if rank_info:
            st.header("Rank Information")
            st.write(f"**Tier**: {rank_info}")
        else:
            st.error("User might not be ranked yet or an error occurred.")

        match_ids = user_data.get_matches('ranked', number_of_games)

        if match_ids:
            st.header("Recent Matches")
            session = Session()  # Create a new session
            
            try:
                for i, match_id in enumerate(match_ids, 1):
                    match_info = user_display.display_match_info(match_id)

                    if match_info:
                        st.subheader(f"Match {i}")
                        st.write(f"**Game Duration**: {match_info['Duration']}")
                        st.write(f"**Patch**: {match_info['Patch']}")
                        st.write(f"**Timestamp**: {match_info['Date']}")
                        st.write(f"**Game Mode**: {match_info['Game Type']}")
                        st.write("**Participants Info:**")

                        for team, participants in match_info["Teams"].items():
                            team_win_status = "Win" if next(iter(participants.values()))['win'] else "Lose"
                            st.write(f"**{team} - {team_win_status}**")
                            for summoner, details in participants.items():
                                with st.expander(f"**{details['summoner_name']}** ({details['champ_name']}) - KDA: {details['kills']}/{details['deaths']}/{details['assists']}"):
                                    st.write(f"**Role:** {details['role']}")
                                    st.write(f"**Lane:** {details['lane']}")
                                    st.write(f"**Position:** {details['position']}")
                                    st.write(f"**Level:** {details['champ_level']}")
                                    st.write(f"**Total Damage Dealt:** {details['total_damage_dealt']}")
                                    st.write(f"**Total Gold Earned:** {details['gold_earned']}")
                                    st.write(f"**CS:** {details['cs']}")

                    else:
                        st.error(f"Match {i} information could not be retrieved or is not in the expected format.")
                        st.write(f"Raw match data: {user_data.get_match_data(match_id)}")
                    
                # Save match data to the database
                save_match_data(user_data, match_ids, session)

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                session.close()

        else:
            st.error("No recent matches found.")