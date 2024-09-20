import streamlit as st
from main import UserData, UserDisplay, save_match_data, Session
from models import Match, Team, Participant
import time

st.title("League of Legends User Information")

region = st.selectbox("Select your region:", ["EUW1"])  # Add more regions as needed
username = st.text_input("Enter your username:")
tag = st.text_input("Enter your tag (without #):")

st.header("Database Population")
population_mode = st.radio("Choose population mode:", ["Display Only", "Add to Database", "Both"])
number_of_games = st.number_input("Number of recent games to process:", min_value=1, max_value=100, value=10)

if st.button("Process Games"):
    if not username or not tag:
        st.error("Please enter both username and tag.")
    else:
        user_data = UserData(username, tag, region)
        user_display = UserDisplay(user_data)

        rank_info = user_display.display_user_info()
        if rank_info:
            st.subheader("Rank Information")
            st.write(f"**Tier**: {rank_info}")
        else:
            st.error("User might not be ranked yet or an error occurred.")

        match_ids = user_data.get_matches('ranked', number_of_games)

        if match_ids:
            st.subheader(f"Processing {len(match_ids)} Recent Matches")
            progress_bar = st.progress(0)
            session = Session()
            
            try:
                for i, match_id in enumerate(match_ids):
                    match_info = user_display.display_match_info(match_id)

                    if match_info:
                        if population_mode in ["Display Only", "Both"]:
                            with st.expander(f"Match {i+1} - {match_info['Game Type']}"):
                                st.write(f"**Duration**: {match_info['Duration']}")
                                st.write(f"**Patch**: {match_info['Patch']}")
                                st.write(f"**Date**: {match_info['Date']}")
                                
                                for team, participants in match_info["Teams"].items():
                                    team_win_status = "Win" if next(iter(participants.values()))['win'] else "Lose"
                                    st.write(f"**{team} - {team_win_status}**")
                                    for summoner, details in participants.items():
                                        st.write(f"- **{details['summoner_name']}** ({details['champ_name']}) - "
                                                 f"KDA: {details['kills']}/{details['deaths']}/{details['assists']}, "
                                                 f"Position: {details['position']}")

                        if population_mode in ["Add to Database", "Both"]:
                            existing_match = session.query(Match).filter_by(game_id=match_info['Game ID']).first()
                            if existing_match:
                                st.write(f"Match {match_info['Game ID']} already exists in database. Skipping...")
                            else:
                                save_match_data(user_data, [match_id], session)
                                st.write(f"Added match {match_info['Game ID']} to database.")

                        progress_bar.progress((i + 1) / len(match_ids))
                        time.sleep(1.2)
                    else:
                        st.error(f"Match {i+1} information could not be retrieved.")

                st.success(f"Processed {len(match_ids)} matches successfully!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                session.close()

        else:
            st.error("No recent matches found.")