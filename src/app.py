import streamlit as st
from main import UserData, UserDisplay, save_match_data, Session
import logging

logging.basicConfig(level=logging.INFO)

st.title("League of Legends User Information")

region = st.selectbox("Select your region:", ["EUW1"])  # Add more regions as needed
username = st.text_input("Enter your username:")
tag = st.text_input("Enter your tag (without #):")

game_modes = {
    "Ranked Solo/Duo": "420",
    "Ranked Flex": "440",
    "Normal Draft": "400",
    "Normal Blind": "430",
    "ARAM": "450"
}
selected_mode = st.selectbox("Select game mode:", list(game_modes.keys()))

number_of_games = st.number_input("Number of recent games to process:", min_value=1, max_value=100, value=10)

st.header("Database Population")
population_mode = st.radio("Choose population mode:", ["Display Only", "Add to Database", "Both"])

if st.button("Process Games"):
    if not username or not tag:
        st.error("Please enter both username and tag.")
    else:
        user_data = UserData(username, tag, region)
        user_display = UserDisplay(user_data)
        session = Session()  # Create a new session here

        try:
            queue_id = game_modes[selected_mode]
            
            st.subheader("Player Statistics")
            st.text(f"Retrieving statistics for {selected_mode}")

            raw_stats = user_data.get_player_statistics(session, queue_id)
            formatted_stats = user_display.format_player_statistics(raw_stats)

            if formatted_stats['total_games'] > 0:
                st.success(f"Found {formatted_stats['total_games']} games for {selected_mode}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Total Games**: {formatted_stats['total_games']}")
                    st.write(f"**Total Wins**: {formatted_stats['wins']}")
                    st.write(f"**Total Losses**: {formatted_stats['losses']}")
                with col2:
                    st.write(f"**Overall Winrate**: {formatted_stats['win_rate']:.2f}%")
                    st.write(f"**Average KDA**: {formatted_stats['kda_ratio']:.2f}")
                    st.write(f"**Average CS**: {formatted_stats['avg_cs']:.1f}")
                with col3:
                    st.write(f"**Most Played Role**: {formatted_stats['most_played_roles']}")

                st.subheader("Champion Statistics")
                sorted_champs = sorted(formatted_stats['champion_stats'].items(), key=lambda x: x[1]['games'], reverse=True)
                top_5_champs = sorted_champs[:5]
                
                if top_5_champs:
                    champ_cols = st.columns(len(top_5_champs))
                    for i, (champ, champ_stats) in enumerate(top_5_champs):
                        with champ_cols[i]:
                            st.write(f"**{champ}**")
                            st.write(f"Games: {champ_stats['games']}")
                            st.write(f"Winrate: {champ_stats['winrate']:.2f}%")
                            st.write(f"KDA: {champ_stats['kda_ratio']:.2f}")
                            st.write(f"Avg CS: {champ_stats['avg_cs']:.1f}")
                else:
                    st.warning("No champion statistics available.")
            else:
                st.warning(f"No statistics found for {selected_mode} mode. Make sure you have played games in this mode and they are saved in the database.")

            # Process matches
            match_ids = user_data.get_matches(queue_id, number_of_games)

            if match_ids:
                st.subheader(f"Processing {len(match_ids)} Recent {selected_mode} Matches")
                progress_bar = st.progress(0)
                
                for i, match_id in enumerate(match_ids[:number_of_games]):
                    match_info = user_data.get_match_info(match_id)

                    if match_info[0] is not None:
                        general_info, participant_info, team_stats = match_info
                        if population_mode in ["Display Only", "Both"]:
                            with st.expander(f"{selected_mode} Match {i+1}"):
                                st.write(f"**Duration**: {general_info['game_duration']}")
                                st.write(f"**Date**: {general_info['timestamp']}")
                                
                                for team, participants in participant_info.items():
                                    team_win_status = "Win" if next(iter(participants.values()))['win'] else "Lose"
                                    st.write(f"**{team} - {team_win_status}**")
                                    for summoner, details in participants.items():
                                        st.write(f"- **{details['summoner_name']}** ({details['champ_name']}) - "
                                                f"KDA: {details['kills']}/{details['deaths']}/{details['assists']} ({details['kda']:.2f}), "
                                                f"Role: {details['position']}, "
                                                f"{'CS: ' + str(details['cs']) if selected_mode != 'ARAM' else 'Damage: ' + str(details['total_damage_dealt'])}")

                        if population_mode in ["Add to Database", "Both"]:
                            save_match_data(user_data, [match_id], session)

                        progress_bar.progress((i + 1) / number_of_games)
                    else:
                        st.error(f"Match {i+1} information could not be retrieved.")

                st.success(f"Successfully processed {len(match_ids)} {selected_mode} matches.")
            else:
                st.warning(f"No recent {selected_mode} matches found.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error processing games: {str(e)}", exc_info=True)
        finally:
            session.close()  # Make sure to close the session