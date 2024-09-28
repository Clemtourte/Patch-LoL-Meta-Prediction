import streamlit as st
from main import UserData, UserDisplay, save_match_data, Session
import logging
import datetime

logging.basicConfig(level=logging.INFO)

st.title("League of Legends Meta Prediction")

region = st.selectbox("Select your region:", ["EUW1", "NA1", "KR", "JP1"])
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
search_option = st.radio("Choose search option:", ["Date Range", "Number of Recent Games"])
start_date = end_date = None
number_of_games = 0

if search_option == "Date Range":
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")
else:
    number_of_games = st.number_input("Number of recent games to process:", min_value=1, max_value=100, value=10)

st.header("Data Processing Options")
population_mode = st.radio("Choose processing mode:", ["Display Only", "Add to Database", "Both"])

if st.button("Process Games"):
    if not username or not tag:
        st.error("Please enter both username and tag.")
    else:
        user_data = UserData(username, tag, region)
        user_display = UserDisplay(user_data)
        session = Session()

        try:
            queue_id = game_modes[selected_mode]
            
            if search_option == "Date Range":
                start_time = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())
                end_time = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp())
                with st.spinner('Fetching matches... This may take a while due to API rate limits.'):
                    match_ids = user_data.get_matches_by_date(queue_id, start_time, end_time)
                st.write(f"Total matches found in date range: {len(match_ids)}")
            else:
                with st.spinner('Fetching recent matches...'):
                    match_ids = user_data.get_recent_matches(queue_id, number_of_games)
                st.write(f"Fetched {len(match_ids)} recent matches")
            
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

            if match_ids:
                st.subheader(f"Processing {len(match_ids)} {selected_mode} Matches")
                progress_bar = st.progress(0)
                total_api_calls = 0
                
                for i, match_id in enumerate(match_ids):
                    match_info = user_data.get_match_info(match_id)

                    if match_info[0] is not None:
                        general_info, participant_info, team_stats, api_calls = match_info
                        total_api_calls += api_calls
                        
                        if population_mode in ["Display Only", "Both"]:
                            display_info = user_display.display_match_info(match_id)
                            with st.expander(f"{selected_mode} Match {i+1} (API calls: {display_info['API Calls']})"):
                                st.write(f"**Duration**: {display_info['Duration']}")
                                st.write(f"**Date**: {display_info['Date']}")
                                
                                for team, participants in display_info['Teams'].items():
                                    team_win_status = "Win" if next(iter(participants.values()))['win'] else "Lose"
                                    st.write(f"**{team} - {team_win_status}**")
                                    for summoner, details in participants.items():
                                        st.write(f"- **{details['summoner_name']}** ({details['champ_name']}) - "
                                                f"KDA: {details['kills']}/{details['deaths']}/{details['assists']} ({details['kda']:.2f}), "
                                                f"Role: {details['position']}, "
                                                f"{'CS: ' + str(details['cs']) if selected_mode != 'ARAM' else 'Damage: ' + str(details['total_damage_dealt'])}")

                        if population_mode in ["Add to Database", "Both"]:
                            save_match_data(user_data, [match_id], session)

                        progress_bar.progress((i + 1) / len(match_ids))
                    else:
                        st.error(f"Match {i+1} information could not be retrieved.")

                st.success(f"Successfully processed {len(match_ids)} {selected_mode} matches.")
                st.info(f"Total API calls made: {total_api_calls}")
            else:
                if search_option == "Date Range":
                    st.warning(f"No {selected_mode} matches found in the specified date range.")
                else:
                    st.warning(f"No recent {selected_mode} matches found.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error processing games: {str(e)}", exc_info=True)
        finally:
            session.close()