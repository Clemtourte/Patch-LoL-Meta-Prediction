import streamlit as st
from main import UserData, UserDisplay, save_match_data, Session
import logging
import datetime
import os
from perf_rating import analyze_champion_performance
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import time

logging.basicConfig(level=logging.INFO)

CHAMPION_IMAGE_PATH = "../datasets/champion/"

def load_champion_image(champion_name):
    image_path = os.path.join(CHAMPION_IMAGE_PATH, f"{champion_name}.png")
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        return None

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
                if start_date is None or end_date is None:
                    st.error("Please select both start and end dates")
                else:
                    start_time = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())
                    end_time = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp())
                    
                    st.info(f"Debug Info:")
                    st.write(f"Start Date: {start_date}")
                    st.write(f"End Date: {end_date}")
                    st.write(f"Start Timestamp: {start_time}")
                    st.write(f"End Timestamp: {end_time}")
                    st.write(f"Queue ID: {queue_id}")
                    st.write(f"Region: {region}")
                    st.write(f"Username: {username}")
                    st.write(f"Tag: {tag}")
                    
                    with st.spinner('Fetching matches... This may take a while due to API rate limits.'):
                        puuid = user_data.puuid
                        st.write(f"PUUID: {puuid}")
                        
                        api_url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue={queue_id}&startTime={start_time}&endTime={end_time}&start=0&count=100"
                        st.write(f"API URL being called: {api_url}")
                        
                        match_ids = user_data.get_matches_by_date(queue_id, start_time, end_time)
                        
                    st.write(f"Total matches found in date range: {len(match_ids)}")
                    if len(match_ids) == 0:
                        st.warning("No matches found. Debug information:")
                        st.write("1. Check if dates are correct")
                        st.write("2. Verify queue type is correct")
                        st.write("3. Confirm account has games in this period")
                        st.write("4. Verify region is correct")
                        st.write("5. Confirm summoner name and tag are correct")
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
                    for champ, champ_stats in top_5_champs:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            champ_image = load_champion_image(champ)
                            if champ_image:
                                st.image(champ_image, width=50)
                            else:
                                st.write("(No image)")
                        with col2:
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
                
                start_time = time.time()

                if population_mode in ["Add to Database", "Both"]:
                    with st.spinner('Processing matches in batch...'):
                        try:
                            added_matches, updated_matches, total_api_calls = save_match_data(user_data, match_ids, session)
                            st.success(f"Batch processing complete. Added: {added_matches}, Updated: {updated_matches}")
                            progress_bar.progress(1.0)
                        except Exception as e:
                            st.error(f"Error during batch processing: {str(e)}")
                            logging.error(f"Batch processing error: {str(e)}", exc_info=True)
                
                if population_mode in ["Display Only", "Both"]:
                    for i, match_id in enumerate(match_ids):
                        display_info = user_display.display_match_info(match_id)
                        if display_info:
                            with st.expander(f"{selected_mode} Match {i+1} (API calls: {display_info['API Calls']})"):
                                st.write(f"**Duration**: {display_info['Duration']}")
                                st.write(f"**Date**: {display_info['Date']}")
                                
                                for team, participants in display_info['Teams'].items():
                                    team_win_status = "Win" if next(iter(participants.values()))['win'] else "Lose"
                                    st.write(f"**{team} - {team_win_status}**")
                                    for summoner, details in participants.items():
                                        col1, col2 = st.columns([1, 4])
                                        with col1:
                                            champ_image = load_champion_image(details['champ_name'])
                                            if champ_image:
                                                st.image(champ_image, width=50)
                                            else:
                                                st.write("(No image)")
                                        with col2:
                                            st.write(f"**{summoner}** ({details['champ_name']}) - "
                                                    f"KDA: {details['kills']}/{details['deaths']}/{details['assists']} ({details['kda']}), "
                                                    f"Role: {details['position']}, "
                                                    f"{'CS: ' + str(details['cs']) if selected_mode != 'ARAM' else 'Damage: ' + str(details['total_damage_dealt'])}")
                        else:
                            st.error(f"Match {i+1} information could not be retrieved.")
                        
                        if population_mode == "Display Only":
                            progress_bar.progress((i + 1) / len(match_ids))

                end_time = time.time()
                processing_time = end_time - start_time

                st.success(f"Successfully processed {len(match_ids)} {selected_mode} matches.")
                st.info(f"Total API calls made: {total_api_calls}")
                st.info(f"Total processing time: {processing_time:.2f} seconds")
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

st.header("Champion Performance Analysis")

if st.button("Analyze Champion Performances"):
    session = Session()
    try:
        analysis_df = analyze_champion_performance(session)
        
        if analysis_df.empty:
            st.warning("Not enough data to perform analysis. Make sure you have processed games for multiple champions.")
        else:
            positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
            
            for position in positions:
                st.subheader(f"Champion Performance - {position}")
                position_df = analysis_df[analysis_df['position'] == position].sort_values('rating', ascending=False)
                
                cols = st.columns(5)
                for i, (_, row) in enumerate(position_df.iterrows()):
                    with cols[i % 5]:
                        champion_name = row['champion_name']
                        games = row['games']
                        win_rate = row['win_rate']
                        rating = row['rating']
                        
                        icon = load_champion_image(champion_name)
                        st.image(icon, width=50)
                        
                        st.write(f"**{champion_name}**")
                        st.write(f"Games: {games}")
                        st.write(f"WR: {win_rate:.0f}%")
                        st.write(f"Rating: {rating:.2f}")
                        st.write("---")
                st.write("") 
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        logging.error(f"Error in champion performance analysis: {str(e)}", exc_info=True)
    finally:
        session.close()