import streamlit as st
from main import UserData, UserDisplay, save_match_data, Session
import logging
import datetime
import os
from perf_rating import analyze_champion_performance, calculate_performance_ratings
from PIL import Image
import time
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAMPION_IMAGE_PATH = "../datasets/champion/"

def load_champion_image(champion_name):
    image_path = os.path.join(CHAMPION_IMAGE_PATH, f"{champion_name}.png")
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

def process_matches_safely(user_data, match_ids, progress_bar=None):
    """Process matches with proper error handling and transaction management"""
    session = Session()
    try:
        # Process matches
        added_matches, updated_matches, total_api_calls = save_match_data(
            user_data, match_ids, session
        )
        
        # Only calculate ratings if we added new matches
        if added_matches > 0:
            logger.info("New matches added, calculating performance ratings...")
            try:
                # Create a new session for ratings calculation
                rating_session = Session()
                calculate_performance_ratings()
                rating_session.commit()
            except Exception as e:
                logger.error(f"Error calculating ratings: {e}")
                # Continue anyway as this is not critical
            finally:
                rating_session.close()
            
        return added_matches, updated_matches, total_api_calls
    except Exception as e:
        logger.error(f"Error in process_matches_safely: {str(e)}")
        session.rollback()
        raise
    finally:
        if progress_bar:
            progress_bar.progress(1.0)
        session.close()

def main():
    st.title("League of Legends Meta Prediction")

    # User Input Section
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
    
    # Search Options
    search_option = st.radio("Choose search option:", ["Date Range", "Number of Recent Games"])
    if search_option == "Date Range":
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")
        number_of_games = None
    else:
        start_date = end_date = None
        number_of_games = st.number_input(
            "Number of recent games to process:",
            min_value=1,
            max_value=100,
            value=10
        )

    st.header("Data Processing Options")
    population_mode = st.radio(
        "Choose processing mode:",
        ["Display Only", "Add to Database", "Both"]
    )

    if st.button("Process Games"):
        if not username or not tag:
            st.error("Please enter both username and tag.")
            return

        progress_bar = st.progress(0)
        user_data = None
        session = None

        try:
            user_data = UserData(username, tag, region)
            user_display = UserDisplay(user_data)
            queue_id = game_modes[selected_mode]

            # Fetch match IDs
            with st.spinner('Fetching matches...'):
                if search_option == "Date Range":
                    if not all([start_date, end_date]):
                        st.error("Please select both start and end dates")
                        return
                        
                    start_time = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())
                    end_time = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp())
                    match_ids = user_data.get_matches_by_date(queue_id, start_time, end_time)
                else:
                    match_ids = user_data.get_matches(queue_id, number_of_games)

            if not match_ids:
                st.warning("No matches found for the specified criteria.")
                return

            st.info(f"Found {len(match_ids)} matches to process")

            # Process matches based on selected mode
            if population_mode in ["Add to Database", "Both"]:
                with st.spinner('Processing matches...'):
                    try:
                        added, updated, api_calls = process_matches_safely(
                            user_data, match_ids, progress_bar
                        )
                        st.success(
                            f"Successfully processed {added} new matches "
                            f"(updated {updated}, API calls: {api_calls})"
                        )
                    except Exception as e:
                        st.error(f"Error processing matches: {str(e)}")
                        return

            # Display match information if requested
            if population_mode in ["Display Only", "Both"]:
                session = Session()
                try:
                    display_match_information(
                        user_data, user_display, match_ids, selected_mode, progress_bar
                    )
                finally:
                    if session:
                        session.close()

            # Display player statistics
            display_player_statistics(user_data, user_display, selected_mode, queue_id)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main process: {str(e)}", exc_info=True)
        finally:
            progress_bar.progress(1.0)

    # Champion Performance Analysis Section
    st.header("Champion Performance Analysis")
    if st.button("Analyze Champion Performances"):
        perform_champion_analysis()

def display_match_information(user_data, user_display, match_ids, selected_mode, progress_bar):
    """Display detailed match information"""
    for i, match_id in enumerate(match_ids):
        display_info = user_display.display_match_info(match_id)
        if not display_info:
            st.error(f"Could not retrieve information for match {match_id}")
            continue

        with st.expander(f"{selected_mode} Match {i+1}"):
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
                    
                    with col2:
                        display_participant_details(summoner, details, selected_mode)

        if progress_bar:
            progress_bar.progress((i + 1) / len(match_ids))

def display_participant_details(summoner, details, selected_mode):
    """Display participant details in a consistent format"""
    st.write(
        f"**{summoner}** ({details['champ_name']}) - "
        f"KDA: {details['kills']}/{details['deaths']}/{details['assists']} ({details['kda']}), "
        f"Role: {details['position']}, "
        f"{'CS: ' + str(details['cs']) if selected_mode != 'ARAM' else 'Damage: ' + str(details['total_damage_dealt'])}"
    )

def display_player_statistics(user_data, user_display, selected_mode, queue_id):
    """Display player statistics section"""
    session = Session()
    try:
        stats = user_data.get_player_statistics(session, queue_id)
        formatted_stats = user_display.format_player_statistics(stats)

        if formatted_stats['total_games'] > 0:
            display_basic_stats(formatted_stats)
            display_champion_stats(formatted_stats)
        else:
            st.warning(f"No statistics found for {selected_mode} mode.")
    finally:
        session.close()

def display_basic_stats(stats):
    """Display basic player statistics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Total Games**: {stats['total_games']}")
        st.write(f"**Wins/Losses**: {stats['wins']}/{stats['losses']}")
    with col2:
        st.write(f"**Win Rate**: {stats['win_rate']:.2f}%")
        st.write(f"**KDA**: {stats['kda_ratio']:.2f}")
    with col3:
        st.write(f"**Avg CS**: {stats['avg_cs']:.1f}")
        st.write(f"**Most Played**: {stats['most_played_roles']}")

def display_champion_stats(stats):
    """Display champion-specific statistics"""
    st.subheader("Champion Statistics")
    if not stats.get('champion_stats'):
        st.warning("No champion statistics available.")
        return

    sorted_champs = sorted(
        stats['champion_stats'].items(),
        key=lambda x: x[1]['games'],
        reverse=True
    )[:5]

    for champ, champ_stats in sorted_champs:
        col1, col2 = st.columns([1, 4])
        with col1:
            champ_image = load_champion_image(champ)
            if champ_image:
                st.image(champ_image, width=50)
        with col2:
            st.write(f"**{champ}**")
            st.write(
                f"Games: {champ_stats['games']} | "
                f"WR: {champ_stats['winrate']:.1f}% | "
                f"KDA: {champ_stats['kda_ratio']:.2f} | "
                f"CS: {champ_stats['avg_cs']:.1f}"
            )

def perform_champion_analysis():
    """Perform and display champion performance analysis"""
    session = Session()
    try:
        analysis_df = analyze_champion_performance(session)
        if analysis_df.empty:
            st.warning("Insufficient data for analysis.")
            return

        positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        for position in positions:
            st.subheader(f"Champion Performance - {position}")
            position_df = analysis_df[
                analysis_df['position'] == position
            ].sort_values('rating', ascending=False)

            display_position_analysis(position_df)

    except Exception as e:
        st.error(f"Error in champion analysis: {str(e)}")
        logger.error(f"Champion analysis error: {str(e)}", exc_info=True)
    finally:
        session.close()

def display_position_analysis(position_df):
    """Display analysis for a specific position"""
    cols = st.columns(5)
    for i, (_, row) in enumerate(position_df.iterrows()):
        with cols[i % 5]:
            champion_name = row['champion_name']
            icon = load_champion_image(champion_name)
            if icon:
                st.image(icon, width=50)
            
            st.write(
                f"**{champion_name}**\n"
                f"Games: {row['games']}\n"
                f"WR: {row['win_rate']:.0f}%\n"
                f"Rating: {row['rating']:.2f}"
            )
            st.write("---")

if __name__ == "__main__":
    main()