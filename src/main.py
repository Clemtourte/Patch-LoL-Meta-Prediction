import requests
import os
import time
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy.orm.exc import NoResultFound
from models import Match, Team, Participant, PerformanceFeatures, init_db
from perf_rating import calculate_performance_ratings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize database session
Session = init_db()

class UserData:
    def __init__(self, username, tag, region):
        load_dotenv()
        self.API_key = os.getenv('API_KEY')
        self.username = username
        self.tag = tag
        self.region = region
        self.api_call_count = 0
        self.puuid = self.get_puuid()
        self.logger = logging.getLogger(__name__)

    def reset_api_call_count(self):
        self.api_call_count = 0

    def get_safe_summoner_name(self, participant):
        """Get a safe summoner name from participant data"""
        name = participant.get('summonerName', '')
        if not name:
            name = participant.get('riotIdGameName', '')
        if not name:
            puuid = participant.get('puuid', '')
            summoner_id = participant.get('summonerId', '')
            name = f"Unknown-{puuid[:8] if puuid else summoner_id}"
        return name

    def _get_response(self, url, default=None):
        """Make an API request and handle the response with better rate limit handling"""
        self.api_call_count += 1
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 120))
                self.logger.warning(f"Rate limit exceeded. Retry after {retry_after} seconds")
                time.sleep(retry_after + 1)  # Add 1 second buffer
                # Retry the request
                return self._get_response(url, default)
            else:
                self.logger.error(f"API request failed with status code {response.status_code}. Response: {response.text}")
                return default
        except Exception as e:
            self.logger.error(f"Error in API request: {str(e)}")
            return default

    def get_puuid(self):
        """Get PUUID for the user"""
        url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{self.username}/{self.tag}?api_key={self.API_key}"
        response = self._get_response(url)
        return response.get('puuid')

    def get_summoner_id(self):
        """Get Summoner ID from PUUID"""
        url = f"https://{self.region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{self.puuid}?api_key={self.API_key}"
        response = self._get_response(url)
        return response.get('id')

    def get_ranked_data(self, summoner_id):
        """Get ranked data for a summoner"""
        url = f"https://{self.region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}?api_key={self.API_key}"
        return self._get_response(url, default=[])

    def get_matches(self, queue_id, match_count):
        """Get recent matches for user"""
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?queue={queue_id}&start=0&count={match_count}&api_key={self.API_key}"
        return self._get_response(url, default=[])

    def get_match_data(self, match_id):
        """Get detailed match data"""
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}"
        return self._get_response(url, default={})

    def get_matches_by_date(self, queue_id, start_time, end_time):
        """Get matches within a date range"""
        matches = []
        start_index = 0
        self.logger.info(f"Searching matches for PUUID: {self.puuid}, Queue: {queue_id}")
        
        while True:
            url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids"
            url += f"?queue={queue_id}&startTime={start_time}&endTime={end_time}&start={start_index}&count=100&api_key={self.API_key}"
            
            batch = self._get_response(url, default=[])
            if not batch:
                break
                
            matches.extend(batch)
            if len(batch) < 100:
                break
            start_index += 100
        
        self.logger.info(f"Found {len(matches)} matches")
        return matches

    def get_match_info(self, match_id):
        """Process and extract match information"""
        self.reset_api_call_count()
        match_data = self.get_match_data(match_id)
        if not match_data:
            return None, None, None, self.api_call_count

        try:
            # Extract basic match info
            game_id = match_data['metadata']['matchId']
            game_duration = match_data['info']['gameDuration']
            game_version = match_data['info']['gameVersion']
            patch = '.'.join(game_version.split('.')[:2])
            minutes, seconds = divmod(game_duration, 60)
            
            general_info = {
                'game_id': game_id,
                'game_duration': f"{minutes}m {seconds}s",
                'patch': patch,
                'timestamp': datetime.fromtimestamp(match_data['info']['gameCreation'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'mode': match_data['info']['gameMode'],
                'queue_id': match_data['info']['queueId']
            }

            # Initialize team stats
            participant_info = {'Blue Side': {}, 'Red Side': {}}
            team_stats = {'Blue Side': {}, 'Red Side': {}}
            for team in team_stats:
                team_stats[team] = {
                    'total_kills': 0, 'total_deaths': 0, 'total_assists': 0,
                    'total_damage_dealt': 0, 'total_gold_earned': 0, 'total_cs': 0,
                    'wards_placed': 0, 'wards_killed': 0, 'total_xp': 0,
                    'damage_taken': 0, 'heal': 0, 'damage_mitigated': 0,
                    'time_ccing_others': 0
                }

            # Process participant data
            for participant in match_data['info']['participants']:
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'
                summoner_name = participant['riotIdGameName'] if participant['summonerName'] == "" else participant['summonerName']
                
                # Calculate participant stats
                participant_info[team][summoner_name] = {
                    'summoner_id': participant['summonerId'],
                    'summoner_name': summoner_name,
                    'champion_name': participant['championName'],
                    'champion_id': participant['championId'],
                    'position': participant.get('teamPosition', 'N/A'),
                    'role': participant.get('role', 'N/A'),
                    'lane': participant.get('lane', 'N/A'),
                    'champ_level': participant['champLevel'],
                    'kills': participant['kills'],
                    'deaths': participant['deaths'],
                    'assists': participant['assists'],
                    'kda': (participant['kills'] + participant['assists']) / max(1, participant['deaths']),
                    'gold_earned': participant['goldEarned'],
                    'total_damage_dealt': participant['totalDamageDealtToChampions'],
                    'cs': participant['totalMinionsKilled'] + participant['neutralMinionsKilled'],
                    'total_heal': participant['totalHeal'],
                    'damage_taken': participant['totalDamageTaken'],
                    'damage_mitigated': participant['damageSelfMitigated'],
                    'wards_placed': participant['wardsPlaced'],
                    'wards_killed': participant['wardsKilled'],
                    'time_ccing_others': participant['timeCCingOthers'],
                    'xp': participant['champExperience'],
                    'win': participant['win']
                }

                # Update team totals
                for stat, value in participant_info[team][summoner_name].items():
                    if stat in ['kills', 'deaths', 'assists', 'total_damage_dealt', 'gold_earned', 
                              'cs', 'wards_placed', 'wards_killed', 'xp', 'damage_taken', 
                              'total_heal', 'damage_mitigated', 'time_ccing_others']:
                        team_stats[team][f'total_{stat.replace("total_", "")}'] = team_stats[team].get(f'total_{stat.replace("total_", "")}', 0) + value

            return general_info, participant_info, team_stats, self.api_call_count

        except Exception as e:
            self.logger.error(f"Error processing match {match_id}: {str(e)}")
            return None, None, None, self.api_call_count

    def calculate_performance_features(self, participant_info, team_stats, summoner_name):
        """Calculate performance features for a participant"""
        user_team = next((team for team, players in participant_info.items() if summoner_name in players), None)
        if not user_team:
            return None

        user_stats = participant_info[user_team][summoner_name]
        team_totals = team_stats[user_team]
        
        def safe_divide(a, b, min_denominator=1):
            return a / max(b, min_denominator)
        
        try:
            features = {
                'kill_participation': safe_divide(user_stats['kills'] + user_stats['assists'], team_totals['total_kills'], 5),
                'death_share': safe_divide(user_stats['deaths'], team_totals['total_deaths'], 5),
                'damage_share': safe_divide(user_stats['total_damage_dealt'], team_totals['total_damage_dealt'], 100),
                'damage_taken_share': safe_divide(user_stats['damage_taken'], team_totals['damage_taken'], 100),
                'gold_share': safe_divide(user_stats['gold_earned'], team_totals['total_gold_earned'], 100),
                'heal_share': safe_divide(user_stats['total_heal'], team_totals['heal'], 100),
                'damage_mitigated_share': safe_divide(user_stats['damage_mitigated'], team_totals['damage_mitigated'], 100),
                'cs_share': safe_divide(user_stats['cs'], team_totals['total_cs'], 100),
                'vision_share': safe_divide(user_stats['wards_placed'], team_totals['wards_placed'], 5),
                'vision_denial_share': safe_divide(user_stats['wards_killed'], team_totals['wards_killed'], 5),
                'xp_share': safe_divide(user_stats['xp'], team_totals['total_xp'], 100),
                'cc_share': safe_divide(user_stats['time_ccing_others'], team_totals['time_ccing_others'], 10)
            }
            
            features = {k: min(1.0, max(0.0, v)) for k, v in features.items()}
            features['champion_role_patch'] = f"{user_stats['champion_name']}-{user_stats['position']}"
            
            return features
                
        except Exception as e:
            self.logger.error(f"Error calculating features for {summoner_name}: {str(e)}")
            return None
    
    def get_player_statistics(self, session, queue_id):
        """Get comprehensive player statistics"""
        matches = session.query(Match).join(Team).join(Participant).filter(
            Participant.summoner_name == self.username,
            Match.queue_id == queue_id
        ).all()
        
        if not matches:
            matches = session.query(Match).join(Team).join(Participant).filter(
                Participant.summoner_name == self.username
            ).all()

        total_games = len(matches)
        wins = 0
        champion_stats = {}
        total_kills = total_deaths = total_assists = total_cs = 0
        roles_played = {}

        for match in matches:
            for team in match.teams:
                for participant in team.participants:
                    if participant.summoner_name == self.username:
                        if team.win:
                            wins += 1
                        
                        # Update champion stats
                        if participant.champion_name not in champion_stats:
                            champion_stats[participant.champion_name] = {
                                'games': 0, 'wins': 0, 'kills': 0, 
                                'deaths': 0, 'assists': 0, 'cs': 0
                            }
                        
                        stats = champion_stats[participant.champion_name]
                        stats['games'] += 1
                        if team.win:
                            stats['wins'] += 1
                        stats['kills'] += participant.kills
                        stats['deaths'] += participant.deaths
                        stats['assists'] += participant.assists
                        stats['cs'] += participant.cs
                        
                        # Update totals
                        total_kills += participant.kills
                        total_deaths += participant.deaths
                        total_assists += participant.assists
                        total_cs += participant.cs
                        
                        # Track roles
                        roles_played[participant.position] = roles_played.get(participant.position, 0) + 1

        return {
            'total_games': total_games,
            'wins': wins,
            'champion_stats': champion_stats,
            'total_kills': total_kills,
            'total_deaths': total_deaths,
            'total_assists': total_assists,
            'total_cs': total_cs,
            'roles_played': roles_played
        }

class UserDisplay:
    def __init__(self, user_data):
        self.user_data = user_data

    def display_user_info(self):
        """Display user's ranked information"""
        summoner_id = self.user_data.get_summoner_id()
        if not summoner_id:
            return None

        ranked_data = self.user_data.get_ranked_data(summoner_id)
        info = {}
        
        for entry in ranked_data:
            queue_type = entry.get('queueType')
            if queue_type in ['RANKED_SOLO_5x5', 'RANKED_FLEX_SR']:
                key = 'ranked' if queue_type == 'RANKED_SOLO_5x5' else 'flex'
                info[key] = f"{entry['tier']} {entry['rank']} - {entry['leaguePoints']} LP"
                
        return info

    def display_match_info(self, match_id):
        """Format match information for display"""
        match_info = self.user_data.get_match_info(match_id)
        if match_info[0] is None:
            return None
        
        general_info, participant_info, _, api_calls = match_info
        return {
            "Game ID": general_info['game_id'],
            "Duration": general_info['game_duration'],
            "Patch": general_info['patch'],
            "Date": general_info['timestamp'],
            "Game Mode": general_info['mode'],
            "Queue ID": general_info['queue_id'],
            "Teams": participant_info,
            "API Calls": api_calls
        }
    
    def format_player_statistics(self, stats):
        """Format player statistics for display"""
        formatted_stats = {
            'total_games': stats['total_games'],
            'wins': stats['wins'],
            'losses': stats['total_games'] - stats['wins'],
            'win_rate': round(stats['wins'] / stats['total_games'] * 100, 2) if stats['total_games'] > 0 else 0,
            'kda_ratio': ((stats['total_kills'] + stats['total_assists']) / stats['total_deaths']) 
                        if stats['total_deaths'] > 0 else (stats['total_kills'] + stats['total_assists']),
            'avg_cs': stats['total_cs'] / stats['total_games'] if stats['total_games'] > 0 else 0,
            'most_played_roles': max(stats['roles_played'].items(), key=lambda x: x[1])[0] 
                                if stats['roles_played'] else 'N/A',
            'champion_stats': {}
        }
        
        for champ, champ_stats in stats['champion_stats'].items():
            formatted_stats['champion_stats'][champ] = {
                'games': champ_stats['games'],
                'wins': champ_stats['wins'],
                'winrate': champ_stats['wins'] / champ_stats['games'] * 100 if champ_stats['games'] > 0 else 0,
                'kda_ratio': ((champ_stats['kills'] + champ_stats['assists']) / champ_stats['deaths']) 
                            if champ_stats['deaths'] > 0 else (champ_stats['kills'] + champ_stats['assists']),
                'avg_cs': champ_stats['cs'] / champ_stats['games'] if champ_stats['games'] > 0 else 0
            }
        
        return formatted_stats

def save_match_data(user_data, match_ids, session):
    """Save match data to database with improved rate limiting"""
    added_matches = 0
    updated_matches = 0
    total_api_calls = 0
    start_time = time.time()
    retry_after = 0

    try:
        for i, match_id in enumerate(match_ids, 1):
            logger.info(f"Processing match {i} of {len(match_ids)}: {match_id}")
            
            # Handle rate limits
            elapsed_time = time.time() - start_time
            if total_api_calls >= 70:  # Near rate limit
                sleep_time = max(120 - elapsed_time + 5, 5)  # At least 5 seconds
                logger.info(f"Rate limit approaching. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                total_api_calls = 0
                start_time = time.time()
            
            # If we got a retry-after response, honor it
            if retry_after > 0:
                logger.info(f"Waiting {retry_after} seconds due to rate limit")
                time.sleep(retry_after)
                retry_after = 0

            # Get match info
            match_info = user_data.get_match_info(match_id)
            
            # Check for rate limit in response
            if match_info[0] is None and user_data.api_call_count == 0:
                retry_after = 120  # Default to 2 minutes if we hit rate limit
                continue

            if not all(match_info[:3]):  # Check if general_info, participant_info, and team_stats exist
                continue

            general_info, participant_info, team_stats, api_calls = match_info
            total_api_calls += api_calls

            # Skip remakes
            if int(general_info['game_duration'].split('m')[0]) < 14:
                logger.info(f"Skipping remake match {match_id}")
                continue

            # Check if match exists
            if session.query(Match).filter_by(game_id=general_info['game_id']).first():
                logger.info(f"Match {general_info['game_id']} already exists. Skipping.")
                continue

            # Create match
            match = Match(
                game_id=general_info['game_id'],
                game_duration=general_info['game_duration'],
                patch=general_info['patch'],
                timestamp=general_info['timestamp'],
                mode=general_info['mode'],
                queue_id=general_info['queue_id'],
                platform="EUW1"
            )
            session.add(match)
            session.flush()

            # Process teams
            for team_name, participants in participant_info.items():
                if not participants:
                    continue

                # Create team
                team = Team(
                    match_id=match.match_id,
                    team_name=team_name,
                    win=next(iter(participants.values()))['win']
                )
                session.add(team)
                session.flush()

                # Process participants
                for summoner, details in participants.items():
                    participant = Participant(team_id=team.team_id)
                    
                    # Set participant attributes
                    for attr, value in details.items():
                        if attr != 'team' and hasattr(participant, attr):
                            if isinstance(value, (int, float, bool, str)):
                                setattr(participant, attr, value)
                    
                    session.add(participant)
                    session.flush()

                    # Calculate and save performance features
                    features = user_data.calculate_performance_features(participant_info, team_stats, summoner)
                    if features:
                        perf_features = PerformanceFeatures(
                            participant_id=participant.participant_id,
                            champion_role_patch=f"{details['champion_name']}-{details['position']}-{general_info['patch']}"
                        )
                        
                        for key, value in features.items():
                            if hasattr(perf_features, key) and key != 'champion_role_patch':
                                setattr(perf_features, key, value)
                        
                        session.add(perf_features)

            try:
                session.commit()
                added_matches += 1
                logger.info(f"Successfully saved match {general_info['game_id']}")
            except Exception as e:
                logger.error(f"Error saving match {match_id}: {str(e)}")
                session.rollback()
                continue

            # Basic rate limiting between requests
            time.sleep(2.2)

        # Calculate performance ratings after all matches are saved
        if added_matches > 0:
            logger.info("Calculating performance ratings for new matches...")
            calculate_performance_ratings()
        
        return added_matches, updated_matches, total_api_calls

    except Exception as e:
        logger.error(f"Error in save_match_data: {str(e)}")
        session.rollback()
        return 0, 0, total_api_calls