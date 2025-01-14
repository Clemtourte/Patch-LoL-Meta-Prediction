import requests
import os
import time
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import SQLAlchemyError
from models import Match, Team, Participant, PerformanceFeatures, init_db
from perf_rating import calculate_performance_ratings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            elif response.status_code == 429: 
                retry_after = int(response.headers.get('Retry-After', 120))
                self.logger.warning(f"Rate limit exceeded. Retry after {retry_after} seconds")
                time.sleep(retry_after + 1)
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

            for participant in match_data['info']['participants']:
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'
                summoner_name = participant['riotIdGameName'] if participant['summonerName'] == "" else participant['summonerName']
                
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

                for stat, value in participant_info[team][summoner_name].items():
                    if stat in ['kills', 'deaths', 'assists', 'total_damage_dealt', 'gold_earned', 
                              'cs', 'wards_placed', 'wards_killed', 'xp', 'damage_taken', 
                              'total_heal', 'damage_mitigated', 'time_ccing_others']:
                        team_stats[team][f'total_{stat.replace("total_", "")}'] = team_stats[team].get(f'total_{stat.replace("total_", "")}', 0) + value

            return general_info, participant_info, team_stats, self.api_call_count

        except Exception as e:
            self.logger.error(f"Error processing match {match_id}: {str(e)}")
            return None, None, None, self.api_call_count

    def calculate_performance_features(self, participant_info, team_stats, summoner_name, patch):  
        """Calculate performance features for a participant"""
        user_team = next((team for team, players in participant_info.items() if summoner_name in players), None)
        if not user_team:
            return None

        user_stats = participant_info[user_team][summoner_name]
        team_totals = team_stats[user_team]
        
        # Debug prints
        print(f"Champion: {user_stats['champion_name']}")
        print(f"Position: {user_stats['position']}")
        print(f"Patch: {patch}")
        
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
            position = user_stats['position'].upper() if user_stats['position'] != 'N/A' else 'UNKNOWN'
            features['champion_role_patch'] = f"{user_stats['champion_name']}-{position}-{patch}"
            
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
                        
                        total_kills += participant.kills
                        total_deaths += participant.deaths
                        total_assists += participant.assists
                        total_cs += participant.cs

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
    """
    Save match data to database with proper relationship handling
    """
    added_matches = 0
    updated_matches = 0
    total_api_calls = 0
    
    for match_id in match_ids:
        try:
            logger.info(f"Processing match {match_id}")
            
            # Get match info
            match_info = user_data.get_match_info(match_id)
            if not all(match_info[:3]):
                logger.warning(f"Skipping match {match_id}: Incomplete data")
                continue

            general_info, participant_info, team_stats, api_calls = match_info
            total_api_calls += api_calls

            # Check for existing match
            existing_match = session.query(Match).filter_by(game_id=general_info['game_id']).first()
            if existing_match:
                logger.info(f"Match {general_info['game_id']} already exists")
                updated_matches += 1
                continue

            # Create new match
            new_match = Match(
                game_id=general_info['game_id'],
                game_duration=general_info['game_duration'],
                patch=general_info['patch'],
                timestamp=general_info['timestamp'],
                mode=general_info['mode'],
                queue_id=general_info['queue_id'],
                platform="EUW1"
            )
            
            # Create teams and add them to match
            for team_name, participants in participant_info.items():
                if not participants:
                    continue

                # Create team
                team = Team(
                    team_name=team_name,
                    win=next(iter(participants.values()))['win']
                )
                
                # Add team to match
                new_match.teams.append(team)

                # Process participants
                for summoner, details in participants.items():
                    # Create participant
                    participant = Participant(
                        summoner_id=details['summoner_id'],
                        summoner_name=details['summoner_name'],
                        champion_name=details['champion_name'],
                        champion_id=details['champion_id'],
                        position=details['position'],
                        role=details['role'],
                        lane=details['lane'],
                        champ_level=details['champ_level'],
                        kills=details['kills'],
                        deaths=details['deaths'],
                        assists=details['assists'],
                        kda=details['kda'],
                        gold_earned=details['gold_earned'],
                        total_damage_dealt=details['total_damage_dealt'],
                        cs=details['cs'],
                        total_heal=details['total_heal'],
                        damage_taken=details['damage_taken'],
                        damage_mitigated=details['damage_mitigated'],
                        wards_placed=details['wards_placed'],
                        wards_killed=details['wards_killed'],
                        time_ccing_others=details['time_ccing_others'],
                        xp=details['xp']
                    )
                    
                    # Add participant to team
                    team.participants.append(participant)

                    # Calculate and add performance features
                    features = user_data.calculate_performance_features(
                        participant_info, team_stats, summoner, general_info['patch'] 
                    )
                    if features:
                        perf_features = PerformanceFeatures(
                            **{k: v for k, v in features.items() 
                               if hasattr(PerformanceFeatures, k)}
                        )
                        print(f"About to save performance features:")
                        print(f"Champion role patch: {perf_features.champion_role_patch}")
                        participant.performance_features = perf_features

            # Add match to session
            session.add(new_match)
            session.commit()
            
            added_matches += 1
            logger.info(f"Successfully processed match {general_info['game_id']}")

        except Exception as e:
            logger.error(f"Error processing match {match_id}: {str(e)}")
            logger.error(traceback.format_exc())  # Add this line for better error tracking
            session.rollback()
            continue

    return added_matches, updated_matches, total_api_calls