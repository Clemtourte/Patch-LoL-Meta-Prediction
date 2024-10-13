import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from models import Match, Team, Participant, PerformanceFeatures, init_db
from perf_rating import calculate_performance_ratings
from sqlalchemy.orm.exc import NoResultFound
import time
import json
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        logging.basicConfig(level=logging.INFO)

    def reset_api_call_count(self):
        self.api_call_count = 0

    def get_safe_summoner_name(self, participant):
        name = participant.get('summonerName', '')
        if not name:
            name = participant.get('riotIdGameName', '')
        if not name:
            puuid = participant.get('puuid', '')
            summoner_id = participant.get('summonerId', '')
            name = f"Unknown-{puuid[:8] if puuid else summoner_id}"
        return name

    def get_puuid(self):
        url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{self.username}/{self.tag}?api_key={self.API_key}"
        response = self._get_response(url)
        return response.get('puuid')

    def get_summoner_id(self):
        url = f"https://{self.region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{self.puuid}?api_key={self.API_key}"
        response = self._get_response(url)
        return response.get('id')

    def get_ranked_data(self, summoner_id):
        url = f"https://{self.region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}?api_key={self.API_key}"
        return self._get_response(url, default=[])

    def get_matches(self, queue_id, match_count):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?queue={queue_id}&start=0&count={match_count}&api_key={self.API_key}"
        return self._get_response(url, default=[])

    def get_match_data(self, match_id):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}"
        return self._get_response(url, default={})

    def get_match_info(self, match_id):
        self.reset_api_call_count()
        match_data = self.get_match_data(match_id)
        if not match_data:
            self.logger.error(f"invalid match data for match id:{match_id}")
            return None, None, None, self.api_call_count
        
        try:
            game_id = match_data['metadata']['matchId']
            game_duration = match_data['info']['gameDuration']
            game_version = match_data['info']['gameVersion']
            queue_id = match_data['info']['queueId']
            mode = match_data['info']['gameMode']

            minutes, seconds = divmod(game_duration, 60)
            formatted_duration = f"{minutes}m {seconds}s"

            patch = '.'.join(game_version.split('.')[:2])

            timestamp = datetime.fromtimestamp(match_data['info']['gameCreation'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

            general_info = {
                'game_id': game_id,
                'game_duration': formatted_duration,
                'patch': patch,
                'timestamp': timestamp,
                'mode': mode,
                'queue_id': queue_id
            }

            participant_info = {'Blue Side': {}, 'Red Side': {}}
            team_stats = {'Blue Side': {}, 'Red Side': {}}
            for team in ['Blue Side', 'Red Side']:
                team_stats[team] = {
                    'total_kills': 0,
                    'total_deaths': 0,
                    'total_assists': 0,
                    'total_damage_dealt': 0,
                    'total_gold_earned': 0,
                    'total_cs': 0,
                    'wards_placed': 0,
                    'wards_killed': 0,
                    'total_level': 0,
                    'total_xp': 0,
                    'damage_taken': 0,
                    'heal': 0,
                    'damage_mitigated': 0,
                    'time_ccing_others': 0
                }

            for participant in match_data['info']['participants']:
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'

                if participant['summonerName'] == "":
                    participant['summonerName'] = participant['riotIdGameName']
                
                participant_info[team][participant['summonerName']] = {
                    'summoner_id': participant['summonerId'],
                    'summoner_name': participant['summonerName'],
                    'champion_name': participant['championName'],
                    'champion_id': participant['championId'],
                    'champ_level': participant['champLevel'],
                    'role': participant.get('role', 'N/A'),
                    'lane': participant.get('lane', 'N/A'),
                    'position': participant.get('teamPosition', 'N/A'),
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


                team_stats[team]['total_kills'] += participant['kills']
                team_stats[team]['total_deaths'] += participant['deaths']
                team_stats[team]['total_assists'] += participant['assists']
                team_stats[team]['total_damage_dealt'] += participant['totalDamageDealtToChampions']
                team_stats[team]['total_gold_earned'] += participant['goldEarned']
                team_stats[team]['total_cs'] += participant['totalMinionsKilled'] + participant['neutralMinionsKilled']
                team_stats[team]['wards_placed'] += participant['wardsPlaced']
                team_stats[team]['wards_killed'] += participant['wardsKilled']
                team_stats[team]['total_level'] += participant['champLevel']
                team_stats[team]['total_xp'] += participant['champExperience']
                team_stats[team]['damage_taken'] += participant['totalDamageTaken']
                team_stats[team]['heal'] += participant['totalHeal']
                team_stats[team]['damage_mitigated'] += participant['damageSelfMitigated']
                team_stats[team]['time_ccing_others'] += participant['timeCCingOthers']

            logging.info(f"Team stats: {team_stats}")
            logging.info(f"Participant info: {participant_info}")

            for team in ['Blue Side', 'Red Side']:
                if not all(key in team_stats[team] for key in ['total_kills', 'total_deaths', 'total_damage_dealt', 'total_gold_earned', 'total_cs', 'wards_placed', 'wards_killed', 'total_xp']):
                    self.logger.error(f"Missing required team stats for {team}")
                    return None, None, None, self.api_call_count

            for team, players in participant_info.items():
                for player, stats in players.items():
                    required_keys = ['kills', 'deaths', 'assists', 'total_damage_dealt', 'gold_earned', 'cs', 'wards_placed', 'wards_killed', 'xp', 'time_ccing_others']
                    if not all(key in stats for key in required_keys):
                        self.logger.error(f"Missing required player stats for {player}")
                        return None, None, None, self.api_call_count

            return general_info, participant_info, team_stats, self.api_call_count

        except KeyError as e:
            logging.error(f"Key error in match data: {e}. Match data: {match_data}")
            return None, None, None, self.api_call_count
        
    def get_player_statistics(self, session, queue_id):
        logging.info(f"Retrieving statistics for queue_id: {queue_id}")
        
        matches = session.query(Match).join(Team).join(Participant).filter(
            Participant.summoner_name == self.username,
            Match.queue_id == queue_id
        ).all()
        
        logging.info(f"Found {len(matches)} matches with exact queue_id {queue_id}")
        
        if not matches:
            logging.warning(f"No matches found for queue_id {queue_id}. Retrieving all matches for the user.")
            matches = session.query(Match).join(Team).join(Participant).filter(
                Participant.summoner_name == self.username
            ).all()
            logging.info(f"Found {len(matches)} total matches for the user")

        total_games = len(matches)
        wins = 0
        champion_stats = {}
        total_kills, total_deaths, total_assists, total_cs = 0, 0, 0, 0
        roles_played = {}

        for match in matches:
            logging.debug(f"Processing match {match.game_id} with queue_id {match.queue_id}")
            for team in match.teams:
                for participant in team.participants:
                    if participant.summoner_name == self.username:
                        if team.win:
                            wins += 1
                        if participant.champion_name not in champion_stats:
                            champion_stats[participant.champion_name] = {'games': 0, 'wins': 0, 'kills': 0, 'deaths': 0, 'assists': 0, 'cs': 0}
                        champ_stats = champion_stats[participant.champion_name]
                        champ_stats['games'] += 1
                        if team.win:
                            champ_stats['wins'] += 1
                        champ_stats['kills'] += participant.kills
                        champ_stats['deaths'] += participant.deaths
                        champ_stats['assists'] += participant.assists
                        champ_stats['cs'] += participant.cs
                        total_kills += participant.kills
                        total_deaths += participant.deaths
                        total_assists += participant.assists
                        total_cs += participant.cs
                        roles_played[participant.position] = roles_played.get(participant.position, 0) + 1

        logging.info(f"Statistics summary: Total games={total_games}, Wins={wins}, Roles={roles_played}")

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

    def calculate_performance_features(self, participant_info, team_stats, summoner_name):
        user_team = next((team for team, players in participant_info.items() if summoner_name in players), None)
        if not user_team:
            logging.warning(f"Could not find team for {summoner_name}")
            return None
        
        enemy_team = 'Blue Side' if user_team == 'Red Side' else 'Red Side'
        user_stats = participant_info[user_team][summoner_name]
        
        features = {}
        feature_calculations = [
            ('kill_participation', lambda: (user_stats['kills'] + user_stats['assists']) / max(1, team_stats[user_team]['total_kills'])),
            ('death_share', lambda: user_stats['deaths'] / max(1, team_stats[user_team]['total_deaths'])),
            ('damage_share', lambda: user_stats['total_damage_dealt'] / max(1, team_stats[user_team]['total_damage_dealt'])),
            ('damage_taken_share', lambda: user_stats['damage_taken'] / max(1, team_stats[user_team]['damage_taken'])),
            ('gold_share', lambda: user_stats['gold_earned'] / max(1, team_stats[user_team]['total_gold_earned'])),
            ('heal_share', lambda: user_stats['total_heal'] / max(1, team_stats[user_team]['heal'])),
            ('damage_mitigated_share', lambda: user_stats['damage_mitigated'] / max(1, team_stats[user_team]['damage_mitigated'])),
            ('cs_share', lambda: user_stats['cs'] / max(1, team_stats[user_team]['total_cs'])),
            ('vision_share', lambda: user_stats['wards_placed'] / max(1, team_stats[user_team]['wards_placed'])),
            ('vision_denial_share', lambda: user_stats['wards_killed'] / max(1, team_stats[user_team]['wards_killed'])),
            ('xp_share', lambda: user_stats['xp'] / max(1, team_stats[user_team]['total_xp'])),
            ('cc_share', lambda: user_stats['time_ccing_others'] / max(1, team_stats[user_team]['time_ccing_others']))
        ]
        
        for feature, calculation in feature_calculations:
            try:
                value = calculation()
                features[feature] = value
                logging.info(f"Calculated {feature}: {value}")
            except Exception as e:
                logging.error(f"Error calculating {feature}: {str(e)}")
                features[feature] = 0
        
        return features

    def get_matches_by_date(self, queue_id, start_time, end_time):
        matches = []
        start_index = 0
        while True:
            url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?queue={queue_id}&startTime={start_time}&endTime={end_time}&start={start_index}&count=100&api_key={self.API_key}"
            batch = self._get_response(url, default=[])
            if not batch:
                break
            matches.extend(batch)
            if len(batch) < 100:
                break
            start_index += 100
        return matches

    def get_recent_matches(self, queue_id, count):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?queue={queue_id}&start=0&count={count}&api_key={self.API_key}"
        return self._get_response(url, default=[])

    def _get_response(self, url, default=None):
        self.api_call_count += 1
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()
            except Exception as e:
                logging.error(f"Error parsing response: {e}. Response: {response.text}")
                return default
        else:
            logging.error(f"API request failed with status code {response.status_code}. Response: {response.text}")
            return default

class UserDisplay:
    def __init__(self, user_data):
        self.user_data = user_data

    def display_user_info(self):
        summoner_id = self.user_data.get_summoner_id()
        if not summoner_id:
            logging.error("Failed to get summoner ID.")
            return None

        ranked_data = self.user_data.get_ranked_data(summoner_id)
        info = {}
        for entry in ranked_data:
            if entry.get('queueType') == 'RANKED_SOLO_5x5':
                tier = entry['tier']
                rank = entry['rank']
                league_points = entry['leaguePoints']
                info['ranked'] = f'{tier} {rank} - {league_points} LP'
            elif entry.get('queueType') == 'RANKED_FLEX_SR':
                tier = entry['tier']
                rank = entry['rank']
                league_points = entry['leaguePoints']
                info['flex'] = f'{tier} {rank} - {league_points} LP'
        return info

    def display_match_info(self, match_id):
        match_info = self.user_data.get_match_info(match_id)
        if match_info[0] is None:
            return None
        
        general_info, participant_info, team_stats, api_calls = match_info
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
        formatted_stats = {
            'total_games': stats['total_games'],
            'wins': stats['wins'],
            'losses': stats['total_games'] - stats['wins'],
            'win_rate': round(stats['wins'] / stats['total_games'] * 100, 2) if stats['total_games'] > 0 else 0,
            'kda_ratio': ((stats['total_kills'] + stats['total_assists']) / stats['total_deaths']) if stats['total_deaths'] > 0 else (stats['total_kills'] + stats['total_assists']),
            'avg_cs': stats['total_cs'] / stats['total_games'] if stats['total_games'] > 0 else 0,
            'most_played_roles': max(stats['roles_played'], key=stats['roles_played'].get) if stats['roles_played'] else 'N/A',
            'champion_stats': {}
        }
        
        for champ, champ_stats in stats['champion_stats'].items():
            formatted_stats['champion_stats'][champ] = {
                'games': champ_stats['games'],
                'wins': champ_stats['wins'],
                'winrate': champ_stats['wins'] / champ_stats['games'] * 100 if champ_stats['games'] > 0 else 0,
                'kda_ratio': ((champ_stats['kills'] + champ_stats['assists']) / champ_stats['deaths']) if champ_stats['deaths'] > 0 else (champ_stats['kills'] + champ_stats['assists']),
                'avg_cs': champ_stats['cs'] / champ_stats['games'] if champ_stats['games'] > 0 else 0
            }
        return formatted_stats

def save_match_data(user_data, match_ids, session):
    added_matches = 0
    updated_matches = 0
    total_api_calls = 0
    start_time = time.time()

    for i, match_id in enumerate(match_ids, 1):
        logging.info(f"Processing match {i} of {len(match_ids)}: {match_id}")
        
        general_info, participant_info, team_stats, api_calls = user_data.get_match_info(match_id)
        total_api_calls += api_calls

        if general_info is None or participant_info is None or team_stats is None:
            logging.warning(f"Could not retrieve info for match {match_id}")
            continue

        game_duration_minutes = int(general_info['game_duration'].split('m')[0])
        if game_duration_minutes < 5:
            logging.info(f"Skipping remake match {match_id}")
            continue

        logging.info(f"Processing match {i} of {len(match_ids)}: {general_info['game_id']}")
        existing_match = session.query(Match).filter_by(game_id=general_info['game_id']).first()
        if existing_match:
            logging.info(f"Match {general_info['game_id']} already exists. Updating all data.")
            for key, value in general_info.items():
                if hasattr(existing_match, key):
                    setattr(existing_match, key, value)
            updated_matches += 1
            match = existing_match
        else:
            logging.info(f"Creating new match {general_info['game_id']}")
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
            added_matches += 1

        logging.info(f"Match ID after flush: {match.match_id}")

        for team_name, participants in participant_info.items():
            if not participants:
                logging.warning(f"No participants found for {team_name}")
                continue
            
            logging.info(f"Participants for {team_name}: {participants}")
            
            try:
                first_participant = next(iter(participants.values()))
                logging.info(f"First participant data: {first_participant}")
                
                team_win_status = first_participant.get('win')
                if team_win_status is None:
                    logging.warning(f"Win status not found for {team_name}. Defaulting to False.")
                    team_win_status = False
            except StopIteration:
                logging.warning(f"Could not get win status for {team_name} in match {match_id}")
                continue

            team = Team(
                match_id=match.match_id,
                team_name=team_name,
                win=team_win_status
            )
            session.add(team)
            session.flush()
            logging.info(f"Added team: {team_name}, ID: {team.team_id}, Match ID: {team.match_id}")

            for summoner, details in participants.items():
                participant = Participant(team_id=team.team_id)
                
                logging.info(f"Participant details for {summoner}: {details}")

                for attr, value in details.items():
                    if attr == 'team':
                        continue
                    if hasattr(participant, attr):
                        if isinstance(value, (int, float, bool, str)):
                            setattr(participant, attr, value)
                        else:
                            logging.warning(f"Skipping attribute {attr} with unexpected value type: {type(value)}")
                    else:
                        logging.warning(f"Participant model does not have attribute: {attr}")

                required_fields = ['champion_name', 'champion_id', 'champ_level', 'role', 'lane', 'position']
                for field in required_fields:
                    if getattr(participant, field, None) is None:
                        logging.error(f"Required field {field} is None for participant {summoner}")

                session.add(participant)
                session.flush()
                logging.info(f"Added participant: {summoner}, ID: {participant.participant_id}, Team ID: {participant.team_id}")

                features = user_data.calculate_performance_features(participant_info, team_stats, summoner)
                if features:
                    logging.info(f"Calculated features for {summoner}: {features}")
                    existing_features = session.query(PerformanceFeatures).filter_by(participant_id=participant.participant_id).first()
                    if existing_features:
                        for key, value in features.items():
                            if hasattr(existing_features, key):
                                setattr(existing_features, key, value)
                                logging.info(f"Updating {key} to {value} for {summoner}")
                    else:
                        performance_features = PerformanceFeatures(participant_id=participant.participant_id)
                        for key, value in features.items():
                            if hasattr(performance_features, key):
                                setattr(performance_features, key, value)
                                logging.info(f"Setting {key} to {value} for {summoner}")
                            else:
                                logging.warning(f"PerformanceFeatures does not have attribute: {key}")
                        session.add(performance_features)
                        logging.info(f"Added performance features for participant: {summoner}, ID: {performance_features.id}")

        try:
            session.commit()
            logging.info(f"Committed data for match {general_info['game_id']}")
        except Exception as e:
            logging.error(f"Error committing data for match {general_info['game_id']}: {str(e)}")
            session.rollback()

        elapsed_time = time.time() - start_time
        if elapsed_time < 120 and total_api_calls >= 70:
            pause_time = 120 - elapsed_time
            logging.info(f"Pausing for {pause_time:.2f} seconds to avoid rate limit...")
            time.sleep(pause_time)
            total_api_calls = 0
            start_time = time.time()
        
        logging.info("Waiting for 2.2 seconds before next match...")
        time.sleep(2.1)

    logging.info(f"Total games added: {added_matches}, updated: {updated_matches}, API calls: {total_api_calls}")
    
    if added_matches + updated_matches > 0:
        calculate_performance_ratings()
    return added_matches, updated_matches, total_api_calls