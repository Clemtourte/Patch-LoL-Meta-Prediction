import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from models import Match, Team, Participant, PerformanceFeatures, init_db
from sqlalchemy.orm.exc import NoResultFound

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

Session = init_db()

class UserData:
    def __init__(self, username, tag, region):
        load_dotenv() 
        self.API_key = os.getenv('API_KEY')
        self.username = username
        self.tag = tag
        self.region = region
        self.puuid = self.get_puuid()

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

def get_matches(self, queue_id, start_time, end_time, count=100):
    matches = []
    start_index = 0
    while len(matches) < count:
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?queue={queue_id}&startTime={start_time}&endTime={end_time}&start={start_index}&count=100&api_key={self.API_key}"
        batch = self._get_response(url, default=[])
        if not batch:
            break
        matches.extend(batch)
        start_index += 100
    return matches[:count]

    def get_match_data(self, match_id):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}"
        return self._get_response(url, default={})

    def get_match_info(self, match_id):
        match_data = self.get_match_data(match_id)
        if not match_data:
            return None, None, None
        
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
                    'damage_taken': 0,
                    'heal': 0,
                    'damage_mitigated': 0,
                    'time_ccing_others': 0
                }

            for participant in match_data['info']['participants']:
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'
                kills = participant['kills']
                deaths = participant['deaths']
                assists = participant['assists']
                win = participant['win']
                kda = (kills + assists) / deaths if deaths > 0 else kills + assists
                
                participant_info[team][participant['summonerName']] = {
                    'summoner_id': participant['summonerId'],
                    'summoner_name': participant['summonerName'],
                    'team': 'Blue' if participant['teamId'] == 100 else 'Red',
                    'win': win,
                    'champ_level': participant['champLevel'],
                    'champ_name': participant['championName'],
                    'champ_id': participant['championId'],
                    'role': participant.get('role', 'N/A'),
                    'lane': participant.get('lane', 'N/A'),
                    'position': participant.get('teamPosition', 'N/A'),
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'kda': round(kda, 2),
                    'gold_earned': participant['goldEarned'],
                    'total_damage_dealt': participant['totalDamageDealtToChampions'],
                    'cs': participant['totalMinionsKilled'] + participant['neutralMinionsKilled'],
                    'total_heal': participant.get('totalHeal', 'N/A'),
                    'damage_taken': participant['totalDamageTaken'],
                    'heal': participant['totalHeal'],
                    'damage_mitigated': participant['damageSelfMitigated'],
                    'wards_placed': participant['wardsPlaced'],
                    'wards_killed': participant['wardsKilled'],
                    'xp': participant['champExperience'],
                    'time_ccing_others': participant['timeCCingOthers']
                }

                for stat in team_stats[team]:
                    if stat in participant_info[team][participant['summonerName']]:
                        team_stats[team][stat] += participant_info[team][participant['summonerName']][stat]

            return general_info, participant_info, team_stats

        except KeyError as e:
            logging.error(f"Key error in match data: {e}. Match data: {match_data}")
            return None, None, None
        
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
        
        user_stats = participant_info[user_team][summoner_name]
        
        features = {}
        feature_calculations = [
            ('kill_participation', lambda: (user_stats['kills'] + user_stats['assists']) / team_stats[user_team]['total_kills'] if team_stats[user_team]['total_kills'] > 0 else 0),
            ('death_share', lambda: user_stats['deaths'] / team_stats[user_team]['total_deaths'] if team_stats[user_team]['total_deaths'] > 0 else 0),
            ('damage_share', lambda: user_stats['total_damage_dealt'] / team_stats[user_team]['total_damage_dealt'] if team_stats[user_team]['total_damage_dealt'] > 0 else 0),
            ('damage_taken_share', lambda: user_stats['damage_taken'] / team_stats[user_team]['damage_taken'] if team_stats[user_team]['damage_taken'] > 0 else 0),
            ('gold_share', lambda: user_stats['gold_earned'] / team_stats[user_team]['total_gold_earned'] if team_stats[user_team]['total_gold_earned'] > 0 else 0),
            ('heal_share', lambda: user_stats['heal'] / team_stats[user_team]['heal'] if team_stats[user_team]['heal'] > 0 else 0),
            ('damage_mitigated_share', lambda: user_stats['damage_mitigated'] / team_stats[user_team]['damage_mitigated'] if team_stats[user_team]['damage_mitigated'] > 0 else 0),
            ('cs_share', lambda: user_stats['cs'] / team_stats[user_team]['total_cs'] if team_stats[user_team]['total_cs'] > 0 else 0),
            ('vision_share', lambda: user_stats['wards_placed'] / team_stats[user_team]['wards_placed'] if team_stats[user_team]['wards_placed'] > 0 else 0),
            ('vision_denial_share', lambda: user_stats['wards_killed'] / team_stats[user_team]['wards_killed'] if team_stats[user_team]['wards_killed'] > 0 else 0),
            ('xp_share', lambda: user_stats['xp'] / team_stats[user_team]['total_level'] if team_stats[user_team]['total_level'] > 0 else 0),
            ('cc_share', lambda: user_stats['time_ccing_others'] / team_stats[user_team]['time_ccing_others'] if team_stats[user_team]['time_ccing_others'] > 0 else 0)
        ]
        
        for feature, calculation in feature_calculations:
            try:
                features[feature] = calculation()
            except KeyError as e:
                logging.warning(f"Missing key for {feature}: {e}")
                features[feature] = 0
        
        return features

    def _get_response(self, url, default=None):
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
        if match_info is None:
            return None
        
        general_info, participant_info, team_stats = match_info
        return {
            "Game ID": general_info['game_id'],
            "Duration": general_info['game_duration'],
            "Patch": general_info['patch'],
            "Date": general_info['timestamp'],
            "Game Mode": general_info['mode'],
            "Queue ID": general_info['queue_id'],
            "Teams": participant_info
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

import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from models import Match, Team, Participant, PerformanceFeatures, init_db
from sqlalchemy.orm.exc import NoResultFound

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

Session = init_db()

class UserData:
    def __init__(self, username, tag, region):
        load_dotenv() 
        self.API_key = os.getenv('API_KEY')
        self.username = username
        self.tag = tag
        self.region = region
        self.puuid = self.get_puuid()

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
        match_data = self.get_match_data(match_id)
        if not match_data:
            return None, None, None
        
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
                    'damage_taken': 0,
                    'heal': 0,
                    'damage_mitigated': 0,
                    'time_ccing_others': 0
                }

            for participant in match_data['info']['participants']:
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'
                kills = participant['kills']
                deaths = participant['deaths']
                assists = participant['assists']
                win = participant['win']
                kda = (kills + assists) / deaths if deaths > 0 else kills + assists
                
                participant_info[team][participant['summonerName']] = {
                    'summoner_id': participant['summonerId'],
                    'summoner_name': participant['summonerName'],
                    'team': 'Blue' if participant['teamId'] == 100 else 'Red',
                    'win': win,
                    'champ_level': participant['champLevel'],
                    'champ_name': participant['championName'],
                    'champ_id': participant['championId'],
                    'role': participant.get('role', 'N/A'),
                    'lane': participant.get('lane', 'N/A'),
                    'position': participant.get('teamPosition', 'N/A'),
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'kda': round(kda, 2),
                    'gold_earned': participant['goldEarned'],
                    'total_damage_dealt': participant['totalDamageDealtToChampions'],
                    'cs': participant['totalMinionsKilled'] + participant['neutralMinionsKilled'],
                    'total_heal': participant.get('totalHeal', 'N/A'),
                    'damage_taken': participant['totalDamageTaken'],
                    'heal': participant['totalHeal'],
                    'damage_mitigated': participant['damageSelfMitigated'],
                    'wards_placed': participant['wardsPlaced'],
                    'wards_killed': participant['wardsKilled'],
                    'xp': participant['champExperience'],
                    'time_ccing_others': participant['timeCCingOthers']
                }

                for stat in team_stats[team]:
                    if stat in participant_info[team][participant['summonerName']]:
                        team_stats[team][stat] += participant_info[team][participant['summonerName']][stat]

            return general_info, participant_info, team_stats

        except KeyError as e:
            logging.error(f"Key error in match data: {e}. Match data: {match_data}")
            return None, None, None
        
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
        
        user_stats = participant_info[user_team][summoner_name]
        
        features = {}
        feature_calculations = [
            ('kill_participation', lambda: (user_stats['kills'] + user_stats['assists']) / team_stats[user_team]['total_kills'] if team_stats[user_team]['total_kills'] > 0 else 0),
            ('death_share', lambda: user_stats['deaths'] / team_stats[user_team]['total_deaths'] if team_stats[user_team]['total_deaths'] > 0 else 0),
            ('damage_share', lambda: user_stats['total_damage_dealt'] / team_stats[user_team]['total_damage_dealt'] if team_stats[user_team]['total_damage_dealt'] > 0 else 0),
            ('damage_taken_share', lambda: user_stats['damage_taken'] / team_stats[user_team]['damage_taken'] if team_stats[user_team]['damage_taken'] > 0 else 0),
            ('gold_share', lambda: user_stats['gold_earned'] / team_stats[user_team]['total_gold_earned'] if team_stats[user_team]['total_gold_earned'] > 0 else 0),
            ('heal_share', lambda: user_stats['heal'] / team_stats[user_team]['heal'] if team_stats[user_team]['heal'] > 0 else 0),
            ('damage_mitigated_share', lambda: user_stats['damage_mitigated'] / team_stats[user_team]['damage_mitigated'] if team_stats[user_team]['damage_mitigated'] > 0 else 0),
            ('cs_share', lambda: user_stats['cs'] / team_stats[user_team]['total_cs'] if team_stats[user_team]['total_cs'] > 0 else 0),
            ('vision_share', lambda: user_stats['wards_placed'] / team_stats[user_team]['wards_placed'] if team_stats[user_team]['wards_placed'] > 0 else 0),
            ('vision_denial_share', lambda: user_stats['wards_killed'] / team_stats[user_team]['wards_killed'] if team_stats[user_team]['wards_killed'] > 0 else 0),
            ('xp_share', lambda: user_stats['xp'] / team_stats[user_team]['total_level'] if team_stats[user_team]['total_level'] > 0 else 0),
            ('cc_share', lambda: user_stats['time_ccing_others'] / team_stats[user_team]['time_ccing_others'] if team_stats[user_team]['time_ccing_others'] > 0 else 0)
        ]
        
        for feature, calculation in feature_calculations:
            try:
                features[feature] = calculation()
            except KeyError as e:
                logging.warning(f"Missing key for {feature}: {e}")
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
        if match_info is None:
            return None
        
        general_info, participant_info, team_stats = match_info
        return {
            "Game ID": general_info['game_id'],
            "Duration": general_info['game_duration'],
            "Patch": general_info['patch'],
            "Date": general_info['timestamp'],
            "Game Mode": general_info['mode'],
            "Queue ID": general_info['queue_id'],
            "Teams": participant_info
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

    for match_id in match_ids:
        general_info, participant_info, team_stats = user_data.get_match_info(match_id)
        if general_info is None or participant_info is None or team_stats is None:
            logging.warning(f"Could not retrieve info for match {match_id}")
            continue

        logging.info(f"Processing match {general_info['game_id']}")
        existing_match = session.query(Match).filter_by(game_id=general_info['game_id']).first()
        if existing_match:
            logging.info(f"Match {general_info['game_id']} already exists. Updating all data.")
            match = existing_match
            for key, value in general_info.items():
                if hasattr(match, key):
                    setattr(match, key, value)
            updated_matches += 1
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
            added_matches += 1

        session.flush()

        for team_name, participants in participant_info.items():
            team = session.query(Team).filter_by(match_id=match.match_id, team_name=team_name).first()
            if team:
                team.win = next(iter(participants.values()))['win']
            else:
                team = Team(
                    match_id=match.match_id,
                    team_name=team_name,
                    win=next(iter(participants.values()))['win']
                )
                session.add(team)

            session.flush()

            for summoner, details in participants.items():
                participant = session.query(Participant).filter_by(team_id=team.team_id, summoner_name=summoner).first()
                
                if participant:
                    logging.info(f"Updating existing participant: {summoner}")
                else:
                    logging.info(f"Creating new participant: {summoner}")
                    participant = Participant(team_id=team.team_id)
                    session.add(participant)

                # Explicitly set each attribute
                participant.summoner_id = str(details['summoner_id'])
                participant.summoner_name = str(summoner)
                participant.champion_name = str(details['champ_name'])
                participant.champion_id = int(details['champ_id'])
                participant.champ_level = int(details['champ_level'])
                participant.role = str(details['role'])
                participant.lane = str(details['lane'])
                participant.position = str(details['position'])
                participant.kills = int(details['kills'])
                participant.deaths = int(details['deaths'])
                participant.assists = int(details['assists'])
                participant.kda = str(details['kda'])
                participant.gold_earned = int(details['gold_earned'])
                participant.total_damage_dealt = int(details['total_damage_dealt'])
                participant.cs = int(details['cs'])
                participant.total_heal = int(details['total_heal']) if details['total_heal'] != 'N/A' else 0
                participant.damage_taken = int(details['damage_taken'])
                participant.wards_placed = int(details['wards_placed'])
                participant.wards_killed = int(details['wards_killed'])

                session.flush()

                features = user_data.calculate_performance_features(participant_info, team_stats, summoner)
                if features:
                    performance_features = session.query(PerformanceFeatures).filter_by(participant_id=participant.participant_id).first()
                    if not performance_features:
                        performance_features = PerformanceFeatures(participant_id=participant.participant_id)
                        session.add(performance_features)
                    
                    for key, value in features.items():
                        setattr(performance_features, key, float(value))

        try:
            session.commit()
            logging.info(f"Processed match {general_info['game_id']}.")
        except Exception as e:
            logging.error(f"Error committing session: {str(e)}")
            session.rollback()
    
    return added_matches, updated_matches