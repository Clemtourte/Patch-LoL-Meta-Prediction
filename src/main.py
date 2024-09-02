import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from models import init_db
from models import Match, Team, Participant
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

    def get_matches(self, match_type, match_count):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?type={match_type}&start=0&count={match_count}&api_key={self.API_key}"
        return self._get_response(url, default=[])

    def get_match_data(self, match_id):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}"
        return self._get_response(url, default={})

    def get_match_info(self, match_id):
        match_data = self.get_match_data(match_id)
        if not match_data:
            return None, None
        
        mode = match_data['info']['gameMode']
        if mode != 'CLASSIC':
            return None, None
        
        try:
            game_id = match_data['metadata']['matchId']
            game_duration = match_data['info']['gameDuration']
            game_version = match_data['info']['gameVersion']
            type = match_data['info']['gameType']

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
                'type': type
            }

            participant_info = {'Blue Side': {}, 'Red Side': {}}
            for i, participant in enumerate(match_data['info']['participants'], 1):
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'
                kills = participant['kills']
                deaths = participant['deaths']
                assists = participant['assists']
                win = participant['win']
                if deaths == 0:
                    kda = kills + assists
                else:
                    kda = round((kills + assists)/deaths, 2)
                
                participant_info[team][f'Summoner {i}'] = {
                    'summoner_id': participant['summonerId'],
                    'summoner_name': participant['summonerName'],
                    'team': 'Blue' if participant['teamId'] == 100 else 'Red',
                    'win': win,
                    'champ_level': participant['champLevel'],
                    'champ_name': participant['championName'],
                    'champ_id': participant['championId'],
                    'role': participant['role'],
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'kda': kda,
                    'gold_earned': participant['goldEarned'],
                    'total_damage_dealt': participant['totalDamageDealtToChampions'],
                    'cs': participant['totalMinionsKilled'] + participant['neutralMinionsKilled']
                }

            return general_info, participant_info

        except KeyError as e:
            logging.error("Key error: %s. Here's the entire response: %s", e, match_data)
            return None, None

    def _get_response(self, url, default=None):
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()
            except Exception as e:
                logging.error("Error parsing response: %s. Response: %s", e, response.text)
                return default
        else:
            logging.error("API request failed with status code %s. Response: %s", response.status_code, response.json())
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
        for entry in ranked_data:
            if entry.get('queueType') == 'RANKED_SOLO_5x5':
                tier = entry['tier']
                rank = entry['rank']
                league_points = entry['leaguePoints']
                return f"{tier} {rank} - {league_points} LP"
        logging.info("No ranked data found for the summoner.")
        return "No ranked data found."

    def display_match_info(self, match_id):
        general_info, participant_info = self.user_data.get_match_info(match_id)
        if general_info is None or participant_info is None:
            return None
        
        return {
            "Game ID": general_info['game_id'],
            "Duration": general_info['game_duration'],
            "Patch": general_info['patch'],
            "Date": general_info['timestamp'],
            "Game Mode": general_info['mode'],
            "Game Type": general_info['type'],
            "Teams": participant_info
        }

def save_match_data(user_data, match_ids, session):
    for match_id in match_ids:
        general_info, participant_info = user_data.get_match_info(match_id)
        if general_info is None or participant_info is None:
            continue

        # Vérifier si le match existe déjà
        existing_match = session.query(Match).filter_by(game_id=general_info['game_id']).first()
        if existing_match:
            logging.info(f"Match {general_info['game_id']} already exists in the database.")
            continue

        match = Match(
            game_id=general_info['game_id'],
            game_duration=general_info['game_duration'],
            patch=general_info['patch'],
            timestamp=general_info['timestamp'],
            mode=general_info['mode'],
            platform="EUW1"  # Peut être défini en fonction de la région sélectionnée
        )
        session.add(match)
        session.commit()

        for team_name, participants in participant_info.items():
            team = Team(
                match_id=match.match_id,
                team_name=team_name,
                win=next(iter(participants.values()))['win']
            )
            session.add(team)
            session.commit()

            for summoner, details in participants.items():
                existing_participant = session.query(Participant).filter_by(
                    team_id=team.team_id,
                    summoner_id=details['summoner_id']
                ).first()
                if existing_participant:
                    logging.info(f"Participant {details['summoner_name']} already exists in the database.")
                    continue

                participant = Participant(
                    team_id=team.team_id,
                    summoner_id=details['summoner_id'],
                    summoner_name=details['summoner_name'],
                    champion_name=details['champ_name'],
                    champion_id=details['champ_id'],
                    champ_level=details['champ_level'],
                    role=details['role'],
                    kills=details['kills'],
                    deaths=details['deaths'],
                    assists=details['assists'],
                    kda=str(details['kda']),
                    gold_earned=details['gold_earned'],
                    total_damage_dealt=details['total_damage_dealt'],
                    cs=details['cs']
                )
                session.add(participant)

        session.commit()

"""if __name__ == "__main__":
    username = "YourUsername"
    tag = "YourTag"
    region = "euw1" 

    user_data = UserData(username, tag, region)
    match_ids = user_data.get_matches(match_type="ranked", match_count=10)

    session = Session()

    try:
        save_match_data(user_data, match_ids, session)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        session.close()"""