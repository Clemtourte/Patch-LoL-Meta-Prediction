import requests
import os
from dotenv import load_dotenv
import logging
import json
from datetime import datetime
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
                'platform': match_data['info']['platformId']
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
                    kda = round((kills + assists)/deaths,2)
                
                participant_info[team][f'Summoner {i}'] = {
                    'summoner_id': participant['summonerId'],
                    'summoner_name': participant['summonerName'],
                    'team': participant['teamId'],
                    'win': win,
                    'champ_level': participant['champLevel'],
                    'champ_name': participant['championName'],
                    'champ_id': participant['championId'],
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'kda': kda,
                    'gold_earned': participant['goldEarned'],
                    'total_damage_dealt': participant['totalDamageDealtToChampions']
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
                return tier, rank, league_points
        logging.info("No ranked data found for the summoner.")
        return None

    def display_match_info(self, match_id):
        general_info, participant_info = self.user_data.get_match_info(match_id)
        if general_info is None or participant_info is None:
            return None
        
        # Combine both into a single dictionary
        display_data = {
            "game_id": general_info['game_id'],
            "game_duration": general_info['game_duration'],
            "patch": general_info['patch'],
            "timestamp": general_info['timestamp'],
            "mode": general_info['mode'],
            "platform": general_info['platform'],
            "teams": participant_info
        }

        return display_data


def save_match_data(user_data, match_ids, file_path="../datasets/match_data.json"):
    all_participants_info = {}

    try:
        with open(file_path, "r") as in_file:
            all_participants_info = json.load(in_file)
    except FileNotFoundError:
        all_participants_info = {}

    match_number = len(all_participants_info) + 1

    for match_id in match_ids:
        general_info, participant_info = user_data.get_match_info(match_id)
        if general_info is None and participant_info is None:
            continue
        game_id = general_info.get('game_id')
        if game_id is None or any(match["general_info"]["game_id"] == game_id for match in all_participants_info.values()):
            continue
        all_participants_info[f"match_{match_number}"] = {
            "general_info": general_info,
            "details": participant_info
        }
        match_number += 1

    with open(file_path, "w") as out_file:
        json.dump(all_participants_info, out_file, indent=4)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python main.py <username> <tag> <region>")
        sys.exit(1)
    
    username = sys.argv[1]
    tag = sys.argv[2]
    region = sys.argv[3]
    
    user_data = UserData(username, tag, region)
    
    user_display = UserDisplay(user_data)

    user_display.display_user_info()

    match_ids = user_data.get_matches('ranked', 20)

    save_match_data(user_data, match_ids)
