import requests
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class User:
    def __init__(self, username, tag, region):
        load_dotenv()  # Load environment variables from .env file
        self.API_key = os.getenv('API_KEY')
        self.username = username
        self.tag = tag
        self.region = region
        self.puuid = self.get_puuid()

    def get_puuid(self):
        url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{self.username}/{self.tag}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()['puuid']
            except KeyError:
                logging.error("Key 'puuid' not found in response. Response: %s", response.json())
        else:
            logging.error("API request failed with status code %s", response.status_code, response.json())
        return None
    
    def user_info(self):
        url = f"https://{self.region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{self.puuid}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                for entry in data:
                    if entry.get('queueType') == 'RANKED_SOLO_5x5':
                        tier = entry['tier']
                        rank = entry['rank']
                        league_points = entry['leaguePoints']
                        print(f"Tier: {tier} {rank}, League Points: {league_points}")
                        return tier, rank, league_points
                print("No ranked data found for the summoner.")
            except Exception as e:
                print(f"Error parsing response: {e}. Response: {response.text}")
        else:
            print(f"API request failed with status code {response.status_code}. Here's the entire response:")
            print(response.text)

    def get_matches(self, match_type, match_count):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?type={match_type}&start=0&count={match_count}&api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code {response.status_code}. Here's the entire response:")
            print(response.json())
        return []

    def get_match_data(self, match_id):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code {response.status_code}. Here's the entire response:")
            print(response.json())
        return {}

    def get_match_info(self, match_id):
        match_data = self.get_match_data(match_id)
        if not match_data:
            return {}

        try:
            game_id = match_data['metadata']['matchId']
            game_duration = match_data['info']['gameDuration']
            game_version = match_data['info']['gameVersion']

            general_info = {
                'game_id': game_id,
                'game_duration': round(game_duration / 60, 2),
                'patch': game_version.split('.')
            }

            participant_info = {'Blue Side': {}, 'Red Side': {}}
            for i, participant in enumerate(match_data['info']['participants'], 1):
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'
                participant_info[team][f'Summoner {i}'] = {
                    'summoner_id': participant['puuid'],
                    'summoner_name': participant['summonerName'],
                    'team': participant['teamId'],
                    'champ_level': participant['champLevel'],
                    'champ_name': participant['championName'],
                    'champ_id': participant['championId'],
                    'kills': participant['kills'],
                    'deaths': participant['deaths'],
                    'assists': participant['assists'],
                }

            return general_info, participant_info

        except KeyError as e:
            print(f"Key error: {e}. Here's the entire response:")
            print(match_data)
            return {}
    

# Usage
user = User('MenuMaxiBestFlop', 'EUW', 'EUW1')
user.user_info()

match_type = "ranked"  # replace with the actual match type
match_count = 1  # replace with the actual match count
match_id_list = user.get_matches(match_type, match_count)

if match_id_list:
    match_info_list = []
    for match_id in match_id_list:
        match_info = user.get_match_info(match_id)
        if match_info:
            match_info_list.append(match_info)

    # Now match_info_list contains the specific information for all matches
    for match_info in match_info_list:
        print(match_info)
else:
    print("No matches found.")
