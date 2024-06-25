import requests
import os
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class User:
    def __init__(self, username, tag, region):
        load_dotenv()  # Load environment variables from .env file
        self.API_key = os.getenv('API_KEY')
        self.username = username
        self.tag = tag
        self.region = region
        self.puuid = self.get_puuid()
        print(f"Username: {username}")

    def get_puuid(self):
        url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{self.username}/{self.tag}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()['puuid']
            except KeyError:
                logging.error("Key 'puuid' not found in response. Response: %s", response.json())
        else:
            logging.error("API request failed with status code %s. Response: %s", response.status_code, response.json())
        return None
    
    def get_summoner_id(self):
        url = f"https://{self.region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{self.puuid}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()['id']
            except KeyError:
                logging.error("Key 'id' not found in response. Response: %s", response.json())
        else:
            logging.error("API request failed with status code %s. Response: %s", response.status_code, response.json())
        return None

    def get_ranked_data(self, summoner_id):
        url = f"https://{self.region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()
            except Exception as e:
                logging.error("Error parsing response: %s. Response: %s", e, response.text)
        else:
            logging.error("API request failed with status code %s. Response: %s", response.status_code, response.json())
        return []

    def display_user_info(self):
        summoner_id = self.get_summoner_id()
        if not summoner_id:
            logging.error("Failed to get summoner ID.")
            return

        ranked_data = self.get_ranked_data(summoner_id)
        for entry in ranked_data:
            if entry.get('queueType') == 'RANKED_SOLO_5x5':
                tier = entry['tier']
                rank = entry['rank']
                league_points = entry['leaguePoints']
                print(f"Tier: {tier} {rank}, League Points: {league_points}")
                return tier, rank, league_points
        print("No ranked data found for the summoner.")

    def get_matches(self, match_type, match_count):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?type={match_type}&start=0&count={match_count}&api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error("API request failed with status code %s. Response: %s", response.status_code, response.json())
        return []

    def get_match_data(self, match_id):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error("API request failed with status code %s. Response: %s", response.status_code, response.json())
        return {}

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

            general_info = {
                'game_id': game_id,
                'game_duration': round(game_duration / 60, 2),
                'patch': game_version.split('.'),
                'mode' : match_data['info']['gameMode']
            }

            participant_info = {'Blue Side': {}, 'Red Side': {}}
            for i, participant in enumerate(match_data['info']['participants'], 1):
                team = 'Blue Side' if participant['teamId'] == 100 else 'Red Side'
                participant_info[team][f'Summoner {i}'] = {
                    'summoner_id': participant['puuid'],
                    'summoner_name': participant['summonerName'],
                    'team': participant['teamId'],
                    'win': participant['win'],
                    'champ_level': participant['champLevel'],
                    'champ_name': participant['championName'],
                    'champ_id': participant['championId'],
                    'kills': participant['kills'],
                    'deaths': participant['deaths'],
                    'assists': participant['assists'],
                }

            return general_info, participant_info

        except KeyError as e:
            logging.error("Key error: %s. Here's the entire response: %s", e, match_data)
            return None, None

if __name__ == '__main__':
    user = User('MenuMaxiBestFlop', 'EUW', 'EUW1')
    user.display_user_info()
    
    match_ids = user.get_matches('ranked', 20)
    all_participants_info = []
    for match_id in match_ids:
        general_info, participant_info = user.get_match_info(match_id)
        if general_info is None and participant_info is None :
            continue
        else:
            print(f"Match ID: {general_info['game_id']}")
            print(f"Game Duration: {general_info['game_duration']} minutes")
            print(f"Game Mode: {general_info['mode']}")
            print(f"Patch: {'.'.join(general_info['patch'])}")
            print("Participants Info:")
            for team, participants in participant_info.items():
                print(team)
                for summoner, details in participants.items():
                    print(f"{summoner}: {details}")
            print("\n")
            all_participants_info.append(participant_info)
        with open("../datasets/match_data.json", "w") as out_file:
            json.dump(all_participants_info, out_file, indent=4)