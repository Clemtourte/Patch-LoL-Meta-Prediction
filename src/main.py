import requests
import os
from dotenv import load_dotenv

class User:
    load_dotenv()
    API_key = os.getenv('API_KEY')

    def __init__(self, username, tag):
        self.username = username
        self.tag = tag
        self.puuid = self.get_puuid()

    def get_puuid(self):
        response = requests.get(f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{self.username}/{self.tag}?api_key={self.API_key}")
        if response.status_code == 200:
            try:
                return response.json()['puuid']
            except KeyError:
                print("Key 'puuid' not found in the response. Here's the entire response:")
                print(response.json())
        else:
            print(f"API request failed with status code {response.status_code}. Here's the entire response:")
            print(response.json())

    def get_matches(self, match_type, match_count):
        response = requests.get(f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?type={match_type}&start=0&count={match_count}&api_key={self.API_key}")
        return response.json()
    
    def get_match_data(self,match_id):
        response = requests.get(f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}")
        return response.json()
    
    def get_match_info(self, match_id):
        match_data = self.get_match_data(match_id)
        game_id = match_data['metadata']['matchId']
        game_duration = match_data['info']['gameDuration']
        game_version = match_data['info']['gameVersion']

        general_info = {
            'game_id': game_id,
            'game_duration': round(game_duration/60,2),
            'patch': game_version[:2],
        }

        participant_info = {}
        for i,participant in enumerate(match_data['info']['participants'],1):
                participant_info[f'Summoner {i}'] = {
                    'summoner_id' : participant['puuid'],
                    'summoner_name' : participant['summonerName'],
                    'team' : participant['teamId'],
                    'champ_level' : participant['champLevel'],
                    'champ_name' : participant['championName'],
                    'champ_id' : participant['championId'],
                    'kills' : participant['kills'],
                    'deaths' : participant['deaths'],
                    'assists' : participant['assists'],
                }

        return general_info, participant_info

user = User('MenuMaxiBestFlop','EUW')

match_type = "ranked"  # replace with the actual match type
match_count = 1  # replace with the actual match count
match_id_list = user.get_matches(match_type, match_count)
print(match_id_list)

match_info_list = []
for match_id in match_id_list:
    match_info = user.get_match_info(match_id)
    match_info_list.append(match_info)

# Now match_info_list contains the specific information for all matches
for match_info in match_info_list:
    print(match_info)