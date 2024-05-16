import requests

class User:
    API_key = "RGAPI-1c59560a-c466-4cfc-877b-7f54c078b8c3"

    def __init__(self, username, tag):
        self.username = username
        self.tag = tag
        self.puuid = self.get_puuid()

    def get_puuid(self):
        response = requests.get(f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{self.username}/{self.tag}?api_key={self.API_key}")
        return response.json()['puuid']
    
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
        participant_id = match_data['info']['participants'][0]['puuid']
        team = match_data['info']['participants'][0]['teamId']
        champ_level = match_data['info']['participants'][0]['champLevel']

        return {
            'game_id': game_id,
            'game_duration':round(game_duration/60,2),
            'patch': game_version[:2],
            'participant_id': participant_id,
            'team': team,
            'champ_level': champ_level,
        }

user = User('MenuMaxiBestFlop','EUW')
print(user.puuid)

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