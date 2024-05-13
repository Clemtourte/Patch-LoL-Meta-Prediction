import requests

class User:
    API_key = "RGAPI-ce843069-ebfd-4778-af0c-45840713c8a2"

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
        participants = match_data['info']['participants']
        teams = match_data['info']['teams']
        timeline = match_data['info']['timeline']

        return {
            'game_id': game_id,
            'game_duration':game_duration,
            'game_version': game_version,
            'participants': participants,
            'teams': teams,
            'timeline': timeline,
        }

user = User('MenuMaxiBestFlop','EUW')
print(user.puuid)

match_type = "ranked"  # replace with the actual match type
match_count = 50  # replace with the actual match count
match_id_list = user.get_matches(match_type, match_count)
print(match_id_list)

if match_id_list:
    match_data = user.get_match_data(match_id_list[0])
    print(match_data)