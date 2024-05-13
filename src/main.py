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

user = User('MenuMaxiBestFlop','EUW')
print(user.puuid)

match_type = "ranked"  # replace with the actual match type
match_count = 50  # replace with the actual match count
match_id_list = user.get_matches(match_type, match_count)
print(match_id_list)