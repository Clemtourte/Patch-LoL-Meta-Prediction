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
    
    def get_summoner_id(self):
        url = f"https://{self.region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{self.puuid}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()['id']
            except KeyError:
                logging.error("Key 'id' not found in response. Response: %s", response.json())
        else:
            logging.error("API request failed with status code %s", response.status_code, response.json())
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
            logging.error("API request failed with status code %s", response.status_code, response.json())
        return []

    def store_user_info(self, conn):
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
                self.save_to_db(conn, tier, rank, league_points)
                print(f"Tier: {tier} {rank}, League Points: {league_points}")
                return tier, rank, league_points
        print("No ranked data found for the summoner.")

    def save_to_db(self, conn, tier, rank, league_points):
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, tag, region, puuid, tier, rank, league_points)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (self.username, self.tag, self.region, self.puuid, tier, rank, league_points))
        conn.commit()

    def get_matches(self, match_type, match_count):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{self.puuid}/ids?type={match_type}&start=0&count={match_count}&api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error("API request failed with status code %s", response.status_code, response.json())
        return []

    def get_match_data(self, match_id):
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={self.API_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error("API request failed with status code %s", response.status_code, response.json())
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
            logging.error("Key error: %s. Here's the entire response: %s", e, match_data)
            return {}
