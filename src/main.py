import requests

API_key = "RGAPI-76700085-3109-4f8d-a43a-c71acdab0c5d"
Username = "Tourtipouss"
Tag = "9861"


user = requests.get("https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/"+Username+"/"+Tag+"?api_key="+API_key).json()
print(user)