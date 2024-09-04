import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sqlite3
import pandas as pd

conn = sqlite3.connect('matches.db')

query = "SELECT champion_id, champ_level, role, kills, deaths, assists, gold_earned, total_damage_dealt, cs FROM participants;"
df = pd.read_sql_query(query, conn)

conn.close()

le = LabelEncoder()
df['role'] = le.fit_transform(df['role'])
print(df.head())

