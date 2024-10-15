import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sqlite3
import pandas as pd

conn = sqlite3.connect('"sqlite:///../datasets/league_data.db"')

query = "SELECT champion_name, champ_level, role, position, kills, deaths, assists, kda, gold_earned, total_damage_dealt, cs FROM participants;"
df = pd.read_sql_query(query, conn)
conn.close()

le = LabelEncoder()
df['role'] = le.fit_transform(df['role'])
df['champion_name'] = le.fit_transform(df['champion_name'])
print(df.head())

X = df.drop('position', axis=1)
y = df['position']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled,y_train)

y_pred = rf_classifier.predict(X_test_scaled)

print(classification_report(y_test, y_pred))