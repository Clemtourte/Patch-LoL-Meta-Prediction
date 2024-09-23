from sqlalchemy import create_engine, Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'
    match_id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, unique=True, nullable=False)
    game_duration = Column(String, nullable=False)
    patch = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    queue_id = Column(Integer, nullable=False,default=0)
    platform = Column(String, nullable=False)

    teams = relationship("Team", back_populates="match")

class Team(Base):
    __tablename__ = 'teams'
    team_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'), nullable=False)
    team_name = Column(String, nullable=False)
    win = Column(Boolean, nullable=False)

    match = relationship("Match", back_populates="teams")
    participants = relationship("Participant", back_populates="team")

class Participant(Base):
    __tablename__ = 'participants'
    participant_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    summoner_id = Column(String, nullable=False)
    summoner_name = Column(String, nullable=False)
    champion_name = Column(String, nullable=False)
    champion_id = Column(Integer, nullable=False)
    champ_level = Column(Integer, nullable=False)
    role = Column(String, nullable=False)
    lane = Column(String, nullable=False)
    position = Column(String, nullable=False)
    kills = Column(Integer, nullable=False)
    deaths = Column(Integer, nullable=False)
    assists = Column(Integer, nullable=False)
    kda = Column(String, nullable=False)
    gold_earned = Column(Integer, nullable=False)
    total_damage_dealt = Column(Integer, nullable=False)
    cs = Column(Integer, nullable=False)

    team = relationship("Team", back_populates="participants")

def init_db(uri="sqlite:///matches.db"):
    engine = create_engine(uri)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
