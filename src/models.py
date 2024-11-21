from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, ForeignKey, JSON
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
    queue_id = Column(Integer, nullable=False, default=0)
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
    kda = Column(Float, nullable=False)
    gold_earned = Column(Integer, nullable=False)
    total_damage_dealt = Column(Integer, nullable=False)
    cs = Column(Integer, nullable=False)
    total_heal = Column(Integer, nullable=True)
    damage_taken = Column(Integer, nullable=True)
    damage_mitigated = Column(Integer, nullable=True)
    wards_placed = Column(Integer, nullable=True)
    wards_killed = Column(Integer, nullable=True)
    time_ccing_others = Column(Integer, nullable=True)
    xp = Column(Integer, nullable=True)
    performance_score = Column(Float, nullable=True)
    standardized_performance_score = Column(Float, nullable=True)

    team = relationship("Team", back_populates="participants")
    performance_features = relationship("PerformanceFeatures", back_populates="participant", uselist=False)

class PerformanceFeatures(Base):
    __tablename__ = 'performance_features'
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    participant_id = Column(Integer, ForeignKey('participants.participant_id'), nullable=False, unique=True)
    kill_participation = Column(Float, nullable=True)
    death_share = Column(Float, nullable=True)
    damage_share = Column(Float, nullable=True)
    damage_taken_share = Column(Float, nullable=True)
    gold_share = Column(Float, nullable=True)
    heal_share = Column(Float, nullable=True)
    damage_mitigated_share = Column(Float, nullable=True)
    cs_share = Column(Float, nullable=True)
    vision_share = Column(Float, nullable=True)
    vision_denial_share = Column(Float, nullable=True)
    xp_share = Column(Float, nullable=True)
    cc_share = Column(Float, nullable=True)
    champion_role_patch = Column(String, nullable=True)

    participant = relationship("Participant", back_populates="performance_features")

class ChampionStats(Base):
    __tablename__ = 'champion_stats'
    version = Column(String, primary_key=True)
    champion = Column(String, primary_key=True)
    hp = Column(Float)
    mp = Column(Float)
    armor = Column(Float)
    spellblock = Column(Float)
    attackdamage = Column(Float)
    attackspeed = Column(Float)
    hpperlevel = Column(Float)
    mpperlevel = Column(Float)
    armorperlevel = Column(Float)
    spellblockperlevel = Column(Float)
    attackdamageperlevel = Column(Float)
    attackspeedperlevel = Column(Float)
    attackrange = Column(Float)
    movespeed = Column(Float)
    crit = Column(Float)
    critperlevel = Column(Float)

class SpellStats(Base):
    __tablename__ = 'spell_stats'
    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)
    champion = Column(String, nullable=False)
    spell_id = Column(String, nullable=False)
    spell_name = Column(String, nullable=False)
    damage_type = Column(String)
    damage_values = Column(JSON)
    max_rank = Column(Integer)
    cooldown = Column(JSON)
    cost = Column(JSON)
    range = Column(JSON)
    resource = Column(String)
    description = Column(String)
    is_passive = Column(Boolean, default=False)

class ItemStats(Base):
    __tablename__ = 'item_stats'
    version = Column(String, primary_key=True)
    item_id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    plaintext = Column(String)
    total_gold = Column(Integer)
    base_gold = Column(Integer)
    sell_gold = Column(Integer)
    purchasable = Column(Boolean)
    tags = Column(String)

def init_db(uri="sqlite:///../datasets/league_data.db"):
    engine = create_engine(uri)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)