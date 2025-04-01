from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///../datasets/league_data.db")

with engine.connect() as conn:
    # Construire l'objet texte
    statement = text("""
        DELETE FROM patch_changes
        WHERE change_value < -1500 OR change_value > 2000
    """)

    # Exécuter la requête
    conn.execute(statement)

    # Commit si nécessaire (souvent on utilise plutôt `begin()`, voir ci-dessous)
    conn.commit()
