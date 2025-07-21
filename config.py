"""Configuration et constantes pour l'agent CV"""

import os

# Configuration PostgreSQL
POSTGRES_CONFIG = {
    "user": "postgres",
    "password": "bedoui", 
    "db": "vectorCV",
    "host": "localhost",
    "port": "5433"
}

# Chaînes de connexion
CONNECTION_STRING = f"postgresql+psycopg://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['db']}"
PSYCOPG_STRING = f"dbname={POSTGRES_CONFIG['db']} user={POSTGRES_CONFIG['user']} password={POSTGRES_CONFIG['password']} host={POSTGRES_CONFIG['host']} port={POSTGRES_CONFIG['port']}"

# Configuration des modèles
MISTRAL_API_KEY = "AewXzQqG1LsGudzECjHbEQOL9D9fpD5Q"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral-small-2506"

# Configuration du text splitter
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Configuration de recherche
SEARCH_RESULTS_LIMIT = 5

# Définir les variables d'environnement
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY