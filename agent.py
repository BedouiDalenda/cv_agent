"""Classe principale de l'agent CV"""
import os 
from typing import Dict, Any
import psycopg2
from langchain_core.messages import HumanMessage, AIMessage

from models import CVAgentState
from graph import create_cv_agent_graph
from config import PSYCOPG_STRING


class CVAgent:
    def __init__(self):
        self.graph = create_cv_agent_graph()
        self.setup_database()

    def setup_database(self):
        """Configuration DB optimisée avec plus d'index"""
        try:
            with psycopg2.connect(PSYCOPG_STRING) as conn:
                with conn.cursor() as cur:
                    # Index optimisés
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cv_fulltext 
                        ON cv_extraits USING gin(to_tsvector('french', contenu_complet))
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cv_competences_gin
                        ON cv_extraits USING gin(competences)
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cv_nom_prenom
                        ON cv_extraits ((informations_personnelles->>'nom'), (informations_personnelles->>'prenom'))
                    """)
                    conn.commit()
            print("✅ Base optimisée avec index avancés")
        except Exception as e:
            print(f"❌ Erreur DB: {e}")

    def chat(self, message: str) -> str:
        """Chat intelligent avec gestion d'erreurs améliorée"""
        try:
            message = message.strip()
            if not message:
                return "❓ Veuillez poser une question ou donner une instruction."
            
            print(f"🎯 Requête: {message}")
            
            # Détection automatique du type de requête
            if message.lower().startswith("ajouter ") and ("\\" in message or "/" in message):
                file_path = message[8:].strip()
                result = self.process_cv(file_path)
                return result.get("response", "Erreur de traitement")
            
            initial_state = CVAgentState(
                messages=[HumanMessage(content=message)],
                cv_path=None,
                action_needed="",
                filter_type="all",  # Valeur par défaut
                search_type="mixed"
            )
            
            result = self.graph.invoke(initial_state)
            
            # Gestion des erreurs spécifiques
            if "error" in result:
                return f"❌ Erreur: {result['error']}"
                
            return result.get("response", "Réponse vide")
            
        except Exception as e:
            print(f"❌ Erreur chat: {e}")
            return "🤖 Désolé, une erreur est survenue. Veuillez reformuler votre demande."

    def process_cv(self, cv_path: str) -> Dict[str, Any]:
        """Traitement de CV avec feedback détaillé"""
        try:
            # Normalisation du chemin
            cv_path = os.path.abspath(cv_path.replace('"', '').strip())
            
            if not os.path.exists(cv_path):
                return {"error": f"Fichier introuvable: {cv_path}"}
                
            print(f"📁 Traitement de: {cv_path}")
            
            initial_state = CVAgentState(
                messages=[HumanMessage(content=f"Traiter le CV: {cv_path}")],
                cv_path=cv_path,
                action_needed="process_cv",
                filter_type="all"  # Ajout de la valeur par défaut
            )
            
            result = self.graph.invoke(initial_state)
            return result
            
        except Exception as e:
            return {"error": f"Erreur traitement CV: {str(e)}"}


if __name__ == "__main__":
    agent = CVAgent()
    
    tests = {
        "Liste": "liste tous les cvs",
        "Recherche": "cherche développeur python senior",
        "Comptage": "combien de cv dans la base",
        "Similarité": "profil similaire à développeur web",
        "Ajout CV": r"ajouter C:\Users\ASUS\Downloads\Dorra_CV.pdf",
        "Expérience": "cv avec 5 ans experience java",
        "Vide": ""
    }
    
    for name, test in tests.items():
        print(f"\n{'='*50}")
        print(f"TEST {name.upper()}: '{test}'")
        print('='*50)
        try:
            response = agent.chat(test)
            print(response)
        except Exception as e:
            print(f"❌ Erreur critique: {str(e)}")