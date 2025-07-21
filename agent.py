"""Classe principale de l'agent CV"""

from typing import Dict, Any
import psycopg2
from langchain_core.messages import HumanMessage

from models import CVAgentState
from graph import create_cv_agent_graph
from config import PSYCOPG_STRING


class CVAgent:
    """Agent CV principal"""
    
    def __init__(self):
        self.graph = create_cv_agent_graph()
        self.setup_database()
    
    def setup_database(self):
        """Configure la base de données"""
        try:
            with psycopg2.connect(PSYCOPG_STRING) as conn:
                with conn.cursor() as cur:
                    # Créer les extensions nécessaires
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
                    conn.commit()
            print("✅ Base de données configurée")
        except Exception as e:
            print(f"❌ Erreur configuration DB: {e}")
    
    def process_cv(self, cv_path: str) -> Dict[str, Any]:
        """Traite un CV complet"""
        initial_state = CVAgentState(
            messages=[{"role": "user", "content": f"Ajouter le CV: {cv_path}"}],
            cv_path=cv_path,
            cv_content=None,
            cv_hash=None,
            cv_exists=False,
            cv_data=None,
            query=None,
            search_results=None,
            action_needed="",
            error=None
        )
        
        result = self.graph.invoke(initial_state)
        return result
    
    def search_cvs(self, query: str) -> Dict[str, Any]:
        """Recherche dans les CV"""
        initial_state = CVAgentState(
            messages=[{"role": "user", "content": f"Recherche: {query}"}],
            cv_path=None,
            cv_content=None,
            cv_hash=None,
            cv_exists=False,
            cv_data=None,
            query=query,
            search_results=None,
            action_needed="",
            error=None
        )
        
        result = self.graph.invoke(initial_state)
        return result
    
    def chat(self, message: str) -> str:
        """Chat général avec l'agent"""
        try:
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "cv_path": None,
                "analysis_result": None
            }
            
            result = self.graph.invoke(initial_state)
            
            # Filtrer les messages AI et retourner le dernier contenu
            ai_messages = [msg for msg in result["messages"] if hasattr(msg, 'type') and msg.type == "ai"]
            
            if ai_messages:
                return ai_messages[-1].content  # Utiliser .content, pas ["content"]
            else:
                return "Aucune réponse générée."
                
        except Exception as e:
            return f"Erreur: {str(e)}"