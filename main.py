# main.py
"""Point d'entrée principal pour l'agent CV - Version corrigée"""
import os
from langchain_core.messages import HumanMessage, AIMessage
from models import CVAgentState
from graph import create_cv_agent_graph

def run_cv_agent(query: str) -> str:
    """Exécute l'agent CV avec une requête"""
    
    print(f"🚀 Démarrage agent pour: '{query}'")
    
    # Créer le graphe
    app = create_cv_agent_graph()
    
    # État initial avec le bon format de message
    initial_state = CVAgentState(
        messages=[HumanMessage(content=query)],  # Utiliser HumanMessage directement
        query=query,
        query_type="",
        cv_path=None,
        cv_content=None,
        cv_hash=None,
        cv_exists=False,
        cv_data=None,
        sql_query=None,
        search_results=None,
        error=None
    )
    
    # Exécuter le graphe
    try:
        print("⚡ Exécution du graphe...")
        result = app.invoke(initial_state)
        
        print(f"📋 État final:")
        print(f"  - Query type: {result.get('query_type')}")
        print(f"  - SQL query: {result.get('sql_query')}")
        print(f"  - Error: {result.get('error')}")
        print(f"  - Messages count: {len(result.get('messages', []))}")
        
        # Retourner la dernière réponse de l'assistant
        assistant_messages = []
        for i, msg in enumerate(result["messages"]):
            print(f"  Message {i}: {type(msg).__name__}")
            # Vérifier si c'est un message AI
            if hasattr(msg, 'content') and isinstance(msg, AIMessage):
                assistant_messages.append(msg.content)
                print(f"    -> AI Content: {msg.content[:100]}...")
            elif hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                assistant_messages.append(msg.content)
                print(f"    -> AI Content (type): {msg.content[:100]}...")
        
        if assistant_messages:
            return assistant_messages[-1]
        else:
            return "Aucune réponse générée."
            
    except Exception as e:
        error_msg = f"❌ Erreur d'exécution: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    # Tests
    print("🤖 Agent CV démarré!\n")
    
    # Test: Autre recherche
    print("Test : lister le contenu")
    result3 = run_cv_agent("liste moi le contenu de la base ")
    print(result3)