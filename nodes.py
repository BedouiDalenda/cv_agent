# nodes.py
"""Nœuds du graphe pour l'agent CV - Version corrigée avec debug"""
import os
from langchain_core.messages import AIMessage
from models import CVAgentState
from tools import (
    generate_sql_from_query, load_cv_content, check_cv_exists, 
    extract_cv_info, store_cv_data, execute_sql_query
)
from utils import is_pdf_path, format_search_results

def analyze_query_node(state: CVAgentState) -> CVAgentState:
    """Analyse le type de requête (chemin PDF vs phrase naturelle)"""
    query = state["query"].strip()
    
    print(f"🔍 Analyse de la requête: '{query}'")
    
    if is_pdf_path(query):
        state["query_type"] = "pdf_path"
        state["cv_path"] = query
        print(f"📄 Type détecté: PDF Path -> {query}")
    else:
        state["query_type"] = "natural_language"
        print(f"💬 Type détecté: Natural Language")
    
    return state

def load_cv_node(state: CVAgentState) -> CVAgentState:
    """Charge le CV depuis le fichier PDF"""
    print(f"📂 Chargement du CV: {state['cv_path']}")
    result = load_cv_content.invoke({"file_path": state["cv_path"]})
    
    if "error" in result:
        state["error"] = result["error"]
        print(f"❌ Erreur chargement: {result['error']}")
    else:
        state["cv_content"] = result["content"]
        state["cv_hash"] = result["hash"]
        print(f"✅ CV chargé avec succès. Hash: {result['hash'][:10]}...")
    
    return state

def check_existence_node(state: CVAgentState) -> CVAgentState:
    """Vérifie si le CV existe déjà"""
    print(f"🔄 Vérification existence CV avec hash: {state['cv_hash'][:10]}...")
    result = check_cv_exists.invoke({"cv_hash": state["cv_hash"]})
    
    if "error" in result:
        state["error"] = result["error"]
        print(f"❌ Erreur vérification: {result['error']}")
    else:
        state["cv_exists"] = result.get("exists", False)
        print(f"📋 CV existe: {state['cv_exists']}")
    
    return state

def extract_info_node(state: CVAgentState) -> CVAgentState:
    """Extrait les informations du CV (PROMPT 2)"""
    print("🧠 Extraction des informations du CV...")
    result = extract_cv_info.invoke({"cv_content": state["cv_content"]})
    
    if "error" in result:
        state["error"] = result["error"]
        print(f"❌ Erreur extraction: {result['error']}")
    else:
        state["cv_data"] = result["cv_data"]
        print(f"✅ Extraction terminée. Compétences trouvées: {len(result['cv_data'].get('competences', []))}")
    
    return state

def store_cv_node(state: CVAgentState) -> CVAgentState:
    """Stocke le CV dans les deux tables (PROMPT 3)"""
    print("💾 Stockage du CV en base...")
    result = store_cv_data.invoke({
        "cv_content": state["cv_content"],
        "cv_data": state["cv_data"],
        "cv_hash": state["cv_hash"],
        "filename": os.path.basename(state["cv_path"])
    })
    
    if "error" in result:
        state["error"] = result["error"]
        print(f"❌ Erreur stockage: {result['error']}")
    else:
        message = f"✅ CV stocké avec succès!\n"
        message += f"📄 ID: {result['cv_id']}\n"
        message += f"🧩 Chunks embeddings: {result['chunks_stored']}\n"
        message += f"📊 Données JSON: {'✓' if result['json_stored'] else '✗'}"
        
        if result.get("warning"):
            message += f"\n⚠️ {result['warning']}"
        
        print(f"✅ Stockage terminé: {result['cv_id']}")
        state["messages"].append(AIMessage(content=message))
    
    return state

def generate_sql_node(state: CVAgentState) -> CVAgentState:
    """Génère une requête SQL depuis la phrase naturelle (PROMPT 1)"""
    print(f"🔧 Génération SQL pour: '{state['query']}'")
    result = generate_sql_from_query.invoke({"natural_query": state["query"]})
    
    if "error" in result:
        state["error"] = result["error"]
        print(f"❌ Erreur génération SQL: {result['error']}")
    else:
        state["sql_query"] = result["sql_query"]
        print(f"📝 SQL généré: {result['sql_query']}")
    
    return state

def execute_sql_node(state: CVAgentState) -> CVAgentState:
    """Exécute la requête SQL et retourne les résultats"""
    print(f"⚡ Exécution de la requête SQL...")
    result = execute_sql_query.invoke({"sql_query": state["sql_query"]})
    
    if "error" in result:
        state["error"] = result["error"]
        print(f"❌ Erreur exécution SQL: {result['error']}")
    else:
        state["search_results"] = result["results"]
        print(f"📊 Résultats trouvés: {len(result['results'])}")
        
        # Formater la réponse
        response = format_search_results(result["results"])
        response += f"\n🔍 Requête SQL exécutée: {state['sql_query']}"
        
        # Ajouter le message AI correctement
        state["messages"].append(AIMessage(content=response))
    
    return state

def error_node(state: CVAgentState) -> CVAgentState:
    """Gère les erreurs"""
    error_msg = f"❌ Erreur: {state.get('error', 'Erreur inconnue')}"
    print(error_msg)
    state["messages"].append(AIMessage(content=error_msg))
    return state

def already_exists_node(state: CVAgentState) -> CVAgentState:
    """Quand le CV existe déjà"""
    exists_msg = "⚠️ Ce CV existe déjà dans la base de données!"
    print(exists_msg)
    state["messages"].append(AIMessage(content=exists_msg))
    return state