"""Nœuds du graphe pour l'agent CV"""

import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from models import CVAgentState
from tools import load_cv_content, check_cv_exists, extract_cv_info, store_cv_embeddings, search_cvs
from config import LLM_MODEL
from utils import format_search_results

def analyze_request_node(state: CVAgentState) -> CVAgentState:
    """Analyse la demande utilisateur"""
    last_message = state["messages"][-1].content
    
    # Déterminer l'action nécessaire
    if "cv" in last_message.lower() and ("ajouter" in last_message.lower() or "stocker" in last_message.lower()):
        state["action_needed"] = "store_cv"
    elif "recherche" in last_message.lower() or "cherche" in last_message.lower():
        state["action_needed"] = "search_cv" 
        state["query"] = last_message
    else:
        state["action_needed"] = "general_query"
        state["query"] = last_message
    
    return state

def load_cv_node(state: CVAgentState) -> CVAgentState:
    """Node pour charger un CV"""
    if state["cv_path"]:
        result = load_cv_content.invoke({"file_path": state["cv_path"]})
        
        if "error" in result:
            state["error"] = result["error"]
        else:
            state["cv_content"] = result["content"]
            state["cv_hash"] = result["hash"]
    
    return state

def check_existence_node(state: CVAgentState) -> CVAgentState:
    """Node pour vérifier l'existence du CV"""
    if state["cv_hash"]:
        result = check_cv_exists.invoke({"cv_hash": state["cv_hash"]})
        
        if "error" in result:
            state["error"] = result["error"]
        else:
            state["cv_exists"] = result.get("exists", False)
    
    return state

def extract_info_node(state: CVAgentState) -> CVAgentState:
    """Node pour extraire les informations du CV"""
    if state["cv_content"] and not state["cv_exists"]:
        result = extract_cv_info.invoke({"cv_content": state["cv_content"]})
        
        if "error" in result:
            state["error"] = result["error"]
        else:
            state["cv_data"] = result["cv_data"]
    
    return state

def store_cv_node(state: CVAgentState) -> CVAgentState:
    """Node pour stocker le CV"""
    if state["cv_content"] and state["cv_data"] and not state["cv_exists"]:
        result = store_cv_embeddings.invoke({
            "cv_content": state["cv_content"],
            "cv_data": state["cv_data"], 
            "cv_hash": state["cv_hash"],
            "filename": os.path.basename(state["cv_path"] or "unknown.pdf")
        })
        
        if "error" in result:
            state["error"] = result["error"]
        else:
            state["messages"].append({
                "role": "assistant",
                "content": f"✅ CV stocké avec succès! ID: {result['cv_id']}, Chunks: {result['chunks_stored']}"
            })
    
    return state

def search_node(state: CVAgentState) -> CVAgentState:
    """Node pour rechercher dans les CV"""
    if state["query"]:
        result = search_cvs.invoke({
            "query": state["query"],
            "search_type": "mixed"
        })
        
        if "error" in result:
            state["error"] = result["error"]
        else:
            state["search_results"] = result["results"]
            
            # Formater la réponse
            response = format_search_results(result["results"], result["total"])
            
            state["messages"].append({
                "role": "assistant", 
                "content": response
            })
    
    return state

def general_response_node(state: CVAgentState) -> CVAgentState:
    """Node pour les réponses générales"""
    llm = ChatMistralAI(model=LLM_MODEL, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("""
    Tu es un assistant spécialisé dans la gestion de CV. 
    
    Question: {query}
    
    Réponds de manière utile en expliquant ce que tu peux faire:
    - Stocker des CV (avec embeddings et JSON)
    - Rechercher dans la base de CV
    - Analyser les compétences et profils
    """)
    
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})
    
    state["messages"].append({
        "role": "assistant",
        "content": response.content
    })
    
    return state

def error_node(state: CVAgentState) -> CVAgentState:
    """Node pour gérer les erreurs"""
    state["messages"].append({
        "role": "assistant",
        "content": f"❌ Erreur: {state.get('error', 'Erreur inconnue')}"
    })
    return state

def already_exists_node(state: CVAgentState) -> CVAgentState:
    """Node quand le CV existe déjà"""
    state["messages"].append({
        "role": "assistant",
        "content": "⚠️ Ce CV existe déjà dans la base de données!"
    })
    return state