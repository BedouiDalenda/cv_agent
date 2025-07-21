"""Construction du graphe LangGraph pour l'agent CV"""

from langgraph.graph import StateGraph
from models import CVAgentState
from nodes import (
    analyze_request_node, load_cv_node, check_existence_node, 
    extract_info_node, store_cv_node, search_node, 
    general_response_node, error_node, already_exists_node
)

def should_load_cv(state: CVAgentState) -> str:
    """Détermine si on doit charger un CV"""
    if state["action_needed"] == "store_cv" and state.get("cv_path"):
        return "load_cv"
    elif state["action_needed"] == "search_cv":
        return "search"
    else:
        return "general_response"

def should_check_existence(state: CVAgentState) -> str:
    """Détermine si on doit vérifier l'existence"""
    if state.get("error"):
        return "error"
    return "check_existence"

def should_extract_or_store(state: CVAgentState) -> str:
    """Détermine si on doit extraire ou stocker"""
    if state.get("error"):
        return "error"
    elif state.get("cv_exists"):
        return "already_exists"
    else:
        return "extract_info"

def should_store(state: CVAgentState) -> str:
    """Détermine si on doit stocker"""
    if state.get("error"):
        return "error"
    return "store_cv"

def create_cv_agent_graph():
    """Crée le graph de l'agent CV"""
    
    # Initialiser le graph
    workflow = StateGraph(CVAgentState)
    
    # Ajouter les nodes
    workflow.add_node("analyze_request", analyze_request_node)
    workflow.add_node("load_cv", load_cv_node)
    workflow.add_node("check_existence", check_existence_node)
    workflow.add_node("extract_info", extract_info_node)
    workflow.add_node("store_cv", store_cv_node)
    workflow.add_node("search", search_node)
    workflow.add_node("general_response", general_response_node)
    workflow.add_node("error", error_node)
    workflow.add_node("already_exists", already_exists_node)
    
    # Point d'entrée
    workflow.set_entry_point("analyze_request")
    
    # Edges conditionnels
    workflow.add_conditional_edges(
        "analyze_request",
        should_load_cv,
        {
            "load_cv": "load_cv",
            "search": "search", 
            "general_response": "general_response"
        }
    )
    
    workflow.add_conditional_edges(
        "load_cv",
        should_check_existence,
        {
            "check_existence": "check_existence",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "check_existence", 
        should_extract_or_store,
        {
            "extract_info": "extract_info",
            "already_exists": "already_exists",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "extract_info",
        should_store,
        {
            "store_cv": "store_cv",
            "error": "error"
        }
    )
    
    # Edges finaux
    workflow.add_edge("store_cv", "__end__")
    workflow.add_edge("search", "__end__")
    workflow.add_edge("general_response", "__end__") 
    workflow.add_edge("error", "__end__")
    workflow.add_edge("already_exists", "__end__")
    
    return workflow.compile()