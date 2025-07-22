# graph.py
"""Construction du graphe LangGraph pour l'agent CV - Version corrigée"""
from langgraph.graph import StateGraph, END
from models import CVAgentState
from nodes import (
    analyze_query_node, load_cv_node, check_existence_node,
    extract_info_node, store_cv_node, generate_sql_node,
    execute_sql_node, error_node, already_exists_node
)

def should_process_pdf_or_sql(state: CVAgentState) -> str:
    """Détermine le flux selon le type de requête"""
    if state["query_type"] == "pdf_path":
        return "load_cv"
    else:
        return "generate_sql"

def should_check_existence(state: CVAgentState) -> str:
    """Vérifie s'il faut continuer après le chargement"""
    if state.get("error"):
        return "error"
    return "check_existence"

def should_extract_or_exists(state: CVAgentState) -> str:
    """Détermine si extraire ou signaler l'existence"""
    if state.get("error"):
        return "error"
    elif state.get("cv_exists"):
        return "already_exists"
    else:
        return "extract_info"

def should_store(state: CVAgentState) -> str:
    """Détermine si stocker après extraction"""
    if state.get("error"):
        return "error"
    return "store_cv"

def should_execute_sql(state: CVAgentState) -> str:
    """Détermine si exécuter le SQL généré"""
    if state.get("error"):
        return "error"
    return "execute_sql"

def create_cv_agent_graph():
    """Crée le graphe de l'agent CV avec 3 prompts distincts"""
    
    workflow = StateGraph(CVAgentState)
    
    # Ajouter tous les nœuds
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("load_cv", load_cv_node)
    workflow.add_node("check_existence", check_existence_node)
    workflow.add_node("extract_info", extract_info_node)
    workflow.add_node("store_cv", store_cv_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("error", error_node)
    workflow.add_node("already_exists", already_exists_node)
    
    # Point d'entrée
    workflow.set_entry_point("analyze_query")
    
    # Flux principal: PDF ou SQL selon le type de requête
    workflow.add_conditional_edges(
        "analyze_query",
        should_process_pdf_or_sql,
        {
            "load_cv": "load_cv",
            "generate_sql": "generate_sql"
        }
    )
    
    # Flux PDF: load -> check -> extract -> store
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
        should_extract_or_exists,
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
    
    # Flux SQL: generate -> execute
    workflow.add_conditional_edges(
        "generate_sql",
        should_execute_sql,
        {
            "execute_sql": "execute_sql",
            "error": "error"
        }
    )
    
    # Tous les nœuds finaux
    workflow.add_edge("store_cv", END)
    workflow.add_edge("execute_sql", END)
    workflow.add_edge("error", END)
    workflow.add_edge("already_exists", END)
    
    return workflow.compile()
