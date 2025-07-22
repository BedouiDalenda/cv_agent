"""Modèles de données pour l'agent CV"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages

class CVAgentState(TypedDict):
    """État de l'agent CV"""
    messages: Annotated[List[Dict], add_messages]
    query: str
    query_type: str  # "pdf_path", "natural_language"
    cv_path: Optional[str]
    cv_content: Optional[str] 
    cv_hash: Optional[str]
    cv_exists: bool
    cv_data: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    search_results: Optional[List[Dict]]
    error: Optional[str]
