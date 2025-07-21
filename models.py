"""Modèles de données pour l'agent CV"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from langgraph.graph.message import add_messages

class CVAgentState(TypedDict):
    """État de l'agent CV"""
    messages: Annotated[List[Dict], add_messages]
    cv_path: Optional[str]
    cv_content: Optional[str] 
    cv_hash: Optional[str]
    cv_exists: bool
    cv_data: Optional[Dict[str, Any]]
    query: Optional[str]
    search_results: Optional[List[Dict]]
    action_needed: str
    error: Optional[str]

@dataclass
class CVInfo:
    """Structure des informations extraites du CV"""
    informations_personnelles: Dict[str, str]
    experiences: List[Dict[str, str]]
    formations: List[Dict[str, str]]
    competences: List[str]
    langues: List[str]
    resume: str