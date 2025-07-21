"""Fonctions utilitaires pour l'agent CV"""
import os
import json
import hashlib
from typing import Dict, Any, List, Optional, TypedDict, Annotated
import re 

def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les caractères indésirables"""
    text = text.replace('\x00', '')
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_cv_hash(content: str) -> str:
    """Génère un hash unique pour le CV"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def format_search_results(results: List[Dict], total: int) -> str:
    """Formate les résultats de recherche pour l'affichage"""
    if not results:
        return "❌ Aucun résultat trouvé pour votre recherche."
    
    response = f"🔍 Trouvé {total} résultat(s):\n\n"
    
    for i, res in enumerate(results[:3], 1):
        if res["type"] == "json":
            nom = res["info"].get("nom", "N/A")
            prenom = res["info"].get("prenom", "N/A") 
            response += f"{i}. {prenom} {nom} - {res['filename']}\n"
            competences = res['competences'][:3] if res['competences'] else []
            response += f"   Compétences: {', '.join(competences)}...\n\n"
        else:
            response += f"{i}. Contenu: {res['content']}\n\n"
    
    return response