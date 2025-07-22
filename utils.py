"""Fonctions utilitaires pour l'agent CV"""
import os
import hashlib
from typing import Dict, Any, List, Optional
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

def is_pdf_path(query: str) -> bool:
    """Vérifie si le query est un chemin vers un fichier PDF"""
    return query.strip().lower().endswith('.pdf') and ('/' in query or '\\' in query or os.path.exists(query))

def format_search_results(results: List[Dict]) -> str:
    """Formate les résultats de recherche SQL"""
    if not results:
        return "❌ Aucun résultat trouvé pour votre recherche."
    
    response = f"🔍 Trouvé {len(results)} résultat(s):\n\n"
    
    for i, res in enumerate(results, 1):
        nom = res.get("nom", "N/A")
        prenom = res.get("prenom", "N/A")
        response += f"{i}. {prenom} {nom}\n"
        
        if res.get("competences"):
            competences = res["competences"][:3] if len(res["competences"]) > 3 else res["competences"]
            response += f"   Compétences: {', '.join(competences)}\n"
        
        if res.get("resume_professionnel"):
            response += f"   Résumé: {res['resume_professionnel'][:100]}...\n"
        
        response += "\n"
    
    return response