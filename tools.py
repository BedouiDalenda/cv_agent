"""Outils LangChain pour l'agent CV"""

import os
import json
from typing import Dict, Any, List

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.documents import Document

import psycopg2
from psycopg2.extras import RealDictCursor

from config import CONNECTION_STRING, PSYCOPG_STRING, LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_RESULTS_LIMIT
from utils import clean_text, generate_cv_hash

def ensure_tables_exist():
    """S'assure que toutes les tables nécessaires existent"""
    try:
        with psycopg2.connect(PSYCOPG_STRING) as conn:
            with conn.cursor() as cur:
                # Créer les extensions nécessaires
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
                
                # Créer la table cv_extraits
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cv_extraits (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        cv_hash VARCHAR(32) UNIQUE NOT NULL,
                        nom_fichier VARCHAR(255) NOT NULL,
                        date_traitement TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        informations_personnelles JSONB,
                        experiences JSONB,
                        formations JSONB, 
                        competences JSONB,
                        langues JSONB,
                        resume_professionnel TEXT,
                        contenu_complet TEXT
                    );
                """)
                
                # Créer les index
                cur.execute("CREATE INDEX IF NOT EXISTS idx_cv_hash ON cv_extraits (cv_hash);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_cv_competences ON cv_extraits USING GIN (competences);")
                
                conn.commit()
                print("✅ Tables créées avec succès")
    except Exception as e:
        print(f"❌ Erreur création tables: {e}")

@tool
def load_cv_content(file_path: str) -> Dict[str, Any]:
    """Charge le contenu d'un CV PDF"""
    try:
        if not os.path.exists(file_path):
            return {"error": f"Fichier non trouvé: {file_path}"}
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        content = "\n".join([clean_text(doc.page_content) for doc in documents])
        cv_hash = generate_cv_hash(content)
        
        return {
            "content": content,
            "hash": cv_hash,
            "filename": os.path.basename(file_path)
        }
    except Exception as e:
        return {"error": f"Erreur lors du chargement: {str(e)}"}

@tool 
def check_cv_exists(cv_hash: str) -> Dict[str, Any]:
    """Vérifie si un CV existe déjà dans la base"""
    try:
        # S'assurer que les tables existent
        ensure_tables_exist()
        
        with psycopg2.connect(PSYCOPG_STRING) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, nom_fichier FROM cv_extraits WHERE cv_hash = %s", 
                    (cv_hash,)
                )
                result = cur.fetchone()
                
                if result:
                    return {
                        "exists": True, 
                        "cv_id": str(result['id']),
                        "filename": result['nom_fichier']
                    }
                else:
                    return {"exists": False}
    except Exception as e:
        return {"error": f"Erreur base de données: {str(e)}"}

@tool
def extract_cv_info(cv_content: str) -> Dict[str, Any]:
    """Extrait les informations structurées du CV"""
    try:
        llm = ChatMistralAI(
            model=LLM_MODEL, 
            temperature=0.1,
            max_tokens=1000
        )
        
        prompt = ChatPromptTemplate.from_template("""
        Analyse ce CV et extrais les informations en format JSON:
        
        CV Content: {content}
        
        Retourne un JSON avec cette structure:
        {{
            "informations_personnelles": {{
                "nom": "...", "prenom": "...", "email": "...", 
                "telephone": "...", "adresse": "..."
            }},
            "experiences": [
                {{"poste": "...", "entreprise": "...", "duree": "...", "description": "..."}}
            ],
            "formations": [
                {{"diplome": "...", "etablissement": "...", "annee": "..."}}
            ],
            "competences": ["...", "..."],
            "langues": ["...", "..."],
            "resume": "résumé en une phrase"
        }}
        """)
        
        chain = prompt | llm
        response = chain.invoke({"content": cv_content})
        
        # Parse la réponse JSON
        try:
            cv_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Si le JSON n'est pas valide, créer une structure par défaut
            cv_data = {
                "informations_personnelles": {},
                "experiences": [],
                "formations": [],
                "competences": [],
                "langues": [],
                "resume": "Analyse en cours..."
            }
        
        return {"cv_data": cv_data}
        
    except Exception as e:
        return {"error": f"Erreur extraction: {str(e)}"}

@tool
def store_cv_embeddings(cv_content: str, cv_data: Dict, cv_hash: str, filename: str) -> Dict[str, Any]:
    """Stocke le CV avec embeddings et JSON"""
    try:
        # S'assurer que les tables existent
        ensure_tables_exist()
        
        # 1. Stockage embeddings
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_text(cv_content)
        
        documents = [
            Document(
                page_content=text,
                metadata={
                    "cv_hash": cv_hash,
                    "filename": filename,
                    "chunk_id": i
                }
            ) for i, text in enumerate(texts)
        ]
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Créer/connecter à la collection embeddings
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name="cv_embeddings",
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )
        
        # Ajouter les documents
        doc_ids = [f"{cv_hash}_{i}" for i in range(len(documents))]
        vectorstore.add_documents(documents, ids=doc_ids)
        
        # 2. Stockage JSON
        with psycopg2.connect(PSYCOPG_STRING) as conn:
            with conn.cursor() as cur:
                # Insérer le CV
                cur.execute("""
                    INSERT INTO cv_extraits 
                    (cv_hash, nom_fichier, informations_personnelles, experiences, formations, 
                     competences, langues, resume_professionnel, contenu_complet)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cv_hash) DO UPDATE SET
                        nom_fichier = EXCLUDED.nom_fichier,
                        date_traitement = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    cv_hash,
                    filename,
                    json.dumps(cv_data.get("informations_personnelles", {})),
                    json.dumps(cv_data.get("experiences", [])),
                    json.dumps(cv_data.get("formations", [])),
                    json.dumps(cv_data.get("competences", [])),
                    json.dumps(cv_data.get("langues", [])),
                    cv_data.get("resume", ""),
                    cv_content
                ))
                
                cv_id = cur.fetchone()[0]
                conn.commit()
                
        return {
            "success": True,
            "cv_id": str(cv_id),
            "chunks_stored": len(documents)
        }
        
    except Exception as e:
        return {"error": f"Erreur stockage: {str(e)}"}

@tool
def search_cvs(query: str, search_type: str = "mixed") -> Dict[str, Any]:
    """Recherche dans les CV (embeddings + JSON)"""
    try:
        # S'assurer que les tables existent
        ensure_tables_exist()
        
        results = []
        
        if search_type in ["embeddings", "mixed"]:
            try:
                # Recherche par embeddings
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                vectorstore = PGVector(
                    embeddings=embeddings,
                    collection_name="cv_embeddings", 
                    connection=CONNECTION_STRING,
                    use_jsonb=True,
                )
                
                docs = vectorstore.similarity_search(query, k=SEARCH_RESULTS_LIMIT)
                for doc in docs:
                    results.append({
                        "type": "embedding",
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                        "score": "semantic"
                    })
            except Exception as e:
                print(f"Erreur recherche embeddings: {e}")
        
        if search_type in ["json", "mixed"]:
            try:
                # Recherche dans les données JSON
                with psycopg2.connect(PSYCOPG_STRING) as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        # Recherche par compétences
                        cur.execute("""
                            SELECT cv_hash, nom_fichier, informations_personnelles, 
                                   competences, resume_professionnel
                            FROM cv_extraits 
                            WHERE competences::text ILIKE %s 
                               OR resume_professionnel ILIKE %s
                               OR informations_personnelles::text ILIKE %s
                            LIMIT 10
                        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
                        
                        for row in cur.fetchall():
                            results.append({
                                "type": "json",
                                "cv_hash": row['cv_hash'],
                                "filename": row['nom_fichier'],
                                "info": row['informations_personnelles'],
                                "competences": row['competences'],
                                "resume": row['resume_professionnel']
                            })
            except Exception as e:
                print(f"Erreur recherche JSON: {e}")
        
        return {"results": results, "total": len(results)}
        
    except Exception as e:
        return {"error": f"Erreur recherche: {str(e)}"}