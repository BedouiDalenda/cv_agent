"""Outils LangChain pour l'agent CV - Version corrigée"""
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

from config import CONNECTION_STRING, PSYCOPG_STRING, LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from utils import clean_text, generate_cv_hash

def ensure_tables_exist():
    """S'assure que toutes les tables nécessaires existent"""
    try:
        with psycopg2.connect(PSYCOPG_STRING) as conn:
            with conn.cursor() as cur:
                # Créer les extensions nécessaires
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
                
                # Table principale pour les données JSON
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cv_data (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        cv_hash VARCHAR(32) UNIQUE NOT NULL,
                        nom_fichier VARCHAR(255) NOT NULL,
                        date_traitement TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        nom VARCHAR(255),
                        prenom VARCHAR(255),
                        email VARCHAR(255),
                        telephone VARCHAR(50),
                        adresse TEXT,
                        experiences JSONB,
                        formations JSONB, 
                        competences JSONB,
                        langues JSONB,
                        resume_professionnel TEXT,
                        contenu_complet TEXT
                    );
                """)
                
                # Index pour optimiser les recherches
                cur.execute("CREATE INDEX IF NOT EXISTS idx_cv_hash ON cv_data (cv_hash);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_cv_competences ON cv_data USING GIN (competences);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_cv_nom ON cv_data (nom, prenom);")
                
                conn.commit()
                print("✅ Tables créées avec succès")
    except Exception as e:
        print(f"❌ Erreur création tables: {e}")

@tool
def generate_sql_from_query(natural_query: str) -> Dict[str, Any]:
    """PROMPT 1: Transforme une phrase en requête SQL - Version améliorée"""
    try:
        print(f"🔧 Génération SQL pour: '{natural_query}'")
        llm = ChatMistralAI(model=LLM_MODEL, temperature=0.1)
        
        prompt = ChatPromptTemplate.from_template("""
        Tu es un expert en SQL. Convertis cette demande en français en requête SQL PostgreSQL.

        SCHEMA DE LA TABLE cv_data:
        - id (UUID)
        - nom (VARCHAR)
        - prenom (VARCHAR) 
        - email (VARCHAR)
        - telephone (VARCHAR)
        - adresse (TEXT)
        - experiences (JSONB) - array d'objets avec poste, entreprise, duree, description
        - formations (JSONB) - array d'objets avec diplome, etablissement, annee
        - competences (JSONB) - array de strings
        - langues (JSONB) - array de strings
        - resume_professionnel (TEXT)
        - contenu_complet (TEXT)
        - date_traitement (TIMESTAMP)

        DEMANDE: {query}

        RÈGLES STRICTES:
        1. Utilise UNIQUEMENT des SELECT
        2. Pour chercher dans les arrays JSON, utilise: competences @> '["nom_competence"]'
        3. Pour chercher du texte libre, utilise ILIKE avec %
        4. Limite les résultats avec LIMIT 10
        5. Retourne SEULEMENT la requête SQL, sans explication ni formatage
        6. Ne pas utiliser de quotes autour de la requête finale
        7. Assure-toi que la syntaxe SQL est parfaitement valide

        EXEMPLES CORRECTS:
        - "CV avec Python" → SELECT * FROM cv_data WHERE competences @> '["Python"]' LIMIT 10;
        - "développeurs React et Node.js" → SELECT * FROM cv_data WHERE competences @> '["React"]' AND competences @> '["Node.js"]' LIMIT 10;
        - "profils seniors" → SELECT * FROM cv_data WHERE resume_professionnel ILIKE '%senior%' LIMIT 10;
        
        SQL:
        """)
        
        chain = prompt | llm
        response = chain.invoke({"query": natural_query})
        
        # Nettoyer la réponse pour extraire seulement le SQL
        sql_query = response.content.strip()
        
        # Supprimer les blocs de code markdown
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        # Supprimer les guillemets de début et fin si présents
        if sql_query.startswith('"') and sql_query.endswith('"'):
            sql_query = sql_query[1:-1]
        if sql_query.startswith("'") and sql_query.endswith("'"):
            sql_query = sql_query[1:-1]
        
        # S'assurer que la requête se termine par un point-virgule
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        print(f"📝 SQL généré: {sql_query}")
        return {"sql_query": sql_query}
        
    except Exception as e:
        error_msg = f"Erreur génération SQL: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    

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
        ensure_tables_exist()
        
        with psycopg2.connect(PSYCOPG_STRING) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, nom_fichier FROM cv_data WHERE cv_hash = %s", 
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
    """PROMPT 2: Extrait les informations structurées du CV"""
    try:
        llm = ChatMistralAI(model=LLM_MODEL, temperature=0.1, max_tokens=1500)
        
        prompt = ChatPromptTemplate.from_template("""
        Analyse ce CV et extrais les informations en format JSON STRICT.

        CV Content: {content}

        Tu DOIS retourner un JSON valide avec exactement cette structure:
        {{
            "nom": "nom de famille ou vide",
            "prenom": "prénom ou vide", 
            "email": "email ou vide",
            "telephone": "numéro ou vide",
            "adresse": "adresse complète ou vide",
            "experiences": [
                {{"poste": "titre du poste", "entreprise": "nom entreprise", "duree": "période", "description": "description courte"}}
            ],
            "formations": [
                {{"diplome": "nom diplôme", "etablissement": "nom école", "annee": "année"}}
            ],
            "competences": ["competence1", "competence2"],
            "langues": ["langue1", "langue2"],
            "resume": "résumé professionnel en une phrase"
        }}

        IMPORTANT: 
        - Retourne SEULEMENT le JSON, pas d'explication
        - Utilise des strings vides "" pour les champs manquants
        - Utilise des arrays vides [] pour les listes manquantes
        """)
        
        chain = prompt | llm
        response = chain.invoke({"content": cv_content})
        
        # Parse la réponse JSON
        try:
            # Nettoyer la réponse
            json_content = response.content.strip()
            if json_content.startswith("```json"):
                json_content = json_content.replace("```json", "").replace("```", "").strip()
            elif json_content.startswith("```"):
                json_content = json_content.replace("```", "").strip()
                
            cv_data = json.loads(json_content)
            
            # Validation des champs obligatoires
            required_fields = ["nom", "prenom", "email", "telephone", "adresse", "experiences", "formations", "competences", "langues", "resume"]
            for field in required_fields:
                if field not in cv_data:
                    cv_data[field] = "" if field not in ["experiences", "formations", "competences", "langues"] else []
            
        except json.JSONDecodeError as e:
            print(f"Erreur parsing JSON: {e}")
            print(f"Contenu reçu: {response.content}")
            # Structure par défaut en cas d'erreur
            cv_data = {
                "nom": "", "prenom": "", "email": "", "telephone": "", "adresse": "",
                "experiences": [], "formations": [], "competences": [], "langues": [],
                "resume": "Analyse en cours..."
            }
        
        return {"cv_data": cv_data}
        
    except Exception as e:
        return {"error": f"Erreur extraction: {str(e)}"}

@tool
def store_cv_data(cv_content: str, cv_data: Dict, cv_hash: str, filename: str) -> Dict[str, Any]:
    """PROMPT 3: Stocke le CV dans les deux tables (JSON + embeddings)"""
    try:
        ensure_tables_exist()
        
        # 1. Stockage dans la table JSON
        with psycopg2.connect(PSYCOPG_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO cv_data 
                    (cv_hash, nom_fichier, nom, prenom, email, telephone, adresse,
                     experiences, formations, competences, langues, resume_professionnel, contenu_complet)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cv_hash) DO UPDATE SET
                        nom_fichier = EXCLUDED.nom_fichier,
                        date_traitement = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    cv_hash, filename,
                    cv_data.get("nom", ""), cv_data.get("prenom", ""),
                    cv_data.get("email", ""), cv_data.get("telephone", ""), cv_data.get("adresse", ""),
                    json.dumps(cv_data.get("experiences", [])),
                    json.dumps(cv_data.get("formations", [])),
                    json.dumps(cv_data.get("competences", [])),
                    json.dumps(cv_data.get("langues", [])),
                    cv_data.get("resume", ""), cv_content
                ))
                
                cv_id = cur.fetchone()[0]
                conn.commit()
        
        # 2. Stockage embeddings
        try:
            text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            texts = text_splitter.split_text(cv_content)
            
            documents = [
                Document(
                    page_content=text,
                    metadata={"cv_hash": cv_hash, "filename": filename, "chunk_id": i}
                ) for i, text in enumerate(texts)
            ]
            
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = PGVector(
                embeddings=embeddings,
                collection_name="cv_embeddings",
                connection=CONNECTION_STRING,
                use_jsonb=True,
            )
            
            doc_ids = [f"{cv_hash}_{i}" for i in range(len(documents))]
            vectorstore.add_documents(documents, ids=doc_ids)
            
            return {
                "success": True,
                "cv_id": str(cv_id),
                "chunks_stored": len(documents),
                "json_stored": True
            }
            
        except Exception as embed_error:
            # Si les embeddings échouent, on garde au moins le JSON
            return {
                "success": True,
                "cv_id": str(cv_id),
                "chunks_stored": 0,
                "json_stored": True,
                "warning": f"Embeddings failed: {str(embed_error)}"
            }
        
    except Exception as e:
        return {"error": f"Erreur stockage: {str(e)}"}

@tool
def execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """Exécute la requête SQL générée"""
    try:
        ensure_tables_exist()
        
        with psycopg2.connect(PSYCOPG_STRING) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql_query)
                results = cur.fetchall()
                
                # Convertir en dictionnaires Python
                formatted_results = []
                for row in results:
                    row_dict = dict(row)
                    # Convertir les JSONB en listes Python
                    for key in ["experiences", "formations", "competences", "langues"]:
                        if key in row_dict and row_dict[key]:
                            if isinstance(row_dict[key], str):
                                row_dict[key] = json.loads(row_dict[key])
                    formatted_results.append(row_dict)
                
                return {"results": formatted_results, "count": len(formatted_results)}
                
    except Exception as e:
        return {"error": f"Erreur exécution SQL: {str(e)}"}
