from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from pathlib import Path
import os
port = int(os.environ.get("PORT", 8000))

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL ou SUPABASE_KEY manque dans .env")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY manque dans .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Assistant RH IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",

        # TON FRONTEND EN LIGNE
        "https://ia.hrconsult.com",
        "https://www.hrconsult.com",
        "https://hrconsult.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CORS
# A ADAPTER AVEC TES DOMAINES REELS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://tonsite.be",
        "https://www.tonsite.be",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PARAMETRES METIER / QUALITE
# ============================================================

DEFAULT_LIMIT = 6
MAX_LIMIT = 10

# Seuil de pertinence :
# on préfère refuser plutôt que répondre hors sujet
MIN_SIMILARITY = 0.55

# Si le meilleur chunk est trop faible, on refuse
MIN_TOP_SIMILARITY = 0.58

# Nombre minimal de chunks "solides" souhaités
MIN_STRONG_CHUNKS = 1

# ============================================================
# ROUTES BASIQUES
# ============================================================

@app.get("/")
def read_root():
    return {"message": "Assistant RH IA en ligne"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_model": OPENAI_MODEL,
        "supabase_url_present": bool(SUPABASE_URL),
        "supabase_key_present": bool(SUPABASE_KEY),
        "openai_key_present": bool(OPENAI_API_KEY),
    }


@app.get("/documents")
def get_documents():
    try:
        response = supabase.table("documents").select("*").limit(20).execute()
        return {
            "status": "ok",
            "count": len(response.data),
            "documents": response.data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/chunks")
def get_chunks():
    try:
        response = supabase.table("document_chunks").select("*").limit(20).execute()
        return {
            "status": "ok",
            "count": len(response.data),
            "chunks": response.data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ============================================================
# OUTILS IA
# ============================================================

def get_question_embedding(question: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    return response.data[0].embedding


def get_candidate_chunks(question: str, limit: int = DEFAULT_LIMIT):
    question_embedding = get_question_embedding(question)

    response = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": question_embedding,
            "match_count": limit
        }
    ).execute()

    return response.data if response.data else []


def filter_relevant_chunks(chunks: list[dict]) -> list[dict]:
    """
    Filtre les chunks trop faibles pour éviter le hors-sujet.
    """
    filtered = [c for c in chunks if c.get("similarity", 0) >= MIN_SIMILARITY]
    return filtered


def is_context_sufficient(chunks: list[dict]) -> tuple[bool, str]:
    """
    Détermine si le contexte est assez pertinent pour répondre.
    Retourne (bool, raison)
    """
    if not chunks:
        return False, "Aucune source suffisamment pertinente trouvée."

    top_similarity = chunks[0].get("similarity", 0)

    if top_similarity < MIN_TOP_SIMILARITY:
        return False, "Les sources retrouvées semblent trop éloignées de la question."

    strong_chunks = [c for c in chunks if c.get("similarity", 0) >= MIN_SIMILARITY]

    if len(strong_chunks) < MIN_STRONG_CHUNKS:
        return False, "Le nombre de sources réellement pertinentes est insuffisant."

    return True, "Contexte suffisant."


def build_context(chunks: list[dict]) -> str:
    """
    Construit un contexte lisible et exploitable pour le modèle.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("chunk_text", "").strip()
        title = chunk.get("title", "Sans titre")
        url = chunk.get("source_url") or "Source interne / document local"
        similarity = chunk.get("similarity", 0)

        context_parts.append(
            f"""Source {i}
Titre : {title}
URL : {url}
Pertinence : {similarity:.4f}
Contenu :
{text}"""
        )

    return "\n\n---\n\n".join(context_parts)


def generate_insufficient_context_answer(reason: str) -> str:
    """
    Réponse standard quand le contexte n'est pas assez pertinent.
    """
    return (
        "Réponse courte :\n"
        "Je ne trouve pas de source suffisamment pertinente dans la documentation pour répondre de façon fiable à cette question.\n\n"
        "Points d'attention :\n"
        f"- {reason}\n"
        "- Pour éviter une réponse hors sujet, je préfère ne pas formuler de conclusion sur cette base.\n"
        "- Une réponse fiable nécessite des documents plus directement liés au thème demandé.\n\n"
        "Complément général de l'IA :\n"
        "Aucun complément général fiable à ajouter sans risque de sortir du cadre documentaire ou d'induire en erreur.\n\n"
        "Recommandation pratique :\n"
        "Ajoutez ou importez des documents plus ciblés sur ce sujet, ou faites valider la question par un gestionnaire RH."
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Génère une réponse professionnelle, prudente et orientée client.
    Le modèle peut compléter légèrement avec ses connaissances générales,
    mais uniquement de manière limitée et explicitement signalée.
    """
    context = build_context(chunks)

    prompt = f"""
Tu es l'assistant RH IA d'un secrétariat social belge.
Tu aides des employeurs sur des questions de droit du travail, paie, administration sociale et gestion du personnel.

Tu dois répondre comme un expert métier :
- professionnel
- clair
- pratique
- prudent
- orienté client

PRIORITE ABSOLUE
1. Tu dois d'abord t'appuyer sur la documentation fournie.
2. Si la documentation répond à la question, ta réponse doit être principalement basée dessus.
3. Tu peux compléter avec des connaissances générales de ChatGPT, mais de manière limitée, prudente et clairement signalée.
4. Tu ne peux jamais inventer :
   - un chiffre
   - un délai
   - une règle légale
   - une exception sectorielle
   - un calcul de paie précis
   si ce n'est pas confirmé par la documentation.
5. Si la documentation est partielle, tu dois l'indiquer.
6. Si la question nécessite un calcul précis, une analyse juridique individuelle ou une validation humaine, tu dois le dire clairement.

COMPORTEMENT OBLIGATOIRE
- Ne parle jamais d'un thème hors sujet.
- Ne reformule pas des sources non pertinentes pour remplir la réponse.
- Ne mentionne pas un document juste parce qu'il a été trouvé si son contenu ne répond pas à la question.
- Ne donne jamais l'impression qu'une réponse est certaine si elle ne l'est pas.
- Réponds en français professionnel, simple et concret.
- Tu t'adresses à un client non juriste.

CAS PARTICULIERS
- Si la question porte sur la paie ou un calcul :
  ne donne un calcul ou une règle chiffrée que si la documentation le permet réellement.
- Si la question porte sur le droit du travail :
  distingue si possible la règle générale, les points d'attention et la nécessité éventuelle d'une validation.
- Si la question est sensible (licenciement, faute grave, retenues salariales, sanctions, rupture de contrat, etc.) :
  reste prudent et recommande une validation humaine si nécessaire.

STRUCTURE OBLIGATOIRE DE LA REPONSE
Réponse courte :
...
 
Points d'attention :
- ...
- ...

Complément général de l'IA :
...
 
Recommandation pratique :
...

CONSIGNES DE STYLE
- Texte brut uniquement
- Pas de markdown
- Pas d'astérisques
- Pas de gras
- Pas de jargon inutile
- Réponse utile et exploitable

REGLE TRES IMPORTANTE POUR "Complément général de l'IA"
- Cette section est facultative.
- Si tu ajoutes un complément, commence par : "À titre général,"
- Si tu n'as rien à ajouter de fiable, écris exactement :
  "Aucun complément général fiable à ajouter."

CONTEXTE DOCUMENTAIRE :
{context}

QUESTION DU CLIENT :
{question}

Rédige maintenant la réponse finale.
"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant RH belge expert en droit du travail, paie et administration sociale. "
                    "Tu réponds comme un professionnel du secrétariat social. "
                    "Tu utilises d'abord la documentation fournie. "
                    "Tu peux compléter légèrement avec des connaissances générales, "
                    "mais seulement si tu le signales clairement et sans inventer de règle précise."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def log_chat(question: str, answer: str):
    """
    Enregistre la question/réponse dans chat_logs.
    On reste minimal pour éviter un conflit avec un schéma différent.
    """
    try:
        supabase.table("chat_logs").insert({
            "question": question,
            "answer": answer
        }).execute()
    except Exception as e:
        print("Erreur chat_logs :", str(e))

# ============================================================
# ROUTE PRINCIPALE /ask
# ============================================================

@app.get("/ask")
def ask(
    question: str = Query(..., description="Question utilisateur"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Nombre de chunks à récupérer")
):
    question = question.strip()

    if not question:
        return {
            "status": "error",
            "message": "Question vide"
        }

    try:
        candidate_chunks = get_candidate_chunks(question, limit=limit)
        relevant_chunks = filter_relevant_chunks(candidate_chunks)

        context_ok, reason = is_context_sufficient(relevant_chunks)

        if not context_ok:
            answer = generate_insufficient_context_answer(reason)
            log_chat(question, answer)

            return {
                "status": "ok",
                "question": question,
                "answer": answer,
                "sources": [],
                "warning": reason
            }

        answer = generate_answer(question, relevant_chunks)
        log_chat(question, answer)

        return {
            "status": "ok",
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "title": c.get("title"),
                    "url": c.get("source_url"),
                    "similarity": c.get("similarity")
                }
                for c in relevant_chunks
            ],
            "warning": None
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }