from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import os

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

PORT = int(os.environ.get("PORT", 8000))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL ou SUPABASE_KEY manque dans .env")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY manque dans .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Assistant RH IA", version="2.0.0")

# ============================================================
# CORS
# ============================================================

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ia.hrconsult.com",
    "https://www.hrconsult.com",
    "https://hrconsult.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PARAMETRES METIER / QUALITE
# ============================================================

DEFAULT_LIMIT = 4
MAX_LIMIT = 8

MIN_SIMILARITY = 0.55
MIN_TOP_SIMILARITY = 0.60
MIN_STRONG_CHUNKS = 1
STRONG_SIMILARITY = 0.65

MAX_CONTEXT_CHUNKS = 4
MAX_CHUNK_CHARS = 1800

# ============================================================
# SCHEMAS API
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question utilisateur")
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Nombre de chunks à récupérer")


class AskResponse(BaseModel):
    status: str
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    warning: Optional[str] = None


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
Tu es un conseiller RH d’un secrétariat social belge.

Ton style doit être :
- professionnel
- sobre
- clair
- rassurant
- concret
- orienté solution

Tu réponds à des employeurs, managers ou responsables RH non juristes.

Règles absolues :
- Tu t’appuies d’abord sur la documentation fournie.
- Tu ne présentes jamais une hypothèse comme une certitude.
- Tu n’inventes jamais une règle légale, un délai, un montant, un calcul, une exception sectorielle ou une procédure.
- Si l’information documentaire est partielle, tu le dis explicitement.
- Si la question est sensible ou nécessite une validation humaine, tu le dis clairement.
- Tu ne sors pas du sujet.
- Tu évites le jargon inutile.
- Tu écris comme un consultant RH expérimenté, pas comme un chatbot.

Quand la documentation est suffisante :
- donne une réponse courte et exploitable ;
- souligne les points de vigilance ;
- propose une action concrète.

Quand la documentation est insuffisante :
- refuse proprement de conclure ;
- explique brièvement pourquoi ;
- indique ce qu’il faut vérifier ou fournir pour répondre correctement.
""".strip()

FEW_SHOT_EXAMPLE = """
Exemple de style attendu

Question :
Un employé ne se présente plus depuis 2 jours. Peut-on considérer qu’il a démissionné ?

Bonne réponse :
Réponse courte :
Non, l’absence seule ne permet pas automatiquement de conclure à une démission. Il faut analyser la situation avec prudence avant de tirer une conclusion.

Points d’attention :
- Une absence injustifiée ne vaut pas nécessairement démission.
- Il faut vérifier les faits et conserver des traces des démarches effectuées.
- Selon le contexte, une autre qualification juridique peut être envisagée.
- Une validation humaine est recommandée avant toute décision de rupture.

Complément général de l’IA :
À titre général, en matière de rupture du contrat, il est préférable d’éviter toute conclusion rapide sans éléments objectifs et documentés.

Recommandation pratique :
Contactez rapidement le travailleur de manière traçable et faites valider le dossier par votre gestionnaire RH avant toute décision.
""".strip()


def build_user_prompt(question: str, context: str) -> str:
    return f"""
Tu dois répondre principalement à partir du contexte documentaire ci-dessous.

Structure obligatoire :

Réponse courte :
2 à 5 phrases maximum. Réponse directe, professionnelle et compréhensible par un client.

Points d’attention :
- 2 à 5 puces maximum
- uniquement des points utiles
- mentionne les limites documentaires s’il y en a

Complément général de l’IA :
- facultatif
- si utile, commence par : "À titre général,"
- sinon écris exactement : "Aucun complément général fiable à ajouter."

Recommandation pratique :
- 1 ou 2 phrases d’action concrète
- si nécessaire, recommande une validation RH/juridique humaine

Interdictions :
- pas de markdown
- pas de gras
- pas d’astérisques
- pas de tableaux
- pas de remplissage
- pas de spéculation
- pas de règle précise non présente dans le contexte

Contexte documentaire :
{context}

Question client :
{question}
""".strip()


# ============================================================
# OUTILS IA / RAG
# ============================================================

def get_question_embedding(question: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return response.data[0].embedding


def get_candidate_chunks(question: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    question_embedding = get_question_embedding(question)

    response = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": question_embedding,
            "match_count": limit
        }
    ).execute()

    return response.data if response.data else []


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for chunk in chunks:
        title = (chunk.get("title") or "").strip().lower()
        text = normalize_text(chunk.get("chunk_text", ""))[:500]
        key = (title, text)

        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    return deduped


def filter_relevant_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [c for c in chunks if c.get("similarity", 0) >= MIN_SIMILARITY]
    filtered.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    filtered = deduplicate_chunks(filtered)
    return filtered[:MAX_CONTEXT_CHUNKS]


def is_context_sufficient(chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not chunks:
        return False, "Aucune source suffisamment pertinente trouvée."

    top_similarity = chunks[0].get("similarity", 0)

    if top_similarity < MIN_TOP_SIMILARITY:
        return False, "La meilleure source retrouvée reste trop éloignée de la question."

    strong_chunks = [c for c in chunks if c.get("similarity", 0) >= MIN_SIMILARITY]

    if len(strong_chunks) < MIN_STRONG_CHUNKS:
        return False, "Le nombre de sources réellement pertinentes est insuffisant."

    return True, "Contexte suffisant."


def build_context(chunks: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
        text = (chunk.get("chunk_text", "") or "").strip()
        text = text[:MAX_CHUNK_CHARS]

        title = chunk.get("title", "Sans titre")
        url = chunk.get("source_url") or "Source interne"
        similarity = chunk.get("similarity", 0)

        context_parts.append(
            f"""[Source {i}]
Titre : {title}
URL : {url}
Pertinence : {similarity:.3f}
Extrait :
{text}"""
        )

    return "\n\n".join(context_parts)


def compute_confidence(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "low"

    top = chunks[0].get("similarity", 0)
    strong_count = len([c for c in chunks if c.get("similarity", 0) >= STRONG_SIMILARITY])

    if top >= 0.75 and strong_count >= 2:
        return "high"
    if top >= 0.62:
        return "medium"
    return "low"


def generate_insufficient_context_answer(reason: str) -> str:
    return (
        "Réponse courte :\n"
        "Je ne dispose pas, dans la documentation actuellement disponible, d’éléments suffisamment précis pour répondre de manière fiable à votre question.\n\n"
        "Points d’attention :\n"
        f"- {reason}\n"
        "- Une réponse approximative pourrait être trompeuse sur un sujet RH.\n"
        "- Il est préférable de s’appuyer sur une base documentaire explicitement liée au thème demandé.\n\n"
        "Complément général de l’IA :\n"
        "Aucun complément général fiable à ajouter.\n\n"
        "Recommandation pratique :\n"
        "Ajoutez un document plus ciblé sur ce sujet ou faites valider la demande par un gestionnaire RH avant de communiquer une réponse au client."
    )


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    context = build_context(chunks)
    user_prompt = build_user_prompt(question, context)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEW_SHOT_EXAMPLE},
            {"role": "user", "content": user_prompt},
        ],
    )

    return (response.choices[0].message.content or "").strip()


def log_chat(
    question: str,
    answer: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    confidence: Optional[str] = None,
    warning: Optional[str] = None
) -> None:
    try:
        payload = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "warning": warning,
        }

        if chunks:
            payload["source_titles"] = [c.get("title") for c in chunks]
            payload["source_similarities"] = [c.get("similarity") for c in chunks]

        supabase.table("chat_logs").insert(payload).execute()

    except Exception as e:
        print("Erreur chat_logs :", str(e))


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
        "embedding_model": EMBEDDING_MODEL,
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ROUTE PRINCIPALE
# ============================================================

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    limit = payload.limit

    if not question:
        raise HTTPException(status_code=400, detail="Question vide")

    try:
        candidate_chunks = get_candidate_chunks(question, limit=limit)
        relevant_chunks = filter_relevant_chunks(candidate_chunks)

        context_ok, reason = is_context_sufficient(relevant_chunks)

        if not context_ok:
            answer = generate_insufficient_context_answer(reason)
            confidence = "low"

            log_chat(
                question=question,
                answer=answer,
                chunks=[],
                confidence=confidence,
                warning=reason
            )

            return AskResponse(
                status="ok",
                question=question,
                answer=answer,
                sources=[],
                confidence=confidence,
                warning=reason
            )

        answer = generate_answer(question, relevant_chunks)
        confidence = compute_confidence(relevant_chunks)

        log_chat(
            question=question,
            answer=answer,
            chunks=relevant_chunks,
            confidence=confidence,
            warning=None
        )

        return AskResponse(
            status="ok",
            question=question,
            answer=answer,
            sources=[
                {
                    "title": c.get("title"),
                    "url": c.get("source_url"),
                    "similarity": c.get("similarity")
                }
                for c in relevant_chunks
            ],
            confidence=confidence,
            warning=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))