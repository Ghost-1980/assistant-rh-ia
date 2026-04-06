import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from docx import Document

# ----------------------------
# CONFIG
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DOCX_DIR = BASE_DIR / "data" / "docx"

load_dotenv(ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL ou SUPABASE_KEY manquant dans .env")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY manquant dans .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# EMBEDDINGS
# ----------------------------

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ----------------------------
# EXTRACTION DOCX
# ----------------------------

def extract_text_from_docx(docx_path: Path):
    doc = Document(docx_path)
    all_text = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            all_text.append(text)

    clean_text = "\n\n".join(all_text).strip()
    title = docx_path.stem

    return title, clean_text

# ----------------------------
# NORMALISATION EN PARAGRAPHES
# ----------------------------

def normalize_paragraphs(text: str):
    raw_lines = text.splitlines()

    paragraphs = []
    current = []

    for line in raw_lines:
        stripped = line.strip()

        if not stripped:
            if current:
                paragraph = " ".join(current).strip()
                if paragraph:
                    paragraphs.append(paragraph)
                current = []
            continue

        current.append(stripped)

    if current:
        paragraph = " ".join(current).strip()
        if paragraph:
            paragraphs.append(paragraph)

    cleaned = []
    for p in paragraphs:
        p = " ".join(p.split())
        if len(p) >= 40:
            cleaned.append(p)

    return cleaned

# ----------------------------
# CHUNKING PROPRE
# ----------------------------

def split_chunks(text: str, max_size: int = 1500, overlap_paragraphs: int = 1):
    paragraphs = normalize_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if paragraph_length > max_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk).strip())
                current_chunk = []
                current_length = 0

            start = 0
            while start < paragraph_length:
                part = paragraph[start:start + max_size].strip()
                if part:
                    chunks.append(part)
                start += max_size
            continue

        if current_length + paragraph_length + 2 <= max_size:
            current_chunk.append(paragraph)
            current_length += paragraph_length + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk).strip())

            overlap = current_chunk[-overlap_paragraphs:] if current_chunk else []
            current_chunk = overlap + [paragraph]
            current_length = sum(len(p) + 2 for p in current_chunk)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks

# ----------------------------
# DB CHECK
# ----------------------------

def get_existing_document_by_file_name(file_name: str):
    response = (
        supabase
        .table("documents")
        .select("id, title, file_name")
        .eq("file_name", file_name)
        .limit(1)
        .execute()
    )

    if response.data and len(response.data) > 0:
        return response.data[0]

    return None

# ----------------------------
# INSERT CHUNKS
# ----------------------------

def insert_chunks(document_id: int, content_text: str):
    chunks = split_chunks(content_text)

    rows = []
    for index, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        rows.append({
            "document_id": document_id,
            "chunk_index": index,
            "chunk_text": chunk,
            "embedding": embedding
        })

    if rows:
        supabase.table("document_chunks").insert(rows).execute()

    return len(rows)

# ----------------------------
# IMPORT DOCX
# ----------------------------

def import_docx_to_supabase(docx_path: Path, category: str = "docx"):
    if not docx_path.exists():
        raise FileNotFoundError(f"Fichier DOCX introuvable : {docx_path}")

    existing_document = get_existing_document_by_file_name(docx_path.name)

    if existing_document:
        return {
            "status": "already_exists",
            "message": "Ce DOCX existe déjà dans la base.",
            "document_id": existing_document["id"],
            "title": existing_document["title"],
            "file_name": existing_document["file_name"]
        }

    title, content_text = extract_text_from_docx(docx_path)

    if not content_text.strip():
        return {
            "status": "empty",
            "message": "Aucun texte extractible trouvé dans ce DOCX.",
            "file_name": docx_path.name
        }

    document_data = {
        "source_type": "docx",
        "title": title,
        "source_url": None,
        "file_name": docx_path.name,
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "category": category,
        "status": "active",
        "content_text": content_text,
    }

    document_result = supabase.table("documents").insert(document_data).execute()

    inserted_document = document_result.data[0]
    document_id = inserted_document["id"]

    chunk_count = insert_chunks(document_id, content_text)

    return {
        "status": "inserted",
        "document_id": document_id,
        "title": title,
        "file_name": docx_path.name,
        "chunk_count": chunk_count
    }

# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    files = sorted(DOCX_DIR.rglob("*.docx"))

    print(f"Nombre de DOCX trouvés : {len(files)}")

    for i, file in enumerate(files, 1):
        print("-" * 60)
        print(f"[{i}/{len(files)}] {file}")

        try:
            result = import_docx_to_supabase(file, category=file.parent.name)
            print(result)
        except Exception as e:
            print("ERREUR :", str(e))