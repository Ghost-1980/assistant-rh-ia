import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
URLS_FILE = BASE_DIR / "data" / "URL" / "urls.txt"

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
# LECTURE DES URLS
# ----------------------------

def read_urls(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    urls = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()

            if not url:
                continue

            if url.startswith("#"):
                continue

            urls.append(url)

    return urls

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
# EXTRACTION HTML
# ----------------------------

def extract_text(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "fr-BE,fr;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.google.com/"
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]

    clean_text = "\n".join(clean_lines)

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

    # Nettoyage final
    cleaned = []
    for p in paragraphs:
        p = " ".join(p.split())
        if len(p) >= 40:  # on élimine les micro-fragments trop courts
            cleaned.append(p)

    return cleaned

# ----------------------------
# CHUNKING PROPRE PAR BLOCS DE SENS
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

        # si le paragraphe seul est énorme, on le coupe proprement
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

        # si on peut encore ajouter le paragraphe au chunk courant
        if current_length + paragraph_length + 2 <= max_size:
            current_chunk.append(paragraph)
            current_length += paragraph_length + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk).strip())

            # overlap : on reprend le ou les derniers paragraphes
            if overlap_paragraphs > 0 and current_chunk:
                overlap = current_chunk[-overlap_paragraphs:]
            else:
                overlap = []

            current_chunk = overlap + [paragraph]
            current_length = sum(len(p) + 2 for p in current_chunk)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks

# ----------------------------
# VERIFICATION DOUBLON
# ----------------------------

def url_exists(url: str):
    response = (
        supabase
        .table("documents")
        .select("id")
        .eq("source_url", url)
        .limit(1)
        .execute()
    )

    return len(response.data) > 0

# ----------------------------
# INSERTION DES CHUNKS
# ----------------------------

def insert_chunks(document_id: int, text: str):
    chunks = split_chunks(text)

    rows = []

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        rows.append({
            "document_id": document_id,
            "chunk_index": i,
            "chunk_text": chunk,
            "embedding": embedding
        })

    if rows:
        supabase.table("document_chunks").insert(rows).execute()

    return len(rows)

# ----------------------------
# IMPORT D'UNE URL
# ----------------------------

def import_one_url(url: str):
    if url_exists(url):
        return {"status": "already_exists", "url": url}

    title, text = extract_text(url)

    if not text.strip():
        return {"status": "empty", "url": url}

    doc = {
        "source_type": "url",
        "title": title,
        "source_url": url,
        "file_name": None,
        "mime_type": "text/html",
        "category": "bulk_import",
        "status": "active",
        "content_text": text,
    }

    result = supabase.table("documents").insert(doc).execute()
    document_id = result.data[0]["id"]

    chunk_count = insert_chunks(document_id, text)

    return {
        "status": "inserted",
        "url": url,
        "chunks": chunk_count
    }

# ----------------------------
# MAIN
# ----------------------------

def main():
    urls = read_urls(URLS_FILE)

    print(f"{len(urls)} URL(s) trouvée(s)")

    inserted = 0
    exists = 0
    empty = 0
    errors = 0

    for i, url in enumerate(urls, 1):
        print("-" * 70)
        print(f"[{i}/{len(urls)}] {url}")

        try:
            result = import_one_url(url)
            print(result)

            if result["status"] == "inserted":
                inserted += 1
            elif result["status"] == "already_exists":
                exists += 1
            elif result["status"] == "empty":
                empty += 1
            else:
                errors += 1

        except Exception as e:
            print("ERREUR :", str(e))
            errors += 1

    print("\n===== RESULTAT =====")
    print("Nouveaux :", inserted)
    print("Déjà existants :", exists)
    print("Vides :", empty)
    print("Erreurs :", errors)


if __name__ == "__main__":
    main()