from pathlib import Path
import re
import numpy as np
import tiktoken

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -------- CONFIG --------
VAULT_PATH = Path("Knowledge_Base")
DB_PATH = "./chroma_db"

enc = tiktoken.get_encoding("cl100k_base")

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# -------- HELPERS --------

def token_len(text):
    return len(enc.encode(text))


def load_notes(vault_path):
    notes = []
    for md_file in vault_path.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        clean = re.sub(r"^---.*?---\s*", "", text, flags=re.S)
        clean = re.sub(r"!\[\[.*?\]\]", "", clean)
        notes.append({"path": md_file, "text": clean})
    return notes


def split_into_units(text):
    units = re.split(r"\n\s*\n", text)
    return [u.strip() for u in units if len(u.strip()) > 40]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_chunk(units, embeddings, max_tokens=450, similarity_threshold=0.75):
    if not units:
        return []

    chunks = []
    current_chunk = units[0]
    current_tokens = token_len(current_chunk)

    for i in range(1, len(units)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        unit_tokens = token_len(units[i])

        if sim >= similarity_threshold and current_tokens + unit_tokens <= max_tokens:
            current_chunk += "\n\n" + units[i]
            current_tokens += unit_tokens
        else:
            chunks.append(current_chunk)
            current_chunk = units[i]
            current_tokens = unit_tokens

    chunks.append(current_chunk)
    return chunks


def semantic_chunk_document(text):
    units = split_into_units(text)
    if not units:
        return []

    embeddings = embedding_function.embed_documents(units)
    return semantic_chunk(units, embeddings)


# -------- INGESTION --------

def main():
    print("Loading notes...")
    notes = load_notes(VAULT_PATH)

    texts = []
    metadatas = []

    for note in notes:
        print(f"Processing: {note['path']}")
        chunks = semantic_chunk_document(note["text"])

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": note["path"].name,
                    "chunk_id": i,
                    "tokens": token_len(chunk),
                }
            )

    print(f"Total chunks: {len(texts)}")

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        persist_directory=DB_PATH,
    )

    print(f"Vector DB stored at {DB_PATH}")


if __name__ == "__main__":
    main()




#pip install langchain \langchain-community langchain-chroma langchain-huggingface sentence-transformers tiktoken numpy
