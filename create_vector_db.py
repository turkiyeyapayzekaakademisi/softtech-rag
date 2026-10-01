"""
Amaç:
    1. data/softtech_hakkinda.pdf dosyasını oku.
    2. Metni iki farklı yöntemle chunk'lara ayır:
        - Karakter bazlı: 500 karakter, 100 overlap
        - Başlık bazlı
    3. Qwen3-Embedding-0.6B modeli ile embedding oluştur.
    4. İki farklı chunking yöntemi için iki ayrı FAISS Vector DB oluştur.
    5. Vector DB'leri klasöre kaydet.

Kurulum:
    pip install pypdf langchain-text-splitters sentence-transformers faiss-cpu
"""

import os
import json
import faiss

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# 1. PDF oku
pdf_path = "data/softtech_hakkinda.pdf"
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

print("PDF okundu")
print(text)


# 2.1 Karakter bazlı chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks_character = text_splitter.split_text(text)

print("\nKarakter bazlı chunk sayısı:", len(chunks_character))

for i, chunk in enumerate(chunks_character):
    print(f"\nChunk {i + 1}")
    print(chunk)


# 2.2 Başlık bazlı chunking
chunks_heading = []
current_chunk = ""

for line in text.splitlines():
    line = line.strip()

    if not line:
        continue

    if line.isupper():
        if current_chunk:
            chunks_heading.append(current_chunk.strip())

        current_chunk = line + "\n"
    else:
        current_chunk += line + " "

if current_chunk:
    chunks_heading.append(current_chunk.strip())

print("\nBaşlık bazlı chunk sayısı:", len(chunks_heading))

for i, chunk in enumerate(chunks_heading):
    print(f"\nChunk {i + 1}")
    print(chunk)


# 3. Embedding
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

character_embeddings = model.encode(chunks_character, normalize_embeddings=True)
heading_embeddings = model.encode(chunks_heading, normalize_embeddings=True)

print("\nCharacter embedding shape:", character_embeddings.shape)
print("Heading embedding shape:", heading_embeddings.shape)

# 4. Vector DB klasörlerini oluştur
os.makedirs("vector_db/character", exist_ok=True)
os.makedirs("vector_db/heading", exist_ok=True)

# 4.1 Character Vector DB
character_index = faiss.IndexFlatIP(character_embeddings.shape[1])
character_index.add(character_embeddings.astype("float32"))

faiss.write_index(character_index, "vector_db/character/index.faiss")

with open("vector_db/character/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks_character, f, ensure_ascii=False, indent=2)

print("\nCharacter Vector DB kaydedildi")

# 4.2 Heading Vector DB
heading_index = faiss.IndexFlatIP(heading_embeddings.shape[1])
heading_index.add(heading_embeddings.astype("float32"))

faiss.write_index(heading_index, "vector_db/heading/index.faiss")

with open("vector_db/heading/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks_heading, f, ensure_ascii=False, indent=2)

print("Heading Vector DB kaydedildi")