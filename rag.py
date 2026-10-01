"""
Amaç:
    1. FAISS Vector DB'yi yükle.
    2. Qwen3-Embedding-0.6B embedding modelini yükle.
    3. Ollama üzerinde gemma3:4b modelini kullan.
    4. Kullanıcı sorusunu embedding'e çevir.
    5. Vector DB'den en yakın 3 chunk'ı getir.
    6. Qwen3-Reranker-0.6B ile chunk'ları yeniden sırala.
    7. Chunk'ları LLM'e gönder ve cevap üret.

Kurulum:
    pip install faiss-cpu sentence-transformers ollama

Ollama:
    ollama pull gemma3:4b
"""

import json
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import chat

# 0. Modelleri ve Vector DB'yi yükle
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
reranker = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")

index = faiss.read_index("vector_db/character/index.faiss")

with open("vector_db/character/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Vector DB yüklendi")
print("Embedding modeli yüklendi")
print("Reranker modeli yüklendi")
print("LLM: gemma3:4b")


# 1. Kullanıcıdan soru al
question = input("\nSorunuz: ")

print("\nKullanıcı sorusu:")
print(question)


# 2. Soruyu embedding'e çevir
question_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True
).astype("float32")

print("\nSoru embedding'e çevrildi")
print("Embedding shape:", question_embedding.shape)


# 3. Vector DB'den en yakın 3 chunk'ı getir
scores, indices = index.search(question_embedding, 3)

retrieved_chunks = [chunks[i] for i in indices[0]]

print("\nGetirilen chunk'lar:")

for i, chunk in enumerate(retrieved_chunks):
    print(f"\nChunk {i + 1}")
    print(chunk)
    print("Similarity:", scores[0][i])


# 4. Reranker
pairs = [(question, chunk) for chunk in retrieved_chunks]

rerank_scores = reranker.predict(pairs)

reranked = sorted(
    zip(retrieved_chunks, rerank_scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nReranker sonrası:")

for i, (chunk, score) in enumerate(reranked):
    print(f"\nChunk {i + 1}")
    print("Rerank score:", score)
    print(chunk)


# 5. Chunk'ları LLM'e gönder
context = "\n\n".join([chunk for chunk, score in reranked])

prompt = f"""
Aşağıdaki bilgileri kullanarak kullanıcının sorusunu cevapla.
Cevap dokümanda yoksa bilmediğini söyle ama selamlama yapan olursa ona da cevap ver.

Bilgiler:
{context}

Soru:
{question}
"""

print("\nLLM'e gönderilen context:")
print(context)


response = chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nLLM cevabı:")
print(response.message.content)