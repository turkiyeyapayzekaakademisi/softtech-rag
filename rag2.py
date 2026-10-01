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
    pip install --upgrade "transformers>=4.51.0" torch faiss-cpu ollama

Ollama:
    ollama pull gemma3:4b
"""


# =========================================================
# 0. KÜTÜPHANELER
# =========================================================

import json
import faiss
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM
)

from ollama import chat


# =========================================================
# 1. CİHAZ
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Kullanılan cihaz:", device)


# =========================================================
# 2. VECTOR DB'Yİ YÜKLE
# =========================================================

index = faiss.read_index(
    "vector_db/character/index.faiss"
)


with open(
    "vector_db/character/chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print("\n==============================")
print("VECTOR DB YÜKLENDİ")
print("==============================")

print("Vector sayısı:", index.ntotal)

print("Chunk sayısı:", len(chunks))


# =========================================================
# 3. EMBEDDING MODELİNİ YÜKLE
# =========================================================

embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"


print("\n==============================")
print("EMBEDDING MODELİ YÜKLENİYOR")
print("==============================")


embedding_tokenizer = AutoTokenizer.from_pretrained(
    embedding_model_name,
    padding_side="left"
)


embedding_model = AutoModel.from_pretrained(
    embedding_model_name
)


embedding_model = embedding_model.to(device)

embedding_model.eval()


print("Embedding modeli yüklendi:")
print(embedding_model_name)


# =========================================================
# 4. LAST TOKEN POOLING
# =========================================================

def last_token_pool(
    last_hidden_states,
    attention_mask
):

    """
    Transformer çıktısından tek bir embedding
    vektörü oluşturur.
    """

    left_padding = (
        attention_mask[:, -1].sum()
        == attention_mask.shape[0]
    )


    if left_padding:

        return last_hidden_states[:, -1]


    sequence_lengths = (
        attention_mask.sum(dim=1) - 1
    )


    batch_size = (
        last_hidden_states.shape[0]
    )


    return last_hidden_states[

        torch.arange(
            batch_size,
            device=last_hidden_states.device
        ),

        sequence_lengths
    ]


# =========================================================
# 5. EMBEDDING OLUŞTURMA FONKSİYONU
# =========================================================

def create_embedding(text):

    """
    Tek bir metni Qwen3-Embedding-0.6B
    kullanarak embedding'e dönüştürür.
    """

    inputs = embedding_tokenizer(

        [text],

        padding=True,

        truncation=True,

        max_length=8192,

        return_tensors="pt"
    )


    inputs = inputs.to(device)


    with torch.no_grad():

        outputs = embedding_model(
            **inputs
        )


    embedding = last_token_pool(

        outputs.last_hidden_state,

        inputs["attention_mask"]
    )


    # Cosine similarity için normalize et
    embedding = F.normalize(

        embedding,

        p=2,

        dim=1
    )


    # BFloat16 problemi yaşamamak için
    # Float32'ye çevir
    embedding = (
        embedding
        .float()
        .cpu()
        .numpy()
    )


    return embedding


# =========================================================
# 6. KULLANICIDAN SORU AL
# =========================================================

question = input(
    "\nSorunuz: "
)


print("\n==============================")
print("KULLANICI SORUSU")
print("==============================")

print(question)


# =========================================================
# 7. SORUYU EMBEDDING'E ÇEVİR
# =========================================================

question_embedding = create_embedding(
    question
)


print("\n==============================")
print("SORU EMBEDDING")
print("==============================")

print(
    "Embedding shape:",
    question_embedding.shape
)

print(
    "Embedding dtype:",
    question_embedding.dtype
)


# =========================================================
# 8. VECTOR DB'DEN EN YAKIN 3 CHUNK'I GETİR
# =========================================================

top_k = 3


scores, indices = index.search(

    question_embedding.astype("float32"),

    top_k
)


print("\n==============================")
print("RETRIEVAL SONUÇLARI")
print("==============================")


retrieved_chunks = []


for i in range(top_k):

    chunk_index = int(
        indices[0][i]
    )

    similarity = float(
        scores[0][i]
    )


    chunk = chunks[
        chunk_index
    ]


    retrieved_chunks.append(
        chunk
    )


    print(
        f"\nChunk {i + 1}"
    )

    print(
        "FAISS index:",
        chunk_index
    )

    print(
        "Similarity:",
        similarity
    )

    print(
        "\nMetin:"
    )

    print(chunk)


# =========================================================
# 9. EMBEDDING MODELİNİ BELLEKTEN ÇIKAR
# =========================================================

# Bundan sonra embedding modeline ihtiyacımız yok.
# GPU belleğini boşaltabiliriz.

del embedding_model


if torch.cuda.is_available():

    torch.cuda.empty_cache()


print(
    "\nEmbedding modeli bellekten çıkarıldı."
)


# =========================================================
# 10. RERANKER MODELİNİ YÜKLE
# =========================================================

reranker_model_name = (
    "Qwen/Qwen3-Reranker-0.6B"
)


print("\n==============================")
print("RERANKER YÜKLENİYOR")
print("==============================")


reranker_tokenizer = (
    AutoTokenizer.from_pretrained(

        reranker_model_name,

        padding_side="left"
    )
)


reranker_model = (
    AutoModelForCausalLM.from_pretrained(
        reranker_model_name
    )
)


reranker_model = (
    reranker_model.to(device)
)


reranker_model.eval()


print(
    "Reranker modeli yüklendi:"
)

print(
    reranker_model_name
)


# =========================================================
# 11. RERANKER AYARLARI
# =========================================================

# Model "yes" ve "no" tokenlarının olasılığına
# bakarak dokümanın soruyla ne kadar ilgili olduğunu
# hesaplayacak.

token_false_id = (
    reranker_tokenizer
    .convert_tokens_to_ids("no")
)


token_true_id = (
    reranker_tokenizer
    .convert_tokens_to_ids("yes")
)


max_length = 8192


# Qwen'in reranker için kullandığı sistem promptu
prefix = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n"
    "<|im_start|>user\n"
)


suffix = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n"
    "</think>\n\n"
)


prefix_tokens = (
    reranker_tokenizer.encode(
        prefix,
        add_special_tokens=False
    )
)


suffix_tokens = (
    reranker_tokenizer.encode(
        suffix,
        add_special_tokens=False
    )
)


# Reranker'a görevin ne olduğunu söylüyoruz.
# Qwen, instruction'ın İngilizce verilmesini öneriyor.
task = (
    "Given a user question, retrieve relevant passages "
    "that answer the question."
)


# =========================================================
# 12. RERANKER INPUT FORMAT
# =========================================================

def format_reranker_input(
    question,
    document
):

    return (
        f"<Instruct>: {task}\n"
        f"<Query>: {question}\n"
        f"<Document>: {document}"
    )


# =========================================================
# 13. RERANKER INPUT HAZIRLAMA
# =========================================================

def prepare_reranker_inputs(
    question,
    documents
):

    pairs = []


    for document in documents:

        text = format_reranker_input(

            question,

            document
        )

        pairs.append(
            text
        )


    inputs = reranker_tokenizer(

        pairs,

        padding=False,

        truncation="longest_first",

        return_attention_mask=False,

        max_length=(
            max_length
            - len(prefix_tokens)
            - len(suffix_tokens)
        )
    )


    # Her inputun başına ve sonuna
    # gerekli Qwen prompt tokenlarını ekle
    for i, input_ids in enumerate(
        inputs["input_ids"]
    ):

        inputs["input_ids"][i] = (

            prefix_tokens
            + input_ids
            + suffix_tokens

        )


    # Batch padding
    inputs = reranker_tokenizer.pad(

        inputs,

        padding=True,

        return_tensors="pt",

        max_length=max_length
    )


    # CPU / GPU
    inputs = {

        key: value.to(device)

        for key, value
        in inputs.items()
    }


    return inputs


# =========================================================
# 14. RERANK SCORE HESAPLAMA
# =========================================================

@torch.no_grad()
def calculate_rerank_scores(
    inputs
):

    outputs = reranker_model(
        **inputs
    )


    # Son token'ın tüm vocabulary logits
    batch_scores = (
        outputs.logits[:, -1, :]
    )


    # "yes" token score
    true_vector = (
        batch_scores[
            :,
            token_true_id
        ]
    )


    # "no" token score
    false_vector = (
        batch_scores[
            :,
            token_false_id
        ]
    )


    # no / yes skorlarını birleştir
    batch_scores = torch.stack(

        [
            false_vector,
            true_vector
        ],

        dim=1
    )


    # Probability'ye dönüştür
    batch_scores = (
        F.log_softmax(
            batch_scores,
            dim=1
        )
    )


    # "yes" olasılığını al
    scores = (
        batch_scores[:, 1]
        .exp()
        .float()
        .cpu()
        .tolist()
    )


    return scores


# =========================================================
# 15. RERANKER'I ÇALIŞTIR
# =========================================================

reranker_inputs = (
    prepare_reranker_inputs(

        question,

        retrieved_chunks
    )
)


rerank_scores = (
    calculate_rerank_scores(
        reranker_inputs
    )
)


# Chunk ve reranker score'u eşleştir
reranked = list(

    zip(

        retrieved_chunks,

        rerank_scores
    )

)


# Score büyükten küçüğe
reranked = sorted(

    reranked,

    key=lambda x: x[1],

    reverse=True
)


print("\n==============================")
print("RERANKER SONUÇLARI")
print("==============================")


for i, (chunk, score) in enumerate(
    reranked
):

    print(
        f"\nChunk {i + 1}"
    )

    print(
        "Rerank score:",
        score
    )

    print(
        "\nMetin:"
    )

    print(
        chunk
    )


# =========================================================
# 16. RERANKER MODELİNİ BELLEKTEN ÇIKAR
# =========================================================

del reranker_model


if torch.cuda.is_available():

    torch.cuda.empty_cache()


print(
    "\nReranker modeli bellekten çıkarıldı."
)


# =========================================================
# 17. LLM CONTEXT OLUŞTUR
# =========================================================

context = "\n\n".join(

    [
        chunk

        for chunk, score
        in reranked
    ]
)


print("\n==============================")
print("LLM'E GÖNDERİLEN CONTEXT")
print("==============================")

print(context)


# =========================================================
# 18. PROMPT
# =========================================================

prompt = f"""
Aşağıdaki bilgileri kullanarak kullanıcının sorusunu cevapla.

Kurallar:
- Öncelikle verilen bilgilerden yararlan.
- Cevap bilgiler içerisinde yoksa bilmediğini söyle.
- Bilgi uydurma.
- Kullanıcı selamlama yaparsa selamlamaya cevap ver.

Bilgiler:
{context}

Kullanıcı sorusu:
{question}
"""


# =========================================================
# 19. OLLAMA / GEMMA 3
# =========================================================

print("\n==============================")
print("GEMMA 3 CEVAP ÜRETİYOR")
print("==============================")


response = chat(

    model="gemma3:4b",

    messages=[

        {
            "role": "user",
            "content": prompt
        }

    ]
)


# =========================================================
# 20. SONUÇ
# =========================================================

print("\n==============================")
print("LLM CEVABI")
print("==============================")

print(
    response.message.content
)