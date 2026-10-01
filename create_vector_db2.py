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
    pip install pypdf langchain-text-splitters transformers torch faiss-cpu
"""

# =========================================================
# 0. GEREKLİ KÜTÜPHANELER
# =========================================================

import os
import json
import faiss
import torch
import torch.nn.functional as F

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel


# =========================================================
# 1. PDF DOSYASINI OKU
# =========================================================

pdf_path = "data/softtech_hakkinda.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


print("\n==============================")
print("PDF OKUNDU")
print("==============================")

print(text)


# =========================================================
# 2.1 KARAKTER BAZLI CHUNKING
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(

    # Her chunk maksimum 500 karakter
    chunk_size=500,

    # Chunk'lar arasında 100 karakter ortak alan
    chunk_overlap=100
)


chunks_character = text_splitter.split_text(text)


print("\n==============================")
print("KARAKTER BAZLI CHUNKING")
print("==============================")

print("Chunk sayısı:", len(chunks_character))


for i, chunk in enumerate(chunks_character):

    print(f"\nChunk {i + 1}")
    print("-" * 50)

    print(chunk)


# =========================================================
# 2.2 BAŞLIK BAZLI CHUNKING
# =========================================================

chunks_heading = []

current_chunk = ""


for line in text.splitlines():

    line = line.strip()

    # Boş satırları geç
    if not line:
        continue


    # Satır tamamen büyük harf ise
    # bunu başlık olarak kabul ediyoruz
    if line.isupper():

        # Önceki chunk varsa kaydet
        if current_chunk:

            chunks_heading.append(
                current_chunk.strip()
            )


        # Yeni chunk başlat
        current_chunk = line + "\n"


    else:

        # Normal metni mevcut chunk'a ekle
        current_chunk += line + " "


# Son chunk'ı ekle
if current_chunk:

    chunks_heading.append(
        current_chunk.strip()
    )


print("\n==============================")
print("BAŞLIK BAZLI CHUNKING")
print("==============================")

print("Chunk sayısı:", len(chunks_heading))


for i, chunk in enumerate(chunks_heading):

    print(f"\nChunk {i + 1}")
    print("-" * 50)

    print(chunk)


# =========================================================
# 3. QWEN EMBEDDING MODELİNİ YÜKLE
# =========================================================

model_name = "Qwen/Qwen3-Embedding-0.6B"


print("\n==============================")
print("EMBEDDING MODELİ YÜKLENİYOR")
print("==============================")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(

    model_name,

    # Qwen embedding modeli için left padding kullanıyoruz
    padding_side="left"
)


# Model
model = AutoModel.from_pretrained(
    model_name
)


# GPU varsa CUDA kullan
# yoksa CPU kullan
device = "cuda" if torch.cuda.is_available() else "cpu"


model = model.to(device)

model.eval()


print("Model:", model_name)

print("Kullanılan cihaz:", device)


# =========================================================
# 4. LAST TOKEN POOLING
# =========================================================

def last_token_pool(
    last_hidden_states,
    attention_mask
):

    """
    Transformer çıktısından cümleyi temsil edecek
    tek bir embedding vektörü oluşturur.

    Qwen3 Embedding modeli last-token pooling kullanır.
    """

    # Left padding kullanılıp kullanılmadığını kontrol et
    left_padding = (

        attention_mask[:, -1].sum()

        == attention_mask.shape[0]
    )


    # Left padding varsa
    # son token doğrudan kullanılabilir
    if left_padding:

        return last_hidden_states[:, -1]


    # Padding sağdaysa
    # her cümlenin gerçek son token pozisyonunu bul
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

def create_embeddings(
    texts,
    batch_size=8
):

    """
    Verilen metin listesini embedding vektörlerine çevirir.
    """

    all_embeddings = []


    # Batch batch işle
    for i in range(
        0,
        len(texts),
        batch_size
    ):

        batch_texts = texts[
            i:i + batch_size
        ]


        print(
            f"Embedding oluşturuluyor: "
            f"{i + 1} - "
            f"{min(i + batch_size, len(texts))}"
        )


        # Metni token'lara dönüştür
        inputs = tokenizer(

            batch_texts,

            padding=True,

            truncation=True,

            max_length=8192,

            return_tensors="pt"
        )


        # Tensorları CPU/GPU'ya taşı
        inputs = inputs.to(device)


        # Gradient hesaplamaya gerek yok
        with torch.no_grad():

            outputs = model(
                **inputs
            )


        # Transformer çıktısından embedding oluştur
        embeddings = last_token_pool(

            outputs.last_hidden_state,

            inputs["attention_mask"]
        )


        # Embedding vektörlerini normalize et
        embeddings = F.normalize(

            embeddings,

            p=2,

            dim=1
        )


        # ÖNEMLİ:
        #
        # Qwen bazı sistemlerde embedding çıktısını
        # BFloat16 olarak üretebilir.
        #
        # NumPy BFloat16 tipini doğrudan desteklemediği için
        # önce Float32'ye dönüştürüyoruz.

        embeddings = (
            embeddings
            .float()
            .cpu()
        )


        all_embeddings.append(
            embeddings
        )


    # Batch'leri tek tensor halinde birleştir
    all_embeddings = torch.cat(

        all_embeddings,

        dim=0
    )


    # NumPy array'e dönüştür
    return all_embeddings.numpy()


# =========================================================
# 6. CHARACTER CHUNK EMBEDDINGLERİ
# =========================================================

print("\n==============================")
print("CHARACTER EMBEDDING")
print("==============================")


character_embeddings = create_embeddings(
    chunks_character
)


print(
    "\nCharacter embedding shape:",
    character_embeddings.shape
)


print(
    "Character embedding dtype:",
    character_embeddings.dtype
)


# =========================================================
# 7. HEADING CHUNK EMBEDDINGLERİ
# =========================================================

print("\n==============================")
print("HEADING EMBEDDING")
print("==============================")


heading_embeddings = create_embeddings(
    chunks_heading
)


print(
    "\nHeading embedding shape:",
    heading_embeddings.shape
)


print(
    "Heading embedding dtype:",
    heading_embeddings.dtype
)


# =========================================================
# 8. VECTOR DB KLASÖRLERİNİ OLUŞTUR
# =========================================================

os.makedirs(

    "vector_db/character",

    exist_ok=True
)


os.makedirs(

    "vector_db/heading",

    exist_ok=True
)


# =========================================================
# 9. CHARACTER VECTOR DB
# =========================================================

print("\n==============================")
print("CHARACTER VECTOR DB")
print("==============================")


# Embedding boyutunu al
character_dimension = (
    character_embeddings.shape[1]
)


print(
    "Embedding dimension:",
    character_dimension
)


# Inner Product similarity index
#
# Embeddingler normalize edildiği için
# Inner Product = Cosine Similarity olur.
character_index = faiss.IndexFlatIP(
    character_dimension
)


# Embeddingleri FAISS'e ekle
character_index.add(

    character_embeddings.astype(
        "float32"
    )
)


print(
    "FAISS içindeki vektör sayısı:",
    character_index.ntotal
)


# Index'i kaydet
faiss.write_index(

    character_index,

    "vector_db/character/index.faiss"
)


# Chunk'ları JSON olarak kaydet
with open(

    "vector_db/character/chunks.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        chunks_character,

        f,

        ensure_ascii=False,

        indent=2
    )


print(
    "Character Vector DB kaydedildi."
)


# =========================================================
# 10. HEADING VECTOR DB
# =========================================================

print("\n==============================")
print("HEADING VECTOR DB")
print("==============================")


heading_dimension = (
    heading_embeddings.shape[1]
)


print(
    "Embedding dimension:",
    heading_dimension
)


heading_index = faiss.IndexFlatIP(
    heading_dimension
)


heading_index.add(

    heading_embeddings.astype(
        "float32"
    )
)


print(
    "FAISS içindeki vektör sayısı:",
    heading_index.ntotal
)


faiss.write_index(

    heading_index,

    "vector_db/heading/index.faiss"
)


with open(

    "vector_db/heading/chunks.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        chunks_heading,

        f,

        ensure_ascii=False,

        indent=2
    )


print(
    "Heading Vector DB kaydedildi."
)


# =========================================================
# 11. SONUÇ
# =========================================================

print("\n==============================")
print("İŞLEM TAMAMLANDI")
print("==============================")


print(
    "\nCharacter DB:"
)

print(
    "vector_db/character/index.faiss"
)

print(
    "vector_db/character/chunks.json"
)


print(
    "\nHeading DB:"
)

print(
    "vector_db/heading/index.faiss"
)

print(
    "vector_db/heading/chunks.json"
)