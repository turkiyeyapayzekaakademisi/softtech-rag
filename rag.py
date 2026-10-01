# 0. vektör database yüklenmesi
#   Qwen3-Embedding-0.6B tanımlanması
#   llm belirlenmesi: ollama -> gemma3:4b 

# 1. kullanıcı soru sorar

# 2. kullanıcı sorusu embedding ile sayısal vektöre çevrilir. 

# 3. vektor databaseden 3 adet chunk getirilir. 

# 4. reranker eklenmesi 

# 5. getirilen chunlar llm e verilir ve llm cevap üretir. 

# NOT: her adımda çıktıları print olarak yazdır. 