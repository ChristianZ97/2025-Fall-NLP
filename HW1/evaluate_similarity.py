# evaluate_similarity_with_glove.py

import glob
import gensim.downloader
from gensim.models import Word2Vec

# --- 詞彙選擇 ---
target_words = ["king", "cell", "cancer", "science"]

# --- 模型清單 ---
# 手動加入 GloVe 模型
model_collection = {"glove-wiki-gigaword-100": None}

# 自動尋找本地的 .model 檔案
local_model_files = glob.glob("*.model")
for model_path in sorted(local_model_files):
    model_collection[model_path] = None

print(f"Found {len(local_model_files)} local models and 1 GloVe model to evaluate.\n")

# --- 遍歷所有模型進行評估 ---
for model_name in model_collection:
    print(f"{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    try:
        # 根據模型名稱決定載入方式
        if model_name.startswith("glove"):
            # 從 gensim API 下載並載入 GloVe
            model_wv = gensim.downloader.load(model_name)
        else:
            # 載入本地 Word2Vec 模型
            model = Word2Vec.load(model_name)
            model_wv = model.wv
    except Exception as e:
        print(f"  > Failed to load model {model_name}. Error: {e}")
        continue

    # 對每個目標詞彙進行分析
    for word in target_words:
        print(f"\n--- Most similar words for '{word}' ---")
        try:
            # 獲取最相似的5個詞
            similar_words = model_wv.most_similar(word, topn=5)

            # 格式化輸出
            for sim_word, score in similar_words:
                print(f"  - {sim_word:<20} (Score: {score:.4f})")

        except KeyError:
            # 如果詞彙不在模型的詞彙表中
            print(f"  > '{word}' not found in the vocabulary of this model.")
    print("\n")
