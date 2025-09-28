# evaluate_similarity.py

import glob
from gensim.models import Word2Vec

# --- 詞彙選擇 ---
# 1. 通用詞彙 (預期在 PubMed 中不存在或關聯性弱)
# 2. 歧義詞 (在不同領域有不同意義)
# 3. 專業詞彙 (預期在 PubMed 中有更精確的關聯)
target_words = ['king', 'cell', 'cancer', 'science']

# 自動尋找目錄下所有的 .model 檔案
model_files = glob.glob('*.model')

print(f"Found {len(model_files)} models to evaluate: {model_files}\n")

# 遍歷每個模型
for model_path in sorted(model_files):
    print(f"{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}")
    
    # 載入模型
    model = Word2Vec.load(model_path)
    
    # 對每個目標詞彙進行分析
    for word in target_words:
        print(f"\n--- Most similar words for '{word}' ---")
        try:
            # 獲取最相似的5個詞
            similar_words = model.wv.most_similar(word, topn=5)
            
            # 格式化輸出
            for sim_word, score in similar_words:
                print(f"  - {sim_word:<20} (Score: {score:.4f})")
                
        except KeyError:
            # 如果詞彙不在模型的詞彙表中
            print(f"  > '{word}' not found in the vocabulary of this model.")
    print("\n")

