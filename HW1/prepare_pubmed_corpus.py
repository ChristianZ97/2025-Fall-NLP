# 檔案名稱: prepare_pubmed_corpus.py

from datasets import load_dataset
from tqdm import tqdm
import sys

# 增加 Python 的 CSV 欄位大小限制，以防文章過長導致錯誤
import csv

csv.field_size_limit(sys.maxsize)

print("Downloading and loading the pubmed-summarization dataset...")
# 我們指定 'document' 子集，它包含完整的文章
# 這一步會從網路下載數據，可能需要一些時間和磁碟空間
try:
    dataset = load_dataset(
        "ccdv/pubmed-summarization", "document", trust_remote_code=True
    )
except Exception as e:
    print(
        f"Error loading dataset. Please check your internet connection or disk space. Error: {e}"
    )
    sys.exit()

output_filename = "pubmed_texts_combined.txt"
print(f"Processing articles and saving to '{output_filename}'...")

# 我們只使用訓練集 'train'，因為它包含了所有 13.3 萬篇文章
with open(output_filename, "w", encoding="utf-8") as f:
    # 使用 tqdm 顯示處理進度
    for entry in tqdm(dataset["train"], desc="Extracting articles"):
        article_text = entry.get("article")
        if article_text and isinstance(article_text, str):
            # 重要：將文章內部的所有換行符替換為空格
            # 確保最終輸出的檔案中，每篇文章只佔一行，與您現有的處理邏輯一致
            cleaned_text = article_text.replace("\n", " ").replace("\r", "")
            f.write(cleaned_text + "\n")

print(f"\nProcessing complete. Corpus saved to {output_filename}")
