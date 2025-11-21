import os
from pathlib import Path
import pandas as pd
import time
from rag_pipeline import (
    EmbeddingModel,
    VectorStore,
    KeywordRetriever,
    Reranker,
    LLMClient,
    RAGPipeline,
)
from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# 0. 路徑設定（依你實際情況調整）
# ---------------------------------------------------------------------
DATA_DIR = Path("./data/WattBot2025")  # 和你 dummy code 一致
DOC_DIR = DATA_DIR / "download" / "texts"  # data_preprocess.py 產出 .txt 的資料夾

FALLBACK_ANSWER = "Unable to answer with confidence based on the provided documents."


# ---------------------------------------------------------------------
# 1. 載入文件：把所有 .txt 當成 RAG 的 knowledge base
# ---------------------------------------------------------------------
def load_text_documents(doc_dir: Path):
    documents = []
    doc_ids = []

    if not doc_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {doc_dir}")

    for txt_path in sorted(doc_dir.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            text = txt_path.read_text(encoding="latin1", errors="ignore")

        if not text.strip():
            continue

        documents.append(text)
        doc_ids.append(txt_path.stem)  # 檔名（不含 .txt）當作 doc_id，如 2405.01814

    print(f"Loaded {len(documents)} documents from {doc_dir}")
    return documents, doc_ids


# ---------------------------------------------------------------------
# 2. 建立 RAG pipeline（只做一次，避免在迴圈裡重複初始化）
# ---------------------------------------------------------------------
def build_rag_pipeline():
    # A. 載入文件
    documents, doc_ids = load_text_documents(DOC_DIR)

    # B. Embedding
    embedder = EmbeddingModel(model_name="google/embeddinggemma-300m")
    doc_embeddings = embedder.embed(documents)

    # C. Vector Store
    vector_store = VectorStore(dim=doc_embeddings.shape[1])
    vector_store.add(doc_embeddings, doc_ids, documents)

    # D. Keyword Retriever & Reranker
    bm25 = KeywordRetriever(documents, doc_ids)
    reranker = Reranker(model_name="BAAI/bge-reranker-large")

    # E. LLM Client（vLLM / OpenAI）
    llm = LLMClient(
        model_name="openai/gpt-oss-20b",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    pipeline = RAGPipeline(embedder, vector_store, llm, bm25, reranker)
    return pipeline


# ---------------------------------------------------------------------
# 3. 定義 system_prompt 與 template
#    （沿用你原本只輸出 answer + explanation 的 JSON）
# ---------------------------------------------------------------------
SYS_PROMPT = "You are a helpful physics assistant for the WattBot 2025 challenge. Respond in JSON format."

USER_TEMPLATE = """
Question: {question}

Context:
{context}

Metadata: {additional_info}

Answer in JSON format containing 'answer' and 'explanation'.
"""

# ---------------------------------------------------------------------
# 4. full_test：跑完整 test_Q.csv，產出 submission_full_test.csv
# ---------------------------------------------------------------------


def run_full_test():
    tqdm.pandas()
    train_qa = pd.read_csv(DATA_DIR / "train_QA.csv")
    test_q = pd.read_csv(DATA_DIR / "test_Q.csv")

    expected_columns = list(train_qa.columns)

    pipeline_build_start = time.time()
    pipeline = build_rag_pipeline()
    pipeline_build_end = time.time()
    print(f"Pipeline built in {pipeline_build_end - pipeline_build_start:.1f} seconds")

    def answer_one_row(row):
        question = row["question"]
        meta = {"id": row["id"]}

        result = pipeline.run(
            question=question,
            system_prompt=SYS_PROMPT,
            template=USER_TEMPLATE,
            additional_info=meta,
            top_k=5,
        )

        parsed = result.get("answer", {})

        out = {}
        for col in expected_columns:
            if col == "id":
                out[col] = row["id"]
            elif col == "question":
                out[col] = row["question"]
            elif col == "answer":
                out[col] = parsed.get("answer", FALLBACK_ANSWER)
            elif col == "explanation":
                out[col] = parsed.get("explanation", "is_blank")
            elif col in [
                "answer_value",
                "answer_unit",
                "ref_id",
                "ref_url",
                "supporting_materials",
            ]:
                out[col] = "is_blank"
            else:
                out[col] = "is_blank"

        return pd.Series(out)

    n_questions = len(test_q)
    print(f"Running full_test on {n_questions} questions...")

    start_time = time.time()

    # 關鍵：用 progress_apply 取代 apply，會顯示進度條+時間資訊
    submission = test_q.progress_apply(answer_one_row, axis=1)

    end_time = time.time()
    total_sec = end_time - start_time
    per_q = total_sec / max(n_questions, 1)

    print(f"Answering finished in {total_sec:.1f} seconds")
    print(f"Average {per_q:.2f} seconds per question")

    submission = submission[expected_columns]
    out_path = Path("./submission.csv")
    submission.to_csv(out_path, index=False)
    print("Saved full-test submission to", out_path)


# ---------------------------------------------------------------------
# 5. 實際執行
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_full_test()
