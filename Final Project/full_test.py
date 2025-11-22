# full_test.py

import os
from pathlib import Path
import pandas as pd
import time
from tqdm.auto import tqdm

from rag_pipeline import (
    EmbeddingModel,
    VectorStore,
    KeywordRetriever,
    Reranker,
    LLMClient,
    RAGPipeline,
)

from prompts import SYS_PROMPT, USER_TEMPLATE

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
# 4. full_test：跑完整 test_Q.csv，產出 submission_full_test.csv
# ---------------------------------------------------------------------
def run_full_test():
    train_qa = pd.read_csv(DATA_DIR / "train_QA.csv")
    test_q = pd.read_csv(DATA_DIR / "test_Q.csv")
    expected_columns = list(train_qa.columns)

    pipeline = build_rag_pipeline()

    records = []  # 先放在 list，最後再組成 DataFrame

    for idx, row in test_q.iterrows():
        question_id = row["id"]
        question = row["question"]

        meta = {"id": question_id}

        result = pipeline.run(
            question=question,
            system_prompt=SYS_PROMPT,
            template=USER_TEMPLATE,
            additional_info=meta,
            top_k=5,
        )

        # 安全解析：先統一成 dict
        raw_answer = result.get("answer", {})
        if isinstance(raw_answer, dict):
            parsed = raw_answer
        else:
            # 若模型直接回傳字串或亂格式，就當作沒成功解析
            parsed = {}

        out = {}
        for col in expected_columns:
            if col == "id":
                out[col] = question_id
            elif col == "question":
                out[col] = question
            elif col == "answer":
                # 保證是非空字串，否則用 FALLBACK_ANSWER
                answer_text = parsed.get("answer")
                if not isinstance(answer_text, str) or not answer_text.strip():
                    answer_text = FALLBACK_ANSWER
                out[col] = answer_text
            elif col == "explanation":
                expl_text = parsed.get("explanation")
                if not isinstance(expl_text, str) or not expl_text.strip():
                    expl_text = "is_blank"
                out[col] = expl_text
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

        # 每隔 N 筆印一次，確認回應正常
        N = 20  # 例如每 20 題看一次
        if (idx % N) == 0:
            print(f"\n=== Sample #{idx} / id={question_id} ===")
            print("Q:", question)
            print("Model answer:", out["answer"])
            print("Explanation:", out["explanation"])

        records.append(out)


    # 組成 DataFrame 並確保欄位順序
    submission = pd.DataFrame(records)
    submission = submission[expected_columns]

    # 最終安全檢查：把所有可能的 NaN 補成字串
    submission = submission.fillna({
        "answer": FALLBACK_ANSWER,
        "answer_value": "is_blank",
        "answer_unit": "is_blank",
        "ref_id": "is_blank",
        "ref_url": "is_blank",
        "supporting_materials": "is_blank",
        "explanation": "is_blank",
    })

    out_path = Path("./submission.csv")
    submission.to_csv(out_path, index=False)
    print("Saved full-test submission to", out_path)
    validate_submission_csv(out_path, expected_columns)


def validate_submission_csv(path: Path, expected_columns):
    df = pd.read_csv(path)

    # 1. 欄位名稱 / 順序完全相同
    assert list(df.columns) == list(expected_columns), "CSV columns mismatch!"

    # 2. 不允許任何 NaN / None
    assert not df.isnull().any().any(), "Submission contains null/NaN values!"

    # 3. 筆數要等於 test 集大小（可選：用 test_q.shape[0] 傳進來檢查）
    print(f"Submission rows: {len(df)}")
    print(df.head(3))



# ---------------------------------------------------------------------
# 5. 實際執行
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_full_test()
