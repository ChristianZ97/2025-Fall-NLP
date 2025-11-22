# full_test.py

import os
from pathlib import Path
import pandas as pd
import time
from tqdm.auto import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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

MAX_NUM_SEQS = 8  # 可依實際 vLLM / GPU 設定調整

DATA_DIR = Path("./data/WattBot2025")  # 和你 dummy code 一致
DOC_DIR = DATA_DIR / "download" / "texts"  # data_preprocess.py 產出 .txt 的資料夾

FALLBACK_ANSWER = "Unable to answer with confidence based on the provided documents."

# 讀 metadata.csv 建立 id -> url 映射
META_PATH = DATA_DIR / "metadata.csv"
meta_df = pd.read_csv(META_PATH, encoding="latin1")  # 和 data_preprocess 一致
ID2URL = dict(zip(meta_df["id"], meta_df["url"]))  # 若實際欄位名不同再調整


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
    records = []

    # 把原本 for 迴圈裡的內容幾乎原封不動搬進來
    def process_one(row, idx):
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

        raw_answer = result.get("answer", {})
        if isinstance(raw_answer, dict):
            parsed = raw_answer
        else:
            parsed = {}

        out = {}
        for col in expected_columns:
            if col == "id":
                out[col] = question_id
            elif col == "question":
                out[col] = question
            elif col == "answer":
                answer_text = parsed.get("answer")
                if not isinstance(answer_text, str) or not answer_text.strip():
                    answer_text = FALLBACK_ANSWER
                out[col] = answer_text
            elif col == "explanation":
                expl_text = parsed.get("explanation")
                if not isinstance(expl_text, str) or not expl_text.strip():
                    expl_text = "is_blank"
                out[col] = expl_text
            elif col == "answer_value":
                val = parsed.get("answer_value", "is_blank")
                if val is None or val == "":
                    out[col] = "is_blank"
                else:
                    if isinstance(val, bool):
                        val = 1 if val else 0
                    if isinstance(val, (list, dict)):
                        out[col] = json.dumps(val, ensure_ascii=False)
                    else:
                        out[col] = str(val)
            elif col == "answer_unit":
                unit = parsed.get("answer_unit", "is_blank")
                if not isinstance(unit, str) or not unit.strip():
                    unit = "is_blank"
                out[col] = unit
            elif col == "ref_id":
                ref_ids = parsed.get("ref_id", [])
                if isinstance(ref_ids, str):
                    sep = ";" if ";" in ref_ids else ","
                    ref_ids = [x.strip() for x in ref_ids.split(sep)]
                if not isinstance(ref_ids, list):
                    ref_ids = []
                ref_ids = [r for r in ref_ids if r and r in ID2URL]
                if not ref_ids:
                    out[col] = "is_blank"
                else:
                    out[col] = ";".join(ref_ids)
            elif col == "ref_url":
                ref_ids_cell = out.get("ref_id", "is_blank")
                if ref_ids_cell == "is_blank":
                    out[col] = "is_blank"
                else:
                    ids = [x.strip() for x in ref_ids_cell.split(";") if x.strip()]
                    urls = [ID2URL.get(i, "") for i in ids if ID2URL.get(i, "")]
                    out[col] = ";".join(urls) if urls else "is_blank"
            elif col == "supporting_materials":
                sup = parsed.get("supporting_materials", "is_blank")
                if not isinstance(sup, str) or not sup.strip():
                    sup = "is_blank"
                out[col] = sup
            else:
                out[col] = "is_blank"

        # 只在前 3 題印出 sample（注意：是以 DataFrame 的 index 為準）
        if idx < 3:
            print("\n")
            print(f"\n=== Sample #{idx} / id={question_id} ===")
            for c in expected_columns:
                print(f"{c}: {out[c]}")
            print("\n")
        return out

    # 平行跑所有 row
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_NUM_SEQS) as executor:
        for idx, (_, row) in enumerate(test_q.iterrows(), start=1):
            futures.append(executor.submit(process_one, row, idx))

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running full_test inference (parallel)",
        ):
            out = fut.result()
            records.append(out)

    submission = pd.DataFrame(records)
    submission = submission[expected_columns]
    submission = submission.fillna(
        {
            "answer": FALLBACK_ANSWER,
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": "is_blank",
            "ref_url": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "is_blank",
        }
    )

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
