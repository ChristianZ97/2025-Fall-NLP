import os
import json
import pickle
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests


import torch
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Transformers for Embedding & Reranking
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Retrieval Components (Embedding, Indexing, Reranking) ---


class EmbeddingModel:
    """
    負責將文字轉換為向量。使用 SQLite 快取以避免重複計算。
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        cache_path: str = "embed_cache.db",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Embedding Model: {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )
        self.cache_db = cache_path
        self._init_cache()

    def _init_cache(self):
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, embedding BLOB)"
            )

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []

        # 1. Check Cache
        with sqlite3.connect(self.cache_db) as conn:
            cur = conn.cursor()
            for i, text in enumerate(texts):
                h = self._get_hash(text)
                row = cur.execute(
                    "SELECT embedding FROM cache WHERE hash=?", (h,)
                ).fetchone()
                if row:
                    embeddings.append(pickle.loads(row[0]))
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_compute.append(text)
                    indices_to_compute.append(i)

        # 2. Compute Misses
        if texts_to_compute:
            inputs = self.tokenizer(
                texts_to_compute, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                # Mean pooling
                out = (
                    self.model(**inputs)
                    .last_hidden_state.mean(dim=1)
                    .cpu()
                    .numpy()
                    .astype("float32")
                )

            # 3. Update Cache & Result List
            with sqlite3.connect(self.cache_db) as conn:
                for i, idx in enumerate(indices_to_compute):
                    emb = out[i]
                    embeddings[idx] = emb
                    conn.execute(
                        "INSERT OR REPLACE INTO cache VALUES (?,?)",
                        (self._get_hash(texts_to_compute[i]), pickle.dumps(emb)),
                    )

        return np.array(embeddings)


class VectorStore:
    """
    使用 Faiss 進行向量檢索 (IVF-PQ for efficiency)。
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        # 簡單起見使用 IndexFlatL2 (若資料量大可改用 IndexIVFFlat)
        self.index = faiss.IndexFlatL2(dim)
        self.doc_map = []  # Map index ID to (doc_id, text)

    def add(self, embeddings: np.ndarray, doc_ids: List[str], texts: List[str]):
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != Index dim {self.dim}"
            )

        self.index.add(embeddings)
        for did, txt in zip(doc_ids, texts):
            self.doc_map.append({"id": did, "text": txt})

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Dict]:
        scores, indices = self.index.search(query_emb.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc = self.doc_map[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
        return results


class KeywordRetriever:
    """
    傳統關鍵字檢索 (BM25)。
    """

    def __init__(self, texts: List[str], doc_ids: List[str]):
        tokenized_corpus = [
            text.split() for text in texts
        ]  # Simple whitespace tokenizer
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_ids = doc_ids
        self.texts = texts

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        scores = self.bm25.get_scores(query.split())
        top_n = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": self.doc_ids[i], "text": self.texts[i], "score": float(scores[i])}
            for i in top_n
        ]


class Reranker:
    """
    對檢索結果進行語意重排序 (BGE Reranker)。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        if not docs:
            return []

        pairs = [[query, d["text"]] for d in docs]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.view(-1).float()

        for i, score in enumerate(scores):
            docs[i]["score"] = score.item()

        # Sort by new score descending
        return sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]


# --- 2. Generation Components (LLM) ---


class LLMClient:
    """
    直接呼叫 vLLM 的 OpenAI-compatible /v1/chat/completions API。
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",  # 如果有設 --api-key 再換成真的 key
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,  # 明確給一個正整數，避免 max_tokens 計算成負數的 bug
            "stream": False,  # 先關掉 streaming，簡化問題
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"  # 如果 vLLM 有設 --api-key
            )

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )
            if resp.status_code != 200:
                return f"LLM Error: {resp.status_code} {resp.text}"
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM Error (client-side): {e}"


# --- 3. The RAG Pipeline ---


class RAGPipeline:
    def __init__(
        self,
        embedder: EmbeddingModel,
        store: VectorStore,
        llm: LLMClient,
        keyword_retriever: Optional[KeywordRetriever] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.bm25 = keyword_retriever
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        # 1. Dense Retrieval (Vector)
        q_emb = self.embedder.embed([query])[0]
        vector_results = self.store.search(q_emb, top_k=top_k)

        # 2. Sparse Retrieval (BM25) - Optional
        bm25_results = []
        if self.bm25:
            bm25_results = self.bm25.search(query, top_k=top_k)

        # 3. Merge & Deduplicate
        seen_ids = set()
        merged_results = []
        for res in vector_results + bm25_results:
            if res["id"] not in seen_ids:
                merged_results.append(res)
                seen_ids.add(res["id"])

        # 4. Rerank - Optional
        if self.reranker:
            merged_results = self.reranker.rerank(query, merged_results, top_k=top_k)
        else:
            merged_results = merged_results[:top_k]

        return merged_results

    def run(
        self,
        question: str,
        system_prompt: str,
        template: str,
        additional_info: Dict = {},
        top_k: int = 5,
    ) -> Dict:

        # A. Retrieve Context
        context_docs = self.retrieve(question, top_k=top_k)
        context_str = "\n\n".join(
            [f"Doc [{d['id']}]: {d['text']}" for d in context_docs]
        )

        # B. Format Prompt
        formatted_prompt = template.format(
            question=question,
            context=context_str,
            additional_info=json.dumps(additional_info, ensure_ascii=False),
        )

        # C. Generate Answer
        raw_answer = self.llm.generate(formatted_prompt, system_prompt)

        # D. Parse Output (Try JSON)
        try:
            json_start = raw_answer.find("{")
            json_end = raw_answer.rfind("}") + 1
            parsed_answer = json.loads(raw_answer[json_start:json_end])
        except:
            parsed_answer = {"error": "JSON parsing failed", "raw": raw_answer}

        return {
            "question": question,
            "answer": parsed_answer,
            "raw_response": raw_answer,
            "retrieved_docs": [d["id"] for d in context_docs],
        }


# --- 4. Main Execution ---

if __name__ == "__main__":
    # === Step 1: Data Preparation (Toy Example) ===
    """
    documents = [
        "Solar panels convert sunlight into electricity using photovoltaic cells.",
        "Wind turbines generate power by converting kinetic energy from wind.",
        "Hydroelectric dams use flowing water to spin turbines.",
        "Nuclear power plants use fission to generate heat and steam.",
    ]
    doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]
    """

    data_root = Path("./data/WattBot2025/download/texts")  # 和 data_preprocess.py 對應
    documents = []
    doc_ids = []

    if not data_root.exists():
        raise FileNotFoundError(f"Text directory not found: {data_root}")

    for txt_path in sorted(data_root.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            # 如果有亂碼就再試一個寬鬆編碼
            text = txt_path.read_text(encoding="latin1", errors="ignore")

        if not text.strip():
            continue  # 空文件就跳過

        documents.append(text)
        # 用檔名（不含 .txt）當作 doc_id，如 2405.01814
        doc_ids.append(txt_path.stem)

    print(f"Loaded {len(documents)} documents from {data_root}")


    # === Step 2: Initialize Components ===

    # A. Embedding
    embedder = EmbeddingModel(model_name="google/embeddinggemma-300m")

    # B. Vector Store
    doc_embeddings = embedder.embed(documents)
    vector_store = VectorStore(dim=doc_embeddings.shape[1])
    vector_store.add(doc_embeddings, doc_ids, documents)

    # C. Keyword Retriever & Reranker
    bm25 = KeywordRetriever(documents, doc_ids)
    reranker = Reranker(model_name="BAAI/bge-reranker-large")

    # D. LLM (Switch here for vLLM!)
    # 如果您用 vLLM，確保 server 已啟動: python -m vllm.entrypoints.openai.api_server ...
    # llm = LLMClient(
    #    model_name="gpt-4o-mini",  # 換成您的 vLLM 模型名稱 (e.g., "meta-llama/Llama-3-8b-instruct")
    #    base_url="https://api.openai.com/v1", # 若用 vLLM 改為 "http://localhost:8000/v1"
    #    api_key=os.getenv("OPENAI_API_KEY") # vLLM 通常不需要 key，傳 "EMPTY"
    # )

    llm = LLMClient(
        model_name="meta-llama/Llama-3.2-1B",  # 您的 vLLM 模型名稱
        base_url="http://localhost:8000/v1",  # 指向 vLLM server
        api_key="EMPTY",  # vLLM 不需要 key
    )

    # === Step 3: Build Pipeline ===
    pipeline = RAGPipeline(embedder, vector_store, llm, bm25, reranker)

    # === Step 4: Run Inference ===
    sys_prompt = "You are a helpful physics assistant. Respond in JSON format."
    user_template = """
    Question: {question}
    
    Context:
    {context}
    
    Metadata: {additional_info}
    
    Answer in JSON format containing 'answer' and 'explanation'.
    """

    result = pipeline.run(
        question="How do wind turbines work?",
        system_prompt=sys_prompt,
        template=user_template,
        top_k=2,
    )

    print("\n=== Result ===")
    print(json.dumps(result["answer"], indent=2, ensure_ascii=False))
    print(f"\nDocs Used: {result['retrieved_docs']}")
