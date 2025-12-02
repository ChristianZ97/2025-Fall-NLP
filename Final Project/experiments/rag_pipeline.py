# rag_pipeline.py

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
from openai import OpenAI  # Note: imported but not used directly in this file.

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Maximum number of characters from each retrieved document to feed into the LLM.
# This prevents extremely long contexts that might exceed model limits or waste tokens.
MAX_DOC_CHARS = 16384


class EmbeddingModel:
    """
    Wrapper around a Hugging Face embedding model with an on-disk SQLite cache.

    Features
    --------
    - Uses a transformer model (e.g., jinaai/jina-embeddings-v3) to generate embeddings.
    - Maintains an SQLite database to cache embeddings and avoid recomputation.
    - Supports batch embedding of multiple texts.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        cache_path: str = "embed_cache.db",
    ):
        """
        Initialize the embedding model and its cache.

        Parameters
        ----------
        model_name : str
            Name of the Hugging Face model to load.
        cache_path : str
            Path to the SQLite database used to cache embeddings.
        """
        # Use GPU if available, else default to CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Embedding Model: {model_name} on {self.device}...")

        # Load tokenizer and model; trust_remote_code allows custom model implementations.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )

        # Path (file name) for SQLite caching database.
        self.cache_db = cache_path
        self._init_cache()

    def _init_cache(self) -> None:
        """
        Initialize the SQLite cache table if it does not exist.

        Table schema
        ------------
        cache(
            hash TEXT PRIMARY KEY,  -- MD5 hash of the text
            embedding BLOB          -- Pickled numpy array
        )
        """
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, embedding BLOB)"
            )

    def _get_hash(self, text: str) -> str:
        """
        Compute a stable hash for a piece of text, used as the cache key.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Hex-encoded MD5 hash of the text.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def embed(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Embed a list of texts into dense vectors, using an on-disk SQLite cache.

        Parameters
        ----------
        texts : List[str]
            List of raw text strings to embed.
        batch_size : int, optional
            Number of texts to encode per forward pass. Smaller values reduce
            GPU memory usage at the cost of speed.
        max_length : int, optional
            Maximum number of tokens per text. Longer texts will be truncated
            by the tokenizer. This helps control memory usage.

        Returns
        -------
        np.ndarray
            2D numpy array of shape (len(texts), embedding_dim) with dtype float32.
        """
        # Placeholder list for final embeddings (some may be filled from cache).
        embeddings: List[Optional[np.ndarray]] = [None] * len(texts)

        # Lists to track which texts are missing from cache.
        indices_to_compute: List[int] = []
        texts_to_compute: List[str] = []

        # Step 1: check the cache for each text.
        with sqlite3.connect(self.cache_db) as conn:
            cur = conn.cursor()
            for i, text in enumerate(texts):
                h = self._get_hash(text)
                row = cur.execute(
                    "SELECT embedding FROM cache WHERE hash=?", (h,)
                ).fetchone()
                if row is not None:
                    # Cache hit: unpickle stored embedding.
                    embeddings[i] = pickle.loads(row[0])
                else:
                    # Cache miss: remember index & text for later computation.
                    indices_to_compute.append(i)
                    texts_to_compute.append(text)

        # Step 2: compute embeddings for cache-miss texts in mini-batches.
        if indices_to_compute:
            self.model.eval()

            with torch.no_grad():
                with sqlite3.connect(self.cache_db) as conn:
                    for start in range(0, len(indices_to_compute), batch_size):
                        end = start + batch_size
                        batch_indices = indices_to_compute[start:end]
                        batch_texts = [texts[i] for i in batch_indices]

                        # Tokenize this mini-batch with padding & truncation.
                        inputs = self.tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                        ).to(self.device)

                        # Forward pass through the model.
                        # Here we perform simple mean pooling over the last hidden state.
                        outputs = self.model(**inputs)
                        #batch_embs = (
                        #    outputs.last_hidden_state.mean(dim=1)
                        #    cpu()
                        #    .numpy()
                        #    .dtype(float32)
                        #)
                        batch_embs = torch(outputs.last_hidden_state.mean(dim=1), dtype=float32).cpu().numpy()


                        # Store results in memory and in the SQLite cache.
                        for local_idx, global_idx in enumerate(batch_indices):
                            emb = batch_embs[local_idx]
                            embeddings[global_idx] = emb
                            h = self._get_hash(texts[global_idx])
                            conn.execute(
                                "INSERT OR REPLACE INTO cache VALUES (?, ?)",
                                (h, pickle.dumps(emb)),
                            )

                        # Optional: free unused CUDA memory to reduce fragmentation.
                        if self.device == "cuda":
                            torch.cuda.empty_cache()

        # At this point all entries in 'embeddings' should be non-None.
        return np.stack(embeddings, axis=0).astype("float32")


class VectorStore:
    """
    Simple FAISS-based vector store using L2 similarity (IndexFlatL2).

    Features
    --------
    - Stores vectors in a FAISS index for efficient similarity search.
    - Maintains a parallel Python list 'doc_map' to map from FAISS index positions
      back to document IDs and text.
    """

    def __init__(self, dim: int = 1024):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the embeddings.
            Must match the output dimension of the embedding model.
        """
        self.dim = dim
        # IndexFlatL2 is a basic, non-compressed, exact L2-distance index.
        self.index = faiss.IndexFlatL2(dim)
        # doc_map[i] stores metadata for the vector at position i in the FAISS index.
        self.doc_map: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, doc_ids: List[str], texts: List[str]) -> None:
        """
        Add new vectors and their metadata to the vector store.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape (n, dim).
        doc_ids : List[str]
            Identifiers for each embedding (one per row).
        texts : List[str]
            Original text corresponding to each embedding.
        """
        # Sanity check: dimension must match index dimension.
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != Index dim {self.dim}"
            )

        # Add embeddings to FAISS index (in memory).
        self.index.add(embeddings)

        # Add metadata entries in the parallel Python list.
        for did, txt in zip(doc_ids, texts):
            # 'base_id' strips off any fragment after '#', if present.
            base_id = did.split("#", 1)[0]
            self.doc_map.append(
                {
                    "id": did,  # Possibly includes section or chunk info.
                    "doc_id": base_id,  # Base document ID (without fragment).
                    "text": txt,  # Raw text content.
                }
            )

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Perform a vector similarity search against the FAISS index.

        Parameters
        ----------
        query_emb : np.ndarray
            1D embedding vector representing the query (shape: (dim,)).
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        List[Dict]
            List of document metadata dicts, each augmented with a 'score' field
            representing the L2 distance (lower is more similar).
        """
        # FAISS expects a 2D array; reshape the query to (1, dim).
        scores, indices = self.index.search(query_emb.reshape(1, -1), top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 for invalid entries if fewer vectors are in the index.
            if idx != -1:
                doc = self.doc_map[idx].copy()
                # Score is the L2 distance; lower means closer. Consumers may want to invert.
                doc["score"] = float(score)
                results.append(doc)
        return results


class KeywordRetriever:
    """
    Classic keyword-based retriever using BM25Okapi.

    Intended to complement vector retrieval with lexical matching.

    Attributes
    ----------
    bm25 : BM25Okapi
        Underlying BM25 model over the tokenized corpus.
    doc_ids : List[str]
        IDs corresponding to indexed texts.
    texts : List[str]
        Original documents.
    """

    def __init__(self, texts: List[str], doc_ids: List[str]):
        """
        Build a BM25 index over the given corpus of texts.

        Parameters
        ----------
        texts : List[str]
            List of document texts.
        doc_ids : List[str]
            IDs for each text, matched by position.
        """
        # Very simple tokenization by whitespace split.
        # For more advanced usage, you may plug in a real tokenizer.
        tokenized_corpus = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_ids = doc_ids
        self.texts = texts

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Compute BM25 scores for all documents and return the top-k.

        Parameters
        ----------
        query : str
            User query string.
        top_k : int
            Maximum number of results.

        Returns
        -------
        List[Dict]
            Each dict contains fields:
            - 'id': document ID
            - 'text': document text
            - 'score': BM25 score (higher is better)
        """
        # Again, simple whitespace split as tokenizer.
        scores = self.bm25.get_scores(query.split())
        # argsort in descending order by scores to get best matches.
        top_n = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": self.doc_ids[i], "text": self.texts[i], "score": float(scores[i])}
            for i in top_n
        ]


class Reranker:
    """
    Neural reranker based on a cross-encoder model (e.g., BAAI/bge-reranker-large).

    Use case
    --------
    - Take a query and a list of candidate documents.
    - Score each (query, document) pair with a sequence classification model.
    - Return the candidates sorted by this model score.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        Load the cross-encoder reranker model.

        Parameters
        ----------
        model_name : str
            Hugging Face model name for the reranker.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        # Put model in evaluation mode to disable dropout etc.
        self.model.eval()

    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """
        Rerank a list of candidate document dicts according to cross-encoder scores.

        Parameters
        ----------
        query : str
            User query string.
        docs : List[Dict]
            Documents to rerank, each expected to contain a 'text' field.
        top_k : int
            Number of top documents to return after reranking.

        Returns
        -------
        List[Dict]
            Top-k documents, sorted by model score in descending order. Each doc dict
            is updated in-place to include a 'score' key.
        """
        if not docs:
            return []

        # Build input pairs: [query, document_text]
        pairs = [[query, d["text"]] for d in docs]

        # Tokenize the pairs jointly (as cross-encoder expects).
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)

        # Forward pass to obtain logits as scores.
        with torch.no_grad():
            scores = self.model(**inputs).logits.view(-1).float()

        # Attach scores back to docs
        for i, score in enumerate(scores):
            docs[i]["score"] = score.item()

        # Sort by descending score and return top_k
        return sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]


class LLMClient:
    """
    Minimal HTTP client to talk to an OpenAI-compatible /chat/completions endpoint.

    It does not depend on the official OpenAI Python SDK and instead uses `requests`,
    which makes it suitable for custom or self-hosted LLM servers that emulate
    the OpenAI API.

    Attributes
    ----------
    model_name : str
        Model identifier to send in the request payload.
    base_url : str
        Base URL of the API (e.g., http://localhost:8000/v1).
    api_key : str
        Bearer token for authentication (if required).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ):
        """
        Parameters
        ----------
        model_name : str
            The model name that the backend expects.
        base_url : str
            Base URL of the OpenAI-compatible API.
        api_key : str
            Optional API key (use 'EMPTY' if not needed).
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """
        Call the /chat/completions endpoint with a user + optional system prompt.

        Parameters
        ----------
        user_prompt : str
            The main user query or instruction.
        system_prompt : str
            Optional system message that defines assistant behavior.

        Returns
        -------
        str
            The assistant's response content. In case of error, returns an error string.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 8192,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )

            if resp.status_code != 200:
                # Print some diagnostics for debugging HTTP/API issues.
                print("\n=== LLM HTTP Error ===")
                print(f"Status: {resp.status_code}")
                print("Response body:")
                print(resp.text[:2000])
                print("=== End LLM HTTP Error ===\n")
                return f"LLM Error: {resp.status_code} {resp.text}"

            data = resp.json()
            # OpenAI-style response: choices[0].message.content
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            # Catch networking / JSON / other client-side exceptions.
            print("\n=== LLM Client Exception ===")
            print(repr(e))
            print("=== End LLM Client Exception ===\n")
            return f"LLM Error (client-side): {e}"


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.

    Components
    ----------
    - embedder : EmbeddingModel
        Used to embed user queries.
    - store : VectorStore
        FAISS-based vector index for semantic retrieval.
    - llm : LLMClient
        Client to talk to an LLM server.
    - bm25 : Optional[KeywordRetriever]
        Optional lexical retriever to complement semantic retrieval.
    - reranker : Optional[Reranker]
        Optional neural reranker to refine combined candidate list.

    Methods
    -------
    - retrieve(query, top_k):
        Returns a list of retrieved document dicts.
    - run(question, system_prompt, template, additional_info, top_k):
        Performs full RAG: retrieval + prompting LLM + JSON parsing.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        store: VectorStore,
        llm: LLMClient,
        keyword_retriever: Optional[KeywordRetriever] = None,
        reranker: Optional[Reranker] = None,
    ):
        """
        Parameters
        ----------
        embedder : EmbeddingModel
            Embedding model used for semantic search.
        store : VectorStore
            Vector index containing embedded documents.
        llm : LLMClient
            Client used to query the language model.
        keyword_retriever : Optional[KeywordRetriever]
            Additional keyword-based retriever (BM25).
        reranker : Optional[Reranker]
            Cross-encoder reranker applied on merged candidates.
        """
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.bm25 = keyword_retriever
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve relevant documents for a query using:
        1) Vector similarity search.
        2) Optional BM25 lexical search.
        3) Optional cross-encoder reranking.

        Parameters
        ----------
        query : str
            User query string.
        top_k : int
            Target number of final results.

        Returns
        -------
        List[Dict]
            List of retrieved document dicts with 'id', 'text', 'score', and 'doc_id'.
        """
        # Embed the query once to perform vector search.
        q_emb = self.embedder.embed([query])[0]
        vector_results = self.store.search(q_emb, top_k=top_k)

        bm25_results: List[Dict[str, Any]] = []
        if self.bm25:
            # Optionally get lexical retrieval results as well.
            bm25_results = self.bm25.search(query, top_k=top_k)

        # Merge results from both retrievers, ensuring unique IDs.
        seen_ids = set()
        merged_results: List[Dict[str, Any]] = []
        for res in vector_results + bm25_results:
            if res["id"] not in seen_ids:
                merged_results.append(res)
                seen_ids.add(res["id"])

        # Optionally rerank with a more expensive model.
        if self.reranker:
            merged_results = self.reranker.rerank(query, merged_results, top_k=top_k)
        else:
            # Otherwise, just truncate to top_k by original ordering.
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
        """
        Run the full RAG process for a single question.

        Steps
        -----
        1. Retrieve top-k context documents.
        2. Format a prompt using a user-supplied template.
        3. Call the LLM via LLMClient.
        4. Try to parse the LLM response as JSON (for structured answers).

        Parameters
        ----------
        question : str
            User's question for the RAG system.
        system_prompt : str
            System message to control the LLM's behavior.
        template : str
            Template string for building the final user prompt; must contain
            placeholders {question}, {context}, and {additional_info}.
        additional_info : Dict
            Additional metadata or parameters to pass into the prompt (JSON-encoded).
        top_k : int
            Number of documents to use as context.

        Returns
        -------
        Dict
            Dictionary containing:
            - 'question': original question string.
            - 'answer': parsed JSON from the LLM (or error structure).
            - 'raw_response': raw text returned by the LLM.
            - 'retrieved_docs': list of document IDs used as context.
        """
        # Step 1: retrieve relevant documents
        context_docs = self.retrieve(question, top_k=top_k)

        # Helper to decide how to label each doc in the final context string.
        def _label(d: Dict[str, Any]) -> str:
            # Prefer base doc_id if present; otherwise use full id.
            return d.get("doc_id") or d["id"]

        # Build a textual context with doc labels and truncated text.
        context_str = "\n\n".join(
            [f"Doc [{_label(d)}]: {d['text'][:MAX_DOC_CHARS]}" for d in context_docs]
        )

        # Step 2: fill in the provided template.
        formatted_prompt = template.format(
            question=question,
            context=context_str,
            additional_info=json.dumps(additional_info, ensure_ascii=False),
        )

        # Step 3: query the LLM.
        raw_answer = self.llm.generate(formatted_prompt, system_prompt)

        # Step 4: try to parse the response as JSON (robust to extra text).
        try:
            # Heuristic: take the substring from first '{' to last '}'.
            json_start = raw_answer.find("{")
            json_end = raw_answer.rfind("}") + 1
            parsed_answer = json.loads(raw_answer[json_start:json_end])
        except Exception:
            # Fallback: return raw answer for debugging if JSON fails.
            parsed_answer = {"error": "JSON parsing failed", "raw": raw_answer}

        return {
            "question": question,
            "answer": parsed_answer,
            "raw_response": raw_answer,
            "retrieved_docs": [d["id"] for d in context_docs],
        }


if __name__ == "__main__":
    """
    Minimal example to demonstrate usage of the RAG pipeline.

    It builds a tiny in-memory corpus of energy-related documents,
    creates all components (EmbeddingModel, VectorStore, BM25, Reranker, LLMClient),
    and then runs a sample query.
    """

    # Example corpus of 4 short documents.
    documents = [
        "Solar panels convert sunlight into electricity using photovoltaic cells.",
        "Wind turbines generate power by converting kinetic energy from wind.",
        "Hydroelectric dams use flowing water to spin turbines.",
        "Nuclear power plants use fission to generate heat and steam.",
    ]
    doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]

    # Initialize the embedding model.
    # Note: Using "google/embeddinggemma-300m" here as an example; make sure this
    # model is available and supports embedding usage.
    embedder = EmbeddingModel()

    # Compute embeddings for all documents.
    doc_embeddings = embedder.embed(documents)

    # Create a vector store with the correct embedding dimensionality.
    vector_store = VectorStore(dim=doc_embeddings.shape[1])
    # Add document embeddings and metadata into the vector store.
    vector_store.add(doc_embeddings, doc_ids, documents)

    # Initialize BM25 keyword retriever for the same documents.
    bm25 = KeywordRetriever(documents, doc_ids)

    # Initialize cross-encoder reranker model.
    reranker = Reranker(model_name="BAAI/bge-reranker-large")

    # Create an LLM client pointing to a local OpenAI-compatible endpoint.
    llm = LLMClient(
        model_name="openai/gpt-oss-20b",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Assemble the RAG pipeline with all components.
    pipeline = RAGPipeline(embedder, vector_store, llm, bm25, reranker)

    # System prompt to steer the model behavior; asks for JSON responses.
    sys_prompt = "You are a helpful physics assistant. Respond in JSON format."

    # Template defining how the RAG context and question are provided to the LLM.
    user_template = """
    Question: {question}
    
    Context:
    {context}
    
    Metadata: {additional_info}
    
    Answer in JSON format containing 'answer' and 'explanation'.
    """

    # Run the RAG pipeline for a sample question.
    result = pipeline.run(
        question="How do wind turbines work?",
        system_prompt=sys_prompt,
        template=user_template,
        top_k=2,
    )

    # Pretty-print final structured answer.
    print("\n=== Result ===")
    print(json.dumps(result["answer"], indent=2, ensure_ascii=False))
    print(f"\nDocs Used: {result['retrieved_docs']}")
