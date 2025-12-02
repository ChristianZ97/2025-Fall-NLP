# full_test.py

import os
from pathlib import Path
import pandas as pd
import time
from tqdm.auto import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import RAG components defined in rag_pipeline.py
from rag_pipeline import (
    EmbeddingModel,
    VectorStore,
    KeywordRetriever,
    Reranker,
    LLMClient,
    RAGPipeline,
)

# Import prompt templates (system + user)

# Maximum number of parallel threads (concurrent questions being processed at once)
MAX_NUM_SEQS = 12

# Maximum character and their overlapping of each document chunk
MAX_CHARS = 512
OVERLAP = 64

# This controls how to retrieve the chunks
TOP_K = 20

# Data directories and file paths
DATA_DIR = Path("./data/WattBot2025")

# Fallback answer if the model fails to produce a valid answer
FALLBACK_ANSWER = "Unable to answer with confidence based on the provided documents."

# Load metadata mapping from ID to URL so that ref_id -> ref_url can be filled
META_PATH = DATA_DIR / "metadata.csv"
meta_df = pd.read_csv(META_PATH, encoding="latin1")
ID2URL = dict(zip(meta_df["id"], meta_df["url"]))


## hypers
# This controls how to parse the source documents
# DOC_DIR = DATA_DIR / "download" / "pdf_texts" # 0
DOC_DIR = DATA_DIR / "download" / "tex_texts"  # 1

# This controls what embedding model to use
EMBEDDING = "jinaai/jina-embeddings-v3"  # 0
# EMBEDDING = "google/embeddinggemma-300m" # 1

# This controls how to prompt the LLM
from prompt_v0 import SYS_PROMPT, USER_TEMPLATE  # 0

# from prompt_v1 import SYS_PROMPT, USER_TEMPLATE # 1
# from prompt_v2 import SYS_PROMPT, USER_TEMPLATE # 2

# This controls the output path
OUT_PATH = Path("./result_100/submission.csv")


def load_text_documents(doc_dir: Path):
    """
    Load all .txt documents from a directory.

    Parameters
    ----------
    doc_dir : Path
        Directory containing the .txt files produced by the preprocessing pipeline.

    Returns
    -------
    documents : List[str]
        List of document texts.
    doc_ids : List[str]
        Corresponding document IDs (filename stem, without extension).
    """
    documents = []
    doc_ids = []

    if not doc_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {doc_dir}")

    # Read all .txt files in sorted order (deterministic)
    for txt_path in sorted(doc_dir.glob("*.txt")):
        try:
            # Primary attempt: UTF-8
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            # Fallback encoding if UTF-8 fails
            text = txt_path.read_text(encoding="latin1", errors="ignore")

        # Skip empty or whitespace-only files
        if not text.strip():
            continue

        documents.append(text)
        # Use the file stem as document ID (e.g., "1234.5678v1")
        doc_ids.append(txt_path.stem)

    print(f"Loaded {len(documents)} documents from {doc_dir}")
    return documents, doc_ids


def chunk_document(
    doc_id: str,
    text: str,
    max_chars: int = 1200,
    overlap: int = 200,
):
    """
    Split a long document into overlapping character-based chunks.

    Rationale
    ---------
    - Long documents may be too large for the embedding model or RAG context.
    - Chunking allows finer-grained retrieval and better coverage.

    Parameters
    ----------
    doc_id : str
        Base document ID.
    text : str
        Full text of the document.
    max_chars : int, default 1200
        Maximum number of characters per chunk.
    overlap : int, default 200
        Number of overlapping characters between consecutive chunks to avoid
        cutting in the middle of important context.

    Returns
    -------
    List[Tuple[str, str]]
        List of (chunk_id, chunk_text) pairs, where chunk_id is of the form
        "<doc_id>#0000", "<doc_id>#0001", etc.
    """
    assert max_chars > 0, "max_chars should > 0"
    # Ensure overlap is smaller than max_chars
    overlap = min(overlap, max_chars - 1)

    chunks = []
    start = 0
    idx = 0
    n = len(text)

    # Slide a window across the document, creating overlapping segments
    while start < n:
        end = min(n, start + max_chars)
        chunk_text = text[start:end].strip()
        if chunk_text:
            # Construct a unique chunk ID from doc_id and chunk index
            chunk_id = f"{doc_id}#{idx:04d}"
            chunks.append((chunk_id, chunk_text))
            idx += 1

        if end == n:
            # Reached the document end
            break
        # Move window with overlap
        start = end - overlap

    return chunks


def build_rag_pipeline():
    """
    Build and return a fully initialized RAG pipeline.

    Steps
    -----
    1. Load base documents from DOC_DIR.
    2. Chunk them into overlapping segments.
    3. Embed all chunks and build a FAISS vector store.
    4. Initialize BM25 keyword retriever and cross-encoder reranker.
    5. Initialize the LLM client for generation.

    Returns
    -------
    RAGPipeline
        A ready-to-use RAG pipeline instance.
    """
    # Load raw documents from disk
    documents, doc_ids = load_text_documents(DOC_DIR)

    chunk_texts = []
    chunk_ids = []

    # Convert each document into one or more chunks
    for text, did in zip(documents, doc_ids):
        chunks = chunk_document(did, text, max_chars=MAX_CHARS, overlap=OVERLAP)
        for cid, ctext in chunks:
            chunk_ids.append(cid)
            chunk_texts.append(ctext)

    print(f"Loaded {len(documents)} docs, produced {len(chunk_texts)} chunks.")

    # Initialize the embedding model once; this is usually the heaviest step
    embedder = EmbeddingModel(model_name=EMBEDDING)

    # Compute embeddings for all chunks
    chunk_embeddings = embedder.embed(chunk_texts)

    # Build vector store with correct dimensionality
    vector_store = VectorStore(dim=chunk_embeddings.shape[1])
    vector_store.add(chunk_embeddings, chunk_ids, chunk_texts)

    # Build keyword retriever (BM25) and reranker
    bm25 = KeywordRetriever(chunk_texts, chunk_ids)
    reranker = Reranker(model_name="BAAI/bge-reranker-large")

    # Initialize LLM client for an OpenAI-compatible endpoint.
    # Adjust model_name / base_url / api_key according to your setup.
    llm = LLMClient(
        model_name="openai/gpt-oss-20b",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Compose all components into a RAG pipeline
    pipeline = RAGPipeline(embedder, vector_store, llm, bm25, reranker)
    return pipeline


def run_full_test():
    """
    Main entry point for running full evaluation on the test set.

    Procedure
    ---------
    1. Load train_QA.csv for expected column structure of the submission.
    2. Load test_Q.csv for actual questions to answer.
    3. Build a single shared RAG pipeline (used by all threads).
    4. For each test question, run the pipeline and parse the response into
       a row consistent with the train_QA columns.
    5. Save all results to submission.csv.
    6. Validate the submission for basic sanity checks.
    """
    # Load training QA just to get expected submission columns
    train_qa = pd.read_csv(DATA_DIR / "train_QA.csv")
    test_q = pd.read_csv(DATA_DIR / "test_Q.csv")
    expected_columns = list(train_qa.columns)

    # Build the RAG pipeline (shared across all inference threads)
    pipeline = build_rag_pipeline()
    records = []

    def process_one(row, idx):
        """
        Process a single test question row.

        This function:
        - Calls the RAG pipeline for the question.
        - Parses the structured JSON response into the expected submission fields.
        - Applies default / fallback values when fields are missing or invalid.

        Parameters
        ----------
        row : pd.Series
            A row from test_Q.csv containing at least 'id' and 'question'.
        idx : int
            Index of the question (for logging / sampling).

        Returns
        -------
        Dict[str, Any]
            One submission row as a dict matching 'expected_columns'.
        """
        question_id = row["id"]
        question = row["question"]
        meta = {"id": question_id}

        # Call RAG pipeline to get an answer for the question
        result = pipeline.run(
            question=question,
            system_prompt=SYS_PROMPT,
            template=USER_TEMPLATE,
            additional_info=meta,
            top_k=TOP_K,
        )

        # 'answer' field of result is expected to be a dict (parsed JSON) or error
        raw_answer = result.get("answer", {})
        if isinstance(raw_answer, dict):
            parsed = raw_answer
        else:
            parsed = {}

        out: Dict[str, Any] = {}

        # Fill each column according to expected schema
        for col in expected_columns:
            if col == "id":
                out[col] = question_id

            elif col == "question":
                out[col] = question

            elif col == "answer":
                # Main natural language answer text.
                answer_text = parsed.get("answer")
                if not isinstance(answer_text, str) or not answer_text.strip():
                    answer_text = FALLBACK_ANSWER
                out[col] = answer_text

            elif col == "explanation":
                # Additional explanation text
                expl_text = parsed.get("explanation")
                if not isinstance(expl_text, str) or not expl_text.strip():
                    expl_text = "is_blank"
                out[col] = expl_text

            elif col == "answer_value":
                # Numeric or structured answer value (may be scalar, list, dict).
                val = parsed.get("answer_value", "is_blank")
                if val is None or val == "":
                    out[col] = "is_blank"
                else:
                    # Convert booleans to 1/0 as integers.
                    if isinstance(val, bool):
                        val = 1 if val else 0
                    # Convert lists/dicts to JSON strings to preserve structure.
                    if isinstance(val, (list, dict)):
                        out[col] = json.dumps(val, ensure_ascii=False)
                    else:
                        # Everything else to string.
                        out[col] = str(val)

            elif col == "answer_unit":
                # Unit of the answer_value (if numeric), else "is_blank".
                unit = parsed.get("answer_unit", "is_blank")
                if not isinstance(unit, str) or not unit.strip():
                    unit = "is_blank"
                out[col] = unit

            elif col == "ref_id":
                # Document IDs supporting the answer.
                # The model may return a list or a string with delimited IDs.
                ref_ids = parsed.get("ref_id", [])
                if isinstance(ref_ids, str):
                    # Try splitting by ';' or ',' depending on content
                    sep = ";" if ";" in ref_ids else ","
                    ref_ids = [x.strip() for x in ref_ids.split(sep)]
                if not isinstance(ref_ids, list):
                    ref_ids = []
                # Filter out empty IDs and ensure they exist in ID2URL
                ref_ids = [r for r in ref_ids if r and r in ID2URL]
                if not ref_ids:
                    out[col] = "is_blank"
                else:
                    # Join multiple IDs with ';'
                    out[col] = ";".join(ref_ids)

            elif col == "ref_url":
                # URLs corresponding to ref_id values.
                ref_ids_cell = out.get("ref_id", "is_blank")
                if ref_ids_cell == "is_blank":
                    out[col] = "is_blank"
                else:
                    ids = [x.strip() for x in ref_ids_cell.split(";") if x.strip()]
                    urls = [ID2URL.get(i, "") for i in ids if ID2URL.get(i, "")]
                    out[col] = ";".join(urls) if urls else "is_blank"

            elif col == "supporting_materials":
                # Free-form description or list of extra references/materials.
                sup = parsed.get("supporting_materials", "is_blank")
                if not isinstance(sup, str) or not sup.strip():
                    sup = "is_blank"
                out[col] = sup

            else:
                # For any unexpected columns, fill with "is_blank" by default.
                out[col] = "is_blank"

        # Print a few sample rows for inspection.
        if idx < 3:
            print("\n")
            print(f"\n=== Sample #{idx} / id={question_id} ===")
            for c in expected_columns:
                print(f"{c}: {out[c]}")
            print("\n")

        return out

    # Use multithreading to process multiple questions in parallel.
    # This is mainly beneficial when LLM calls or retrieval are network/IO-bound.
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_NUM_SEQS) as executor:
        for idx, (_, row) in enumerate(test_q.iterrows(), start=1):
            # Submit each question to the executor
            futures.append(executor.submit(process_one, row, idx))

        # Progress bar over all completed futures
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running full_test inference (parallel)",
        ):
            out = fut.result()
            records.append(out)

    # Create a DataFrame from all result rows
    submission = pd.DataFrame(records)
    # Enforce column order
    submission = submission[expected_columns]

    # Ensure no missing values remain; fill them with defaults.
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

    # Save submission CSV to the project root.
    submission.to_csv(OUT_PATH, index=False)
    print("Saved full-test submission to", OUT_PATH)

    # Run simple validation checks.
    validate_submission_csv(OUT_PATH, expected_columns)


def validate_submission_csv(path: Path, expected_columns):
    """
    Validate the generated submission CSV.

    Checks
    ------
    - Column names exactly match the expected schema.
    - No null/NaN values are present.
    - Prints row count and a preview of first few rows.

    Parameters
    ----------
    path : Path
        Path to the submission CSV file.
    expected_columns : List[str]
        Column names expected to be present in the submission.
    """
    df = pd.read_csv(path)

    # Ensure that column order and names match exactly.
    assert list(df.columns) == list(expected_columns), "CSV columns mismatch!"

    # Ensure there are no null or NaN values.
    assert not df.isnull().any().any(), "Submission contains null/NaN values!"

    print(f"Submission rows: {len(df)}")
    print(df.head(3))


if __name__ == "__main__":
    # When called as a script, run the full test pipeline.
    run_full_test()
