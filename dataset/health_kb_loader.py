"""
graph/ingestion/health_kb_loader.py
Complete local RAG ingestion with batching, retries, and verification.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dataset.dataset_config import (EMBEDDING_PROVIDERS, INGESTION_CONFIG,
                                    STRATEGY_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()


def load_docs(data_dir: Path) -> List[Document]:
    documents = []
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist.")
        return []

    files = sorted(
        [f for f in data_dir.glob("*.txt") if not f.name.startswith(("test", "."))]
    )
    for file_path in files:
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata.update(
                    {
                        "source": file_path.name,
                        "topic": file_path.stem.replace("_", " "),
                    }
                )
                documents.append(doc)
            logger.info(f"Loaded {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
    return documents


def split_docs(docs: List[Document]) -> List[Document]:
    final_chunks = []
    default = INGESTION_CONFIG["default_chunking"]

    for doc in docs:
        fname = doc.metadata.get("source")
        strat = STRATEGY_MAP.get(
            fname,
            {
                "size": default["chunk_size"],
                "overlap": default["chunk_overlap"],
                "cat": "General",
            },
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=strat["size"],
            chunk_overlap=strat["overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {"chunk_id": f"{fname}_{i}", "category": strat["cat"]}
            )
            final_chunks.append(chunk)
    return final_chunks


def generate_health_knowledge_base():
    paths = INGESTION_CONFIG["paths"]
    data_path = Path(paths["data_dir"]).resolve()
    db_path = Path(paths["db_dir"]).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load and Split
    raw_docs = load_docs(data_path)
    if not raw_docs:
        logger.warning("No documents found to process. Exiting.")
        return

    chunked_docs = split_docs(raw_docs)
    logger.info(f"Total chunks created: {len(chunked_docs)}")

    # 2. Setup Embeddings
    model_cfg = INGESTION_CONFIG["models"]
    embedding_provider = model_cfg["embedding_provider"]
    if embedding_provider == "ollama":
        embedding_class = OllamaEmbeddings
    elif embedding_provider == "openai":
        embedding_class = OpenAIEmbeddings
    else:
        raise ValueError(
            f"Embedding provider must be one of {EMBEDDING_PROVIDERS} but it was {embedding_provider} instead"
        )
    embeddings = embedding_class(model=model_cfg["embedding_model"])

    # 3. Vector Store Initialization
    vs_cfg = INGESTION_CONFIG["vector_store"]

    # We initialize the store first to check for existing data
    vectorstore = Chroma(
        collection_name=vs_cfg["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(db_path),
        collection_metadata={"hnsw:space": vs_cfg["distance_function"]},
    )

    # Check if we already have data (optional: allows for incremental updates)
    current_count = vectorstore._collection.count()
    if current_count > 0:
        logger.info(
            f"Collection already exists with {current_count} documents. Skipping full re-index."
        )
        # Optional: return here or proceed to add only new docs

    # 4. Batch Ingestion
    batch_size = INGESTION_CONFIG["processing"]["batch_size"]
    total_chunks = len(chunked_docs)

    logger.info(
        f"Starting ingestion of {total_chunks} chunks in batches of {batch_size}..."
    )

    for i in range(0, total_chunks, batch_size):
        batch = chunked_docs[i : i + batch_size]
        retry_count = 0
        success = False

        while (
            not success and retry_count < INGESTION_CONFIG["processing"]["max_retries"]
        ):
            try:
                vectorstore.add_documents(batch)
                success = True
                logger.info(
                    f"Indexed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}"
                )
            except Exception as e:
                retry_count += 1
                wait_time = retry_count * 2
                logger.warning(
                    f"Batch failed (attempt {retry_count}): {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        if not success:
            logger.error(
                f"Critical failure: Batch starting at index {i} could not be indexed."
            )

    # 5. Final Verification
    final_count = vectorstore._collection.count()
    logger.info("=" * 30)
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"Final Document Count: {final_count}")
    logger.info(f"Database Persisted to: {db_path}")
    logger.info("=" * 30)


if __name__ == "__main__":
    generate_health_knowledge_base()
