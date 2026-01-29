import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from graph.consts import KB_PATH, EMBED_MODEL

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

retriever = Chroma(
    collection_name="health_knowledge",
    persist_directory=str(KB_PATH),
    embedding_function=embeddings,
).as_retriever()
