from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from graph.consts import CURRENT_DATE

Provider = Literal["ollama", "anthropic"]

import json
import os




def get_current_date():

    # return datetime.now().strftime("%Y-%m-%d")
    # Debug POC: OVERRIDE CURRENT DATE since the data is not current.
    return CURRENT_DATE


def load_config(config_file) -> Dict[str, Any]:
    """Load config from JSON."""
    with open(config_file, "r") as f:
        return json.load(f)


import os
from pathlib import Path
from typing import Literal, Optional

# Import necessary embedding classes
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings # Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from graph.consts import EMBED_MODEL, KB_PATH

# Define supported providers
EmbeddingProvider = Literal["ollama", "openai"]


def get_embedding_model(provider: str, model: str) -> Embeddings:
    """
    Factory to create an embedding model instance.
    Add more providers (Azure, HuggingFace) here as needed.
    """
    provider = provider.lower()

    if provider == "ollama":
        return OllamaEmbeddings(model=model)

    if provider == "openai":
        return OpenAIEmbeddings(model=model)

    # Example for HuggingFace (local)
    # if provider == "huggingface":
    #     return HuggingFaceEmbeddings(model_name=model)

    raise ValueError(f"Unsupported embedding provider: {provider}")


def get_retriever(
        persist_dir: str | Path,
        collection_name: str,
        embedding_model: Embeddings,
        k: int = 4
) -> VectorStoreRetriever:
    """
    Initializes the VectorStore and returns a retriever.

    Args:
        persist_dir: Path to the ChromaDB directory.
        collection_name: Name of the collection to load.
        embedding_model: The embedding model instance to use.
        k: Number of documents to retrieve.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})


from langchain_core.messages import BaseMessage, SystemMessage
import re


def serialize_context_to_json(obj: Any, label: str) -> str:
    """
    Serialize Pydantic models or dicts to formatted JSON strings for LLM context.

    Args:
        obj: The object to serialize (Pydantic model, dict, or None)
        label: Human-readable label for the context

    Returns:
        Formatted JSON string suitable for system message
    """
    if obj is None:
        return f"{label}: Not available"

    try:
        # Pydantic v2 uses model_dump(), v1 uses dict()
        if hasattr(obj, "model_dump"):
            obj_dict = obj.model_dump()
        elif hasattr(obj, "dict"):
            obj_dict = obj.dict()
        else:
            obj_dict = obj

        return f"{label}:\n{json.dumps(obj_dict, indent=2, default=str)}"
    except Exception as e:
        return f"{label}: Error serializing - {str(e)}"


def extract_json_from_markdown(text: str) -> dict:
    """
    Extract JSON from markdown code blocks.

    Many models (especially smaller or local ones) wrap JSON in ```json ... ```
    instead of using proper tool calling.

    Args:
        text: Text potentially containing JSON in markdown blocks

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If no valid JSON found
    """
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON (no markdown)
    try:
        # Look for JSON object pattern
        json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_obj_pattern, text, re.DOTALL)
        if matches:
            # Try each match (in case of multiple JSON objects)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    # Last resort: try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Could not extract valid JSON from text: {text[:200]}...")




def build_context_messages(
        trimmed_messages: list[BaseMessage],
        conversation_state: Any,
        intent_metadata: Any,
        user_profile: Any,
) -> list[BaseMessage]:
    """
    Build the complete message list with context injected as system messages.

    The agent needs to see:
    1. User profile (demographics, baselines, goals, etc.)
    2. Intent metadata (routing signals, confidence, etc.)
    3. Conversation state (topic, turn count, etc.)
    4. The actual conversation history

    Args:
        trimmed_messages: The conversation history (HumanMessage, AIMessage, etc.)
        conversation_state: ConversationState Pydantic model or None
        intent_metadata: IntentMetadata Pydantic model or None
        user_profile: UserProfile Pydantic model or None

    Returns:
        Complete message list with context system messages prepended
    """
    context_messages = []

    # Add user profile context
    if user_profile is not None:
        profile_context = serialize_context_to_json(user_profile, "User Profile")
        context_messages.append(SystemMessage(content=profile_context))

    # Add intent metadata context
    if intent_metadata is not None:
        intent_context = serialize_context_to_json(intent_metadata, "Intent Metadata")
        context_messages.append(SystemMessage(content=intent_context))

    # Add conversation state context
    if conversation_state is not None:
        conv_context = serialize_context_to_json(conversation_state, "Conversation State")
        context_messages.append(SystemMessage(content=conv_context))

    # Combine: context messages first, then conversation history
    return context_messages + list(trimmed_messages)

