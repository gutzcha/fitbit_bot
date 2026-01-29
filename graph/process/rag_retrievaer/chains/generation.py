from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from graph.helpers import make_llm, load_config
from graph.consts import CHAT_CONFIG_PATH

config = load_config(CHAT_CONFIG_PATH)
provider = config.get("provider", "ollama")
llm = make_llm(provider=provider, model_type="slow")

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
