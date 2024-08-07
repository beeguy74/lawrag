
from typing import Tuple
from llama_index.llms.openrouter import OpenRouter
from os import getenv
from llama_index.embeddings.cohere import CohereEmbedding

def init_models(llm_name, embeddings_model_name) -> Tuple[OpenRouter, CohereEmbedding]:
    """
    Initialize the llm and the embeddings
    """
    llm = None
    embed_model = None
    try:
        llm = OpenRouter(llm_name)
    except Exception as e:
        print(f"Error while loading the llm: {e}")

    try:
        embed_model = CohereEmbedding(
            api_key=getenv("COHERE_API_KEY"),
            model_name=embeddings_model_name,
            input_type="search_query",
        )
    except Exception as e:
        print(f"Error while loading the embeddings: {e}")


    return llm, embed_model

