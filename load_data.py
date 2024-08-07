#!/usr/bin/env python3

from dotenv import load_dotenv
from init import init_models
from llama_index.core.llms import ChatMessage

load_dotenv()
llm, embed_model = init_models("cohere/command-r", "embed-multilingual-light-v3.0")

