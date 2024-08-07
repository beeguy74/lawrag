#!/usr/bin/env python3

from dotenv import load_dotenv
from init import init_models
from sys import argv
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.readers.file import PandasCSVReader


load_dotenv()
if (len(argv) < 2):
    print("Provide a path to data to load")
    exit(1)

path = Path(argv[1])
if (not path.exists()):
    print("File does not exist")
    exit(1)

llm, embed_model = init_models("cohere/command-r", "embed-multilingual-light-v3.0")

if llm is None or embed_model is None:
    print("Error while initializing the models")
    exit(1)

# CSVReader does not work properly, so i use PandasCSVReader
reader = PandasCSVReader(concat_rows=False)
print("Loading data from ", path)
documents = reader.load_data(file = path)

Settings.llm=llm
Settings.embed_model=embed_model

index = VectorStoreIndex.from_documents(
    documents=documents,
    show_progress=True,
)

persist_dir_name = Path("./storage_" + path.name)
if persist_dir_name.exists():
    persist_dir_name = Path("./storage_" + path.name + "_1")
index.storage_context.persist(persist_dir_name)
print("Index saved in ", persist_dir_name)
