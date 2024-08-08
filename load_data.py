#!/usr/bin/env python3

from dotenv import load_dotenv
from init import init_models
from sys import argv
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from time import sleep, localtime, strftime
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from MyPandasCSVReader import MyPandasCSVReader


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

# PandasCSVReader uses pandas.read_csv() to load data from a CSV file
reader = MyPandasCSVReader(
    concat_rows=False,
)
print("Loading data from ", path)
documents = reader.load_data(file = path)
print("Number of documents loaded: ", len(documents))

Settings.llm=llm
Settings.embed_model=embed_model

transformations = [
    SentenceSplitter(),
    # TitleExtractor(nodes=5),
    # QuestionsAnsweredExtractor(questions=3),
    # SummaryExtractor(summaries=["prev", "self"]),
    # KeywordExtractor(keywords=10),
    # EntityExtractor(prediction_threshold=0.5),
]

pipeline = IngestionPipeline(
    transformations=transformations
    
)

nodes = pipeline.run(documents=documents)

for i, node in enumerate(nodes):
    print(f"Node [{i}]")
    for key, value in node.metadata.items():
        print(key, " : ", value)
    print("\n")
    if i == 5:
        break


print("Number of documents ", len(documents))
index = VectorStoreIndex(nodes=nodes)

persist_dir_name = Path("./storage_" + path.name + strftime("%Y-%m-%d_%H-%M-%S", localtime()))
index.storage_context.persist(persist_dir_name)
print("Index saved in ", persist_dir_name)
