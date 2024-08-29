#!/usr/bin/env python3

# this script loads data from a CSV file and creates an vector index using embedding model

from dotenv import load_dotenv
from modules.init import init_models
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
from modules.MyPandasCSVReader import MyPandasCSVReader
from modules.MyXMLReader import MyXMLReader


load_dotenv()
if (len(argv) < 2):
    print("Provide a path to data to load")
    exit(1)

path = Path(argv[1])
if (not path.exists()):
    print("File does not exist")
    exit(1)

reader = None
# check the extension of file - is it csv, xml or smth else
if path.suffix == ".csv":
    # PandasCSVReader uses pandas.read_csv() to load data from a CSV file
    reader = MyPandasCSVReader(
        concat_rows=False,
    )
elif path.suffix == ".xml":
    # MyXMLReader uses pandas.read_csv() to load data from an XML file
    reader = MyXMLReader(
        concat_rows=False,
    )
else:
    print("Unsupported file format, provide a CSV or XML file")
    exit(1)

# docs = reader.load_data(file = path)


llm, embed_model = init_models("cohere/command-r", "OrdalieTech/Solon-embeddings-large-0.1", local=True)

if llm is None or embed_model is None:
    print("Error while initializing the models")
    exit(1)

print("Loading data from ", path)
documents = reader.load_data(file = path)
print("Number of documents loaded: ", len(documents))

Settings.llm=llm
Settings.embed_model=embed_model
Settings.chunk_size=2048

transformations = [
    SentenceSplitter(chunk_size=Settings.chunk_size),
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
index = VectorStoreIndex(nodes=nodes, show_progress=True)

persist_dir_name = Path("./storage_" + path.name + strftime("%Y-%m-%d_%H-%M-%S", localtime()))
index.storage_context.persist(persist_dir_name)
print("Index saved in ", persist_dir_name)
