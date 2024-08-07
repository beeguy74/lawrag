#!/usr/bin/env python3

from dotenv import load_dotenv
from init import init_models
from sys import argv
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.readers.file import PagedCSVReader
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
from llama_index.core.schema import Document
import re

from typing import List

def parse_metadata(documents: List[Document]):
    for doc in documents:
        # in document["text"] we have "article_id: LEGIARTI000006900781\nsource_id: LEGITEXT000006072050\narticle_num: L1111-1
        # we can extract the article_id until the first \n
        article_id_regex = r"article_id: ([^\n]+)"
        source_id_regex = r"source_id: ([^\n]+)"
        article_num_regex = r"article_num: ([^\n]+)"
        # now using regex we add extracted metadata to the document
        doc.metadata["article_id"] = re.search(article_id_regex, doc.text).group(1)
        doc.metadata["source_id"] = re.search(source_id_regex, doc.text).group(1)
        doc.metadata["article_num"] = re.search(article_num_regex, doc.text).group(1)
    return documents

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
reader = PagedCSVReader()
print("Loading data from ", path)
documents = reader.load_data(file = path)
documents = parse_metadata(documents)
for i, doc in enumerate(documents):
    print(f"Document {i} : {doc.metadata}")
    print("\n")

Settings.llm=llm
Settings.embed_model=embed_model

transformations = [
    SentenceSplitter(),
    TitleExtractor(nodes=5),
    # QuestionsAnsweredExtractor(questions=3),
    # SummaryExtractor(summaries=["prev", "self"]),
    KeywordExtractor(keywords=10),
    # EntityExtractor(prediction_threshold=0.5),
]

pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents)

for i, node in enumerate(nodes):
    print(f"Node [{i}]")
    for key, value in node.metadata.items():
        print(key, " : ", value)
    print("\n")
    if i == 5:
        break


print("Number of documents ", len(documents))
# if len(documents) < 50:
#     index = VectorStoreIndex.from_documents(
#         documents=documents,
#         show_progress=True,
#     )
# else:
#     start = 60
#     index = VectorStoreIndex.from_documents(
#         documents=documents[:start],
#         show_progress=True,
#     )
#     for i in range(start, len(documents), start - 1):
#         chunk = documents[i:i + start - 1]
#         out = index.refresh(chunk)
#         print(i, out)
#         sleep(4)
index = VectorStoreIndex(nodes=nodes)

persist_dir_name = Path("./storage_" + path.name + strftime("%Y-%m-%d_%H-%M-%S", localtime()))
index.storage_context.persist(persist_dir_name)
print("Index saved in ", persist_dir_name)
