# rerank

from llama_index.llms.cohere import Cohere
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from os import getenv

from dotenv import load_dotenv

load_dotenv()

APIKEY = getenv("COHERE_API_KEY")

# Create the embedding model
embed_model = CohereEmbedding(
    cohere_api_key=APIKEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

# Create the service context with the cohere model for generation and embedding model
service_context = ServiceContext.from_defaults(
    llm=Cohere(api_key=APIKEY, model="command-r"),
    embed_model=embed_model,
)

# Load the data, for this example data needs to be in a test file 
data = SimpleDirectoryReader(input_files=["example_data_file.txt"]).load_data()
index = VectorStoreIndex.from_documents(data, service_context=service_context)

# Create a cohere reranker 
cohere_rerank = CohereRerank(api_key=APIKEY)

# Create the query engine
query_engine = index.as_query_engine(node_postprocessors=[cohere_rerank])

# Generate the response
response = query_engine.query("who is guilty goddes?")

print(response)