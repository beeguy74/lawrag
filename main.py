from dotenv import load_dotenv
from sys import argv
from pathlib import Path
from init import init_models
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import load_index_from_storage, StorageContext, Settings, VectorStoreIndex


def search(index: VectorStoreIndex, query, rerank):
    query_engine = index.as_query_engine(
        postprocessors=[rerank],
    )

    # retriever = index.as_retriever(
    #     postprocessors=[rerank],
    # )
    # print(">>> Documents:")
    # results = retriever.retrieve(query)
    # print(results)

    print("\n>>> Answer:")
    response = query_engine.query(query)
    print(response)
    for i, node in enumerate(response.source_nodes):
        print(f">>> Source node [{i}] :")
        for j in node.get_text().split("\n"):
            print(">>> ", j)


if __name__ == '__main__':
    load_dotenv()
    
    if len(argv) < 2:
        print("provide a path to vector storage")
        exit(1)
    path_storage = Path(argv[1])
    if not path_storage.exists():
        print("Storage does not exist")
        exit(1)

    llm, embed_model = init_models("cohere/command-r", "embed-multilingual-light-v3.0")
    Settings.llm=llm
    Settings.embed_model=embed_model

    cohere_rerank = CohereRerank()

    storage_context = StorageContext.from_defaults(
        persist_dir=path_storage
    )

    index = load_index_from_storage(storage_context=storage_context)

    while True:
        question = input("\033[92mPosez votre question par rapport au code du travail ou tapez 'q' pour quitter: \033[0m")
        if question == 'q':
            break
        # on cherche la reponse dans le code du travail
        search(index, question, cohere_rerank)
