from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def build_rag_chain(vector_store):
    #Load the LLM
    llm = Ollama(model="llama3.2")

    #Build the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k":5,
                "fetch_k":15
            }
        ),
        chain_type="stuff"
    )

    print("RAG chain ready!")
    return qa_chain