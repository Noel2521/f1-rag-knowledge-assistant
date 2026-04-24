from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vector_store):
    # Load the LLM
    llm = OllamaLLM(model="llama3.2")

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    # Create prompt
    prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:""")

    # Build chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain ready!")
    return chain