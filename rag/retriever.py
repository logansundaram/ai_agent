
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain.tools import Tool
from rag.doc_loader import load_documents
import os

vectordb = None
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def build_initial_rag(llm):
    global vectordb

    docs = load_documents()
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, embedder)

    retriever = vectordb.as_retriever()

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question concisely.

    Context:
    {context}

    Question:
    {question}
    """)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return Tool(
        name="LocalDocsQA",
        func=rag_chain.invoke,
        description="Answer questions using your local documents in /docs"
    )

def add_new_file_to_rag(file_path: str) -> str:
    global vectordb

    if vectordb is None:
        return "Vector store not initialized yet."

    if file_path.endswith(".txt") or file_path.endswith(".md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    else:
        return "Unsupported file type for RAG ingestion."

    docs = loader.load()
    if not docs:
        return f" No content loaded from {file_path}."

    chunks = splitter.split_documents(docs)
    if not chunks:
        return f" File loaded but could not be split into usable chunks: {file_path}"

    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.embed_documents(texts)
    if not embeddings:
        return f" No embeddings returned for {file_path}. Skipping update."

    vectordb.add_documents(chunks)
    return f" {file_path} added to RAG database."
