import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(directory="./docs"):
    docs = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(full_path)
        else:
            continue
        docs.extend(loader.load())
    return docs
