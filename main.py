import os
import torch

from langchain_community.vectorstores import FAISS

from vector_database import SentenceTransformerEmbeddings, create_db_from_files, search_top_k
from llm import load_llm__model, answering_by_llm

def main():
    # Load SentenceTransformer model
    embedding_model = SentenceTransformerEmbeddings("BAAI/bge-m3")

    # Create vector database
    data_path = "data"
    vector_db_path = "vector_db"
    if not os.path.exists(vector_db_path):
        db = create_db_from_files(data_path, vector_db_path)
    else:
        db = FAISS.load_local(vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        
    # Load LLM model
    tokenizer, model = load_llm__model("Viet-Mistral/Vistral-7B-Chat")
    
    # Answering by LLM
    answering_by_llm(db)
