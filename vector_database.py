import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents)

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0]

    def encode(self, text):
        return self.model.encode(text, normalize_embeddings=True)


def create_db_from_files(data_path: str, vector_db_path: str, embedding_model: SentenceTransformerEmbeddings):
    # Check if the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The directory {data_path} does not exist.")

    # Initialize the directory loader to scan the data directory
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls = TextLoader)

    # Load documents
    documents = loader.load()
    if not documents:
        raise ValueError("No documents found in the specified directory.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embedding = embedding_model)
    db.save_local(vector_db_path)
    
#search top k from db by query which embedd from ebmedding model
def search_top_k(db, embedding_model, query, k=3):
    # Generate query vector
    query_embedding = embedding_model.embed_query(query)

    # Search for top k candidates
    results = db.similarity_search_by_vector(query_embedding, k)

    return results

    

if __name__ == "__main__":
    embedding_model = SentenceTransformerEmbeddings("BAAI/bge-m3")

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
                "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = embedding_model.encode(sentences_1,
                                # normalize_embeddings=True
                            )
    embeddings_2 = embedding_model.encode(sentences_2,
                                # normalize_embeddings=True
                                )
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
    # [[0.6265, 0.3477], [0.3499, 0.678 ]]