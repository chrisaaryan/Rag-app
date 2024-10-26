import numpy as np
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from components.utils import load_or_save_embedding_model
from llama_index.core import Settings

def ingestion_pipeline(documents_path: str, model_name: str, model_path: str):
    # Step 1: Data Ingestion
    print("Ingesting data...")
    documents = SimpleDirectoryReader(documents_path).load_data()
    print(f"Loaded {len(documents)} documents.")
    
    # Step 2: Load or save the embedding model
    print("Creating embedding model...")
    embed_model= load_or_save_embedding_model(model_name, model_path)

    Settings.embed_model = embed_model

    # Step 3: Indexing using LlamaIndex
    print("Creating in-memory vector index...")
    vector_index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # Return the vector index stored in memory
    return vector_index

if __name__ == "__main__":
    index = ingestion_pipeline("/data/attention_is_all_you_need", "huggingface_model", "path/to/model")
    print(f"Indexing completed !!")

