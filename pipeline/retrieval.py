#from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document
# import os
# import faiss
# from faiss import write_index, read_index
# import pickle
# import numpy as np

# def search_faiss_index(query, vector_store_path, embedding_model_path):
#     #Load the embedding model (use the one saved during ingestion)
#     print("Loading embedding model...")
#     embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
    
#     # Load the FAISS index
#     # print("Loading FAISS index...")
#     # # faiss_store = FAISS.load_local(faiss_index_path, embedding_model,  allow_dangerous_deserialization=True)
#     # index = read_index("F:/PROJECTS/RAG app/artifacts/database/large.index")
#     # print("FAISS index loaded successfully.")
    
#     print("Loading vector store...")
#     with open(vector_store_path, 'rb') as f:
#         vector_store = pickle.load(f)
#     print("Vector store loaded successfully.")

#     # Convert the user query to embeddings
#     print(f"Converting query: '{query}' to embeddings...")
#     query_embedding = embedding_model.embed_documents([query])

#     # Ensure the embedding is a 2D array
#     query_embedding = np.array(query_embedding).astype('float32')
#     print("Dimensio= ", query_embedding.ndim)
#     similar_docs = vector_store.similarity_search(query_embedding)
#     # Return the most similar documents
#     return similar_docs

# if __name__ == "__main__":
#     user_query = input("Enter your query: ")
#     faiss_index_path = "F:/PROJECTS/RAG app/artifacts/faiss_index.bin"
#     embedding_model_path = "F:/PROJECTS/RAG app/artifacts/embedding_model"
    
#     # Search for documents in the FAISS index
#     retrieved_docs = search_faiss_index(user_query, faiss_index_path, embedding_model_path)
    
#     # Display the retrieved documents
#     print("Top documents retrieved from the vector database:")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"Document {i + 1}: {doc.page_content}")
def search_llamaindex(query, vector_index, model, tokenizer):
    print(f"Querying for: {query}")
    retriever = vector_index.as_retriever()
    results = retriever.retrieve(query)
    document_texts = [doc.text for doc in results if hasattr(doc, 'text')]
    combined_text = "\n".join(document_texts)

    t5_query = f"Context: {combined_text}\nQuestion: {query}"
    input_ids = tokenizer(t5_query, return_tensors='pt').input_ids
    # llama_query = query_wrapper_prompt.format(query_str=combined_text)
    # llama_response = llm.generate(llama_query)
    outputs = model.generate(input_ids, max_length= 256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response