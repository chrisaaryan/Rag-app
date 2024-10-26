from pipeline.ingestion import ingestion_pipeline
from pipeline.retrieval import search_llamaindex
from pipeline.llama2 import load_or_download_t5_model
def main():
    try:

        # Stage 1: Ingestion and indexing
        documents_path = "F:/PROJECTS/RAG app/data"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_path = "F:/PROJECTS/RAG app/artifacts/embedding_model"
        
        print("Starting ingestion pipeline(stage 1)...")
        index = ingestion_pipeline(documents_path, model_name, model_path)
        print(f"Stage 1 completed !!")
        #t5_model_name= "google-t5/t5-small"

        print("Starting model loading(stage 2)...")
        model, tokenizer= load_or_download_t5_model()
        print("Stage 2 completed !!")

        print("T5 model loaded and ready to use.")

        #Stage 2: Retrieval Stage
        user_query = input("Enter your query: ")
        print("Searching relevant documents...")
        response = search_llamaindex(user_query, index, model, tokenizer)

        print("Response from t5:", response)

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()