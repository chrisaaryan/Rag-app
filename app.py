import streamlit as st
from pipeline.ingestion import ingestion_pipeline
from pipeline.retrieval import search_llamaindex
from pipeline.llama2 import load_or_download_t5_model

# Initialize state variables
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

def run_ingestion():
    documents_path = "F:/PROJECTS/RAG app/data"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_path = "F:/PROJECTS/RAG app/artifacts/embedding_model"
    
    st.write("Starting ingestion pipeline (stage 1)...")
    index = ingestion_pipeline(documents_path, model_name, model_path)
    st.session_state.index = index
    st.write("Stage 1 completed!!")

def run_model_loading():
    st.write("Starting model loading (stage 2)...")
    model, tokenizer = load_or_download_t5_model()
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.write("T5 model loaded and ready to use!")

# Streamlit UI
st.title("RAG (Retrieval Augmented Generation) App")

# Button to run ingestion pipeline
if st.button("Run Ingestion Pipeline (Stage 1)"):
    run_ingestion()

# Button to load the T5 model
if st.button("Load T5 Model (Stage 2)"):
    run_model_loading()

# Input field for user query
user_query = st.text_input("Enter your query:")

# Button to retrieve and generate response
if st.button("Search & Generate Response"):
    if st.session_state.index is not None and st.session_state.model is not None:
        st.write("Searching relevant documents...")
        response = search_llamaindex(user_query, st.session_state.index, st.session_state.model, st.session_state.tokenizer)
        st.write(f"Response from T5: {response}")
    else:
        st.write("Please run the ingestion pipeline and load the model first.")

