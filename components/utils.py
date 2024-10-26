from langchain_huggingface import HuggingFaceEmbeddings
import os
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

def get_hf_api():
    return os.getenv("HUGGINGFACE_API_KEY")

def huggingface_login():
    hf_api_key = get_hf_api()
    if not hf_api_key:
        raise ValueError("Hugging Face API key not found. Make sure it's set in your .env file.")
    
    try:
        login(token=hf_api_key)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"Login failed: {e}")

def load_or_save_embedding_model(model_name: str, model_path: str):

    huggingface_login()

    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Loading the embedding model...")
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        print(f"Model not found at {model_path}.")
        print(f"Downloading and saving the embedding model {model_name}...")
        #Download the model using Hugging Face's AutoModel and AutoTokenizer
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save the model locally for future use
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved locally at {model_path}")
        
        return HuggingFaceEmbeddings(model_name=model_name)