from transformers import AutoTokenizer, T5ForConditionalGeneration
# import logging
# from transformers import logging as transformers_logging

# transformers_logging.set_verbosity_debug()
# logging.basicConfig(level=logging.DEBUG)

def load_or_download_t5_model():

    try:
        # Download the model and tokenizer
        
        print("Downloading tokens...\n")
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", cache_dir="F:/PROJECTS/RAG app/artifacts/huggingface_cache")
        print("Downloading Model...\n")
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", cache_dir="F:/PROJECTS/RAG app/artifacts/huggingface_cache")
        
        print(f"Model and tokenizer saved")
        return model, tokenizer
    except Exception as e:
        print(f"An error occurred while downloading the model or tokenizer: {e}")