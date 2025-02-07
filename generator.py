
import requests
import os
import time

HF_API_KEY = os.getenv("HF_API_KEY")

def generate_answer(query, context):
    """Generate a response using Hugging Face API with retry logic."""
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    prompt = f"""
    You are an intelligent assistant. Answer the following question using the provided context.
    
    Context: {context}
    
    Question: {query}
    
    Answer:
    """
    
    while True:
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        data = response.json()

        if response.status_code == 200:
            return data[0]["generated_text"]
        elif "error" in data and "currently loading" in data["error"]:
            wait_time = data.get("estimated_time", 30)  # Default to 30 seconds if not provided
            print(f"⚠️ Model is still loading. Retrying in {int(wait_time)} seconds...")
            time.sleep(int(wait_time))
        else:
            return f"Error: {data}"

