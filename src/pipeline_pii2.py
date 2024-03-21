from transformers import pipeline
from huggingface_hub import login
import pickle
import math
login("")


def read_pickle_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return ''.join(data)
    

def pii_pipeline(input_data, chunk_size, overlap_size):
    token_classifier = pipeline("token-classification", model="bigcode/starpii", aggregation_strategy="first")
    
    tokens = []
    start_idx = 0
    while start_idx < len(input_data):
        end_idx = min(start_idx + chunk_size, len(input_data))
        chunk = input_data[start_idx:end_idx]
        chunk_tokens = token_classifier(chunk)
        tokens.extend(chunk_tokens)
        start_idx += chunk_size - overlap_size 
        
    return tokens

if __name__ == "__main__":
    file = read_pickle_file("D:\Dataset Paper\src\RedPajamaComments.pkl")
    pii_data = pii_pipeline(file[:10000], 3485, 1000)
    unique_vals = set()
    unique_pii = []
    for item in pii_data:
        if item["word"] not in unique_vals:
            unique_vals.add(item["word"])
            unique_pii.append(item)

    print(len(unique_pii))
    print(unique_pii)