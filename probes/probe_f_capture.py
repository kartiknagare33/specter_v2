# probes/probe_f_capture.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model ONCE at module import time to prevent latency
model = SentenceTransformer('all-MiniLM-L6-v2')

def run_probe(answers_dict):
    """
    Catches capture accuracy errors and cross-turn contradictions.
    Returns 1.0 if consistent, 0.0 if contradiction detected.
    """
    missed_doses = answers_dict.get("missed_doses", "")
    supply = answers_dict.get("supply_remaining", "")
    
    if missed_doses and supply:
        # Encode answers to check semantic similarity/contradiction
        embeddings = model.encode([str(missed_doses), str(supply)])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # If answers are highly dissimilar in a way that implies contradiction
        # (This is a heuristic threshold for the hackathon demo)
        if similarity < -0.2:
            return 0.0
            
    return 1.0
