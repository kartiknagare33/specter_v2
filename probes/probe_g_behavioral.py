# probes/probe_g_behavioral.py

def run_probe(elevenlabs_flag):
    """
    Catches hesitation, filler words, and uncertainty based on the agent's flag.
    Returns 1.0 if confident, 0.0 if behavioral uncertainty detected.
    """
    if elevenlabs_flag == "behavioral_uncertainty":
        return 0.0
        
    return 1.0
