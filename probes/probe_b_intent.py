# probes/probe_b_intent.py

def run_probe(patient_answer):
    """
    Catches opt-out or wrong number intent mid-conversation.
    Returns 1.0 if safe, 0.0 if opt-out detected.
    """
    answer_lower = patient_answer.lower()
    triggers = [
        "stop calling", "remove me", "shut up", 
        "don't call me", "wrong number", "who is this"
    ]
    
    for trigger in triggers:
        if trigger in answer_lower:
            return 0.0
            
    return 1.0
