# probes/probe_d_safety.py

def run_probe(patient_answer):
    """
    Catches potential medical advice solicitations.
    Returns 1.0 if safe, 0.0 if medical advice risk.
    """
    answer_lower = patient_answer.lower()
    advice_triggers = [
        "should i take", "what should i do", 
        "is it safe to", "can i mix", "what is the dosage"
    ]
    
    for trigger in advice_triggers:
        if trigger in answer_lower:
            return 0.0
            
    return 1.0

