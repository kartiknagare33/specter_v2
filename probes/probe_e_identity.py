# probes/probe_e_identity.py

def run_probe(call_outcome):
    """
    Catches wrong number misclassifications.
    Returns 1.0 if nominal, 0.0 if identity mismatch flagged.
    """
    if call_outcome == "wrong_number":
        return 0.0
        
    return 1.0
