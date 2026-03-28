# probes/probe_c_coverage.py

def run_probe(call_state_answers):
    """
    Checks if all 14 questions were successfully addressed.
    Returns 1.0 if complete, 0.0 if questions were skipped.
    """
    answered_count = sum(1 for v in call_state_answers.values() if v is not None)
    
    if answered_count < 14:
        return 0.0
        
    return 1.0
