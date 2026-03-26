# probes/probe_a_acoustic.py
from word2number import w2n

def run_probe(question_topic, patient_answer):
    """
    Checks for STT mishearing on numeric values.
    Returns 1.0 if plausible, 0.0 if physiologically implausible.
    """
    if question_topic not in ["weight", "pain_level"]:
        return 1.0
        
    try:
        # Attempt to parse spoken words to a number, or fallback to float
        try:
            val = float(patient_answer)
        except ValueError:
            val = float(w2n.word_to_num(patient_answer))
            
        if question_topic == "weight":
            # Implausible adult weight (under 50 lbs or over 500 lbs)
            if val < 50 or val > 500:
                return 0.0
        elif question_topic == "pain_level":
            # Implausible pain scale
            if val < 0 or val > 10:
                return 0.0
    except Exception:
        # If we can't parse it cleanly, we don't flag it blindly
        pass
        
    return 1.0