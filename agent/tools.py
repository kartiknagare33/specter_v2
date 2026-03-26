# agent/tools.py
from agent.call_state import state
from probes import (
    probe_a_acoustic,
    probe_b_intent,
    probe_c_coverage,
    probe_d_safety,
    probe_e_identity,
    probe_f_capture,
    probe_g_behavioral
)

def log_answer(question_topic: str, patient_answer: str, confidence: str, flag: str) -> str:
    """
    Tool called natively by ElevenLabs after every confirmed patient response.
    Intercepts the answer and runs real-time clinical probes before logging.
    """
    # Track previous state to detect demo "recoveries"
    prev_score_a = state.probes["A_acoustic_fidelity"]
    prev_score_g = state.probes["G_behavioral_signal"]

    # 1. Update the shared memory
    state.answers[question_topic] = patient_answer

    # 2. Run the Real-Time Probes
    score_a = probe_a_acoustic.run_probe(question_topic, patient_answer)
    score_b = probe_b_intent.run_probe(patient_answer)
    score_d = probe_d_safety.run_probe(patient_answer)
    score_f = probe_f_capture.run_probe(state.answers)
    score_g = probe_g_behavioral.run_probe(flag)

    # 3. Update the Radar Chart State
    state.probes["A_acoustic_fidelity"] = score_a
    state.probes["B_intent_alignment"] = score_b
    state.probes["D_safety_compliance"] = score_d
    state.probes["F_capture_accuracy"] = score_f
    
    # Behavioral logic: Spoke collapses on flag, recovers on clean follow-up
    if flag == "behavioral_uncertainty":
        state.probes["G_behavioral_signal"] = 0.0
    elif flag == "none" and prev_score_g == 0.0:
        state.probes["G_behavioral_signal"] = 1.0

    # 4. Handle Demo Moments & Build the Audit Log
    
    # Moment 1: Acoustic Recovery
    if score_a == 0.0:
        state.corrections_log.append(f"[FLAG] Acoustic anomaly on {question_topic}: '{patient_answer}'")
    elif score_a == 1.0 and prev_score_a == 0.0:
        state.corrections_log.append(f"[RESOLVED] Acoustic anomaly corrected. Verified {question_topic}: '{patient_answer}'")

    # Moment 2: Behavioral Recovery
    if flag == "behavioral_uncertainty":
        state.behavioral_flags_raised += 1
        state.corrections_log.append(f"[FLAG] Behavioral uncertainty detected on {question_topic}")
    elif flag == "none" and prev_score_g == 0.0:
        state.behavioral_flags_resolved += 1
        state.corrections_log.append(f"[RESOLVED] Behavioral uncertainty probed. Confirmed {question_topic}: '{patient_answer}'")
        
    # Catching Contradictions
    if score_f == 0.0:
        state.corrections_log.append(f"[FLAG] Cross-turn contradiction detected involving {question_topic}")

    return f"Successfully logged {question_topic}. SPECTER probes updated."


def end_call(outcome: str, summary: str) -> str:
    """
    Tool called natively by ElevenLabs when the call ends.
    Runs the final checklist and identity verification probes.
    """
    state.outcome = outcome
    
    # Run Post-Call Probes
    state.probes["C_coverage_integrity"] = probe_c_coverage.run_probe(state.answers)
    state.probes["E_identity_signal"] = probe_e_identity.run_probe(outcome)
    
    return "Call ended successfully. Post-call probes complete. Ready for deliberation."