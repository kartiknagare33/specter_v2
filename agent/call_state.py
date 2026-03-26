# agent/call_state.py

class CallState:
    def __init__(self):
        # The 14 clinical questions ARIA must address
        self.answers = {
            "weight": None,
            "side_effects": None,
            "new_allergies": None,
            "current_meds": None,
            "missed_doses": None,
            "effectiveness": None,
            "new_symptoms": None,
            "doctor_visits": None,
            "blood_pressure": None,
            "pain_level": None,
            "mood": None,
            "supply_remaining": None,
            "refill_timing": None,
            "other_concerns": None
        }
        
        # The 7 SPECTER Probes - 1.0 is nominal (green), 0.0 is failed (red)
        self.probes = {
            "A_acoustic_fidelity": 1.0,
            "B_intent_alignment": 1.0,
            "C_coverage_integrity": 1.0,
            "D_safety_compliance": 1.0,
            "E_identity_signal": 1.0,
            "F_capture_accuracy": 1.0,
            "G_behavioral_signal": 1.0
        }
        
        # Tracking for the amber pulse demo moment
        self.behavioral_flags_raised = 0
        self.behavioral_flags_resolved = 0
        
        # Audit log for the investigation report UI
        self.corrections_log = []
        
        # Post-call metadata
        self.outcome = None
        self.duration_seconds = 0
        
    def reset(self):
        """Resets the state for a new call."""
        self.__init__()

# Global singleton instance imported by FastAPI and the tool handlers
state = CallState()