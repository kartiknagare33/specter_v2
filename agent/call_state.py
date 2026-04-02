# agent/call_state.py
# SPECTER v5 — Living Memory added


QUESTION_KEYS = [
    "overall_feeling", "current_weight", "weight_lost", "missed_doses",
    "side_effects", "satisfaction", "new_medications", "new_allergies",
    "surgeries", "energy_levels", "diet_adherence", "treatment_concerns",
    "doctor_questions", "address_change",
    # Hidden validation probes
    "target_gap", "supply_days", "adherence_difficulty",
    "change_request", "pharmacy_recent",
]

PRIMARY_KEYS = QUESTION_KEYS[:14]


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

        # The 8 SPECTER Probes — 1.0 is nominal (green), 0.0 is failed (red)
        self.probes = {
            "A_acoustic_fidelity":  1.0,
            "B_intent_alignment":   1.0,
            "C_coverage_integrity": 1.0,
            "D_safety_compliance":  1.0,
            "E_identity_signal":    1.0,
            "F_capture_accuracy":   1.0,
            "G_behavioral_signal":  1.0,
            "H_mi_fidelity":        1.0,
        }

        # Behavioral tracking
        self.behavioral_flags_raised   = 0
        self.behavioral_flags_resolved = 0
        self.corrections_log           = []

        # Post-call metadata
        self.outcome          = None
        self.duration_seconds = 0

        # ── LIVING MEMORY (v5) ──────────────────────────────────────────
        # Single structured brain shared across all Workflow agents.
        # Every agent reads this before speaking.
        # Every log_answer writes to this immediately.
        self.living_memory = {
            "questions_remaining":      list(PRIMARY_KEYS),   # 14 primary keys
            "questions_answered":       [],
            "emotional_state":          "neutral",            # neutral / concerned / distressed / engaged
            "detected_flags":           [],                   # ["pricing","nausea","safety","reschedule","missed_doses"]
            "patient_risk_level":       "LOW",                # LOW / MEDIUM / HIGH
            "recommended_next_action":  "continue_interview", # continue_interview / empathy_probe /
                                                              # handoff_billing / handoff_pharmacist /
                                                              # handoff_scheduling / close_call
            "handoff_required":         False,
            "handoff_target":           None,                 # "billing" / "pharmacist" / "scheduling"
            "ghost_alerts":             [],                   # live Ghost Analyst observations
            "pre_call_brief":           "",                   # seeded before call starts from patient DB
            "coverage_pct":             0.0,                  # 0.0 – 1.0
            "call_number":              1,                    # which call this is for this patient
        }

    def reset(self):
        """Resets the state for a new call."""
        self.__init__()

    def update_living_memory_routing(self):
        """
        Rule-based routing logic.
        Reads detected_flags and emotional_state to set
        recommended_next_action, handoff_required, handoff_target.
        No LLM — instant, zero latency.
        """
        flags  = self.living_memory["detected_flags"]
        es     = self.living_memory["emotional_state"]
        rem    = self.living_memory["questions_remaining"]

        if "safety" in flags:
            self.living_memory["recommended_next_action"] = "handoff_pharmacist"
            self.living_memory["handoff_required"]        = True
            self.living_memory["handoff_target"]          = "pharmacist"
        elif "pricing" in flags:
            self.living_memory["recommended_next_action"] = "handoff_billing"
            self.living_memory["handoff_required"]        = True
            self.living_memory["handoff_target"]          = "billing"
        elif "reschedule" in flags:
            self.living_memory["recommended_next_action"] = "handoff_scheduling"
            self.living_memory["handoff_required"]        = True
            self.living_memory["handoff_target"]          = "scheduling"
        elif es == "distressed":
            self.living_memory["recommended_next_action"] = "empathy_probe"
            self.living_memory["handoff_required"]        = False
            self.living_memory["handoff_target"]          = None
        elif len(rem) == 0:
            self.living_memory["recommended_next_action"] = "close_call"
            self.living_memory["handoff_required"]        = False
            self.living_memory["handoff_target"]          = None
        else:
            self.living_memory["recommended_next_action"] = "continue_interview"
            self.living_memory["handoff_required"]        = False
            self.living_memory["handoff_target"]          = None


# Global singleton imported by FastAPI and tool handlers
state = CallState()