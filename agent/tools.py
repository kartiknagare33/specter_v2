# agent/tools.py
# SPECTER v5 — Pre-Call Intelligence, Living Memory, Post-Call Closed Loop
# File 2 of 6: Agent Tools & Clinical Logic Engine

import os
import time
import json
import asyncio
import threading
from groq import Groq
from agent.call_state import (
    global_state, 
    QUESTIONS, 
    TOPIC_TO_INDEX, 
    EXTRACTION_HINTS, 
    SIGNAL_WEIGHTS
)

# ─────────────────────────────────────────────────────────────
# STATE MUTATION HELPERS
# ─────────────────────────────────────────────────────────────

def accumulate_signal(flag: str):
    """Adds weighted risk scores to the patient's global signal tracking."""
    weight = SIGNAL_WEIGHTS.get(flag, 0)
    global_state.signal_score += weight
    
    if global_state.signal_score >= 7:
        global_state.interview_depth = "deep_probe"
    elif global_state.signal_score >= 3:
        global_state.interview_depth = "soft_probe"
    else:
        global_state.interview_depth = "standard"
        
    print(f"[SIGNAL] score={global_state.signal_score} flag={flag} depth={global_state.interview_depth}")

def _extract_number(text: str):
    """Helper to pull numerical values out of spoken text."""
    digits = ''.join(c for c in text if c.isdigit() or c == '.')
    try:
        return float(digits) if digits else None
    except ValueError:
        return None

def extract_multi_answers(patient_answer: str, active_topic: str):
    """
    Scans a long patient response for answers to future questions 
    so the AI doesn't ask them redundantly.
    """
    answer_lower = patient_answer.lower()
    word_count   = len(answer_lower.split())
    
    # Only try to extract from substantial, rambling answers
    if word_count < 10:
        return []

    extracted = []
    for topic, keywords in EXTRACTION_HINTS.items():
        if topic == active_topic:
            continue
            
        idx = TOPIC_TO_INDEX.get(topic)
        if idx is None:
            continue
            
        # Skip if we already have an answer for this slot
        if global_state.responses[idx]["answer"]:
            continue
            
        matched_keyword = None
        for kw in keywords:
            if kw in answer_lower:
                matched_keyword = kw
                break
                
        if matched_keyword:
            words = answer_lower.split()
            kw_words = matched_keyword.split()
            for i, w in enumerate(words):
                if w == kw_words[0]:
                    # Grab a context snippet around the matched keyword
                    start   = max(0, i - 2)
                    end     = min(len(words), i + 8)
                    snippet = " ".join(words[start:end])
                    
                    global_state.responses[idx]["answer"]         = snippet
                    global_state.responses[idx]["flag"]           = "none"
                    global_state.responses[idx]["auto_extracted"] = True
                    extracted.append({"topic": topic, "keyword": matched_keyword, "snippet": snippet})
                    print(f"[MULTI-EXTRACT] topic={topic} kw='{matched_keyword}' snippet='{snippet}'")
                    
                    # Update Living Memory (v5)
                    if topic in global_state.living_memory["questions_remaining"]:
                        global_state.living_memory["questions_remaining"].remove(topic)
                    if topic not in global_state.living_memory["questions_answered"]:
                        global_state.living_memory["questions_answered"].append(topic)
                    break

    if extracted:
        global_state.extracted_answers.extend(extracted)
        global_state.transcript.append({
            "role":      "system",
            "message":   f"[MULTI-EXTRACT] {len(extracted)} additional answer(s) captured: " + ", ".join(e["topic"] for e in extracted),
            "timestamp": time.time()
        })
    return extracted

def score_probes_on_answer(patient_answer: str, flag: str, question_topic: str = ""):
    """Updates the 8 spectral probes based on the content of the answer."""
    answer_lower  = patient_answer.lower()
    topic_lower   = question_topic.lower()
    acoustic_issue = None

    # Probe B: Intent Alignment (Opt-outs)
    OPT_OUT = ["stop calling", "remove me", "don't call", "wrong number", "who is this", "not interested"]
    if any(t in answer_lower for t in OPT_OUT):
        global_state.probes["B_intent_alignment"] = 0.0

    # Probe D: Safety Compliance
    SAFETY = ["should i take", "what should i do", "is it safe", "can i mix", "what is the dosage", "can i stop"]
    if any(t in answer_lower for t in SAFETY):
        global_state.probes["D_safety_compliance"] = 0.0
        accumulate_signal("safety_concern")

    # Probe G: Behavioral Signal
    prev_g = global_state.probes["G_behavioral_signal"]
    if flag == "behavioral_uncertainty":
        global_state.probes["G_behavioral_signal"] = 0.0
        global_state._prev_g_score = 0.0
    elif flag == "none" and prev_g == 0.0:
        global_state.probes["G_behavioral_signal"] = 1.0
        global_state._prev_g_score = 1.0

    # Probe C & F: Coverage and Capture Math
    answered = sum(1 for r in global_state.responses if r["answer"])
    global_state.probes["F_capture_accuracy"]   = round(answered / len(QUESTIONS), 2)
    global_state.probes["C_coverage_integrity"] = round(answered / len(QUESTIONS), 2)

    # Probe A: Acoustic Fidelity (STT Sanity Checks)
    METRIC_TRIGGERS = ["kilo", " kg", "kilogram", "stone", " st "]
    if any(u in answer_lower for u in METRIC_TRIGGERS):
        acoustic_issue = "patient gave weight in non-imperial units (kg or stone). Ask patient to provide weight in pounds."
    elif ("current_weight" in topic_lower or ("weight" in topic_lower and "lost" not in topic_lower and "loss" not in topic_lower)):
        val = _extract_number(patient_answer)
        if val is not None:
            if val < 50 or val > 500:
                global_state.probes["A_acoustic_fidelity"] = 0.0
                acoustic_issue = f"reported weight of {int(val)} lbs is outside clinically plausible range (50–500 lbs). Likely STT mishearing."
            else:
                global_state.probes["A_acoustic_fidelity"] = 1.0
    elif ("weight_lost" in topic_lower or ("lost" in topic_lower and "weight" in topic_lower)):
        val = _extract_number(patient_answer)
        if val is not None and val > 30:
            global_state.probes["A_acoustic_fidelity"] = 0.0
            acoustic_issue = f"reported weight loss of {int(val)} lbs in one month is not clinically plausible (clinical max ~30 lbs/month). Likely STT mishearing."
    elif ("supply_days" in topic_lower or ("supply" in topic_lower and "days" in topic_lower) or ("days" in topic_lower and "medication" in topic_lower)):
        val = _extract_number(patient_answer)
        if val is not None and (val > 180 or val < 0):
            acoustic_issue = f"reported supply of {int(val)} days is outside expected range (0–180 days)."

    global_state.acoustic_flag = acoustic_issue
    return acoustic_issue

def check_cross_validation():
    """
    Evaluates 5 specific logical contradiction matrices to flag cognitive or STT errors.
    """
    def get_answer(label_fragment):
        for r in global_state.responses:
            if label_fragment.lower() in r["question"].lower():
                return r["answer"].lower()
        return ""

    contradictions_found = []
    
    # Check 1: Weight Loss vs Satisfaction
    weight_lost  = get_answer("how much weight have you lost")
    satisfaction = get_answer("satisfied with your rate")
    target_gap   = get_answer("how far do you feel you are from your target")
    
    if weight_lost and satisfaction and target_gap:
        lost_positive = any(w in weight_lost for w in ["pound", "lb", "kilo", "yes", "some"])
        sat_negative  = any(w in satisfaction for w in ["no", "not", "barely", "nothing"])
        if lost_positive and sat_negative:
            contradictions_found.append({
                "field_a": "weight_lost", "answer_a": weight_lost,
                "field_b": "satisfaction", "answer_b": satisfaction,
                "summary": "Patient reported weight loss but expressed dissatisfaction — perceived vs actual progress mismatch."
            })

    # Check 2: Supply Days vs Missed Doses
    missed_doses = get_answer("have you missed any doses")
    supply_days  = get_answer("how many days of medication supply")
    if supply_days and missed_doses:
        low_supply = any(w in supply_days for w in ["few", "3", "4", "5", "running out", "almost", "low", "one week"])
        no_missed  = any(w in missed_doses for w in ["no", "none", "haven't", "i haven"])
        if low_supply and no_missed:
            contradictions_found.append({
                "field_a": "missed_doses", "answer_a": missed_doses,
                "field_b": "supply_remaining", "answer_b": supply_days,
                "summary": "Patient denied missing doses but reported unexpectedly low remaining supply."
            })

    # Check 3: Diet Adherence vs Weight Loss
    diet = get_answer("following your recommended diet")
    if diet and weight_lost:
        diet_positive = any(w in diet for w in ["yes", "mostly", "trying", "doing well"])
        weight_negative = any(w in weight_lost for w in ["none", "zero", "gained", "haven't"])
        if diet_positive and weight_negative:
            contradictions_found.append({
                "field_a": "diet_adherence", "answer_a": diet,
                "field_b": "weight_lost", "answer_b": weight_lost,
                "summary": "Patient reports strong diet adherence but zero/negative weight loss."
            })

    # Check 4: Side Effects vs Treatment Concerns
    side_effects = get_answer("side effects from your medication")
    concerns = get_answer("concerns about continuing")
    if side_effects and concerns:
        severe_se = any(w in side_effects for w in ["severe", "terrible", "vomiting", "hospital", "pain"])
        no_concerns = any(w in concerns for w in ["no", "none", "fine", "nope"])
        if severe_se and no_concerns:
            contradictions_found.append({
                "field_a": "side_effects", "answer_a": side_effects,
                "field_b": "treatment_concerns", "answer_b": concerns,
                "summary": "Patient reported severe adverse effects but stated they have zero concerns continuing treatment."
            })

    # Check 5: Energy Levels vs Side Effects (Fatigue)
    energy = get_answer("rate your energy levels")
    if energy and side_effects:
        high_energy = any(w in energy for w in ["great", "high", "better", "good", "excellent"])
        fatigue_se = any(w in side_effects for w in ["tired", "fatigue", "exhausted", "sleepy"])
        if high_energy and fatigue_se:
            contradictions_found.append({
                "field_a": "energy_levels", "answer_a": energy,
                "field_b": "side_effects", "answer_b": side_effects,
                "summary": "Patient reported excellent energy levels but simultaneously listed fatigue as a side effect."
            })

    # Log new contradictions
    for c in contradictions_found:
        if c not in global_state.contradictions:
            global_state.contradictions.append(c)
            accumulate_signal("contradiction")
            global_state.probes["F_capture_accuracy"] = max(0.0, global_state.probes["F_capture_accuracy"] - 0.25)
            print(f"[CONTRADICTION] {c['summary']}")

# ─────────────────────────────────────────────────────────────
# LIVING MEMORY ROUTING ENGINE (v5)
# ─────────────────────────────────────────────────────────────
def recalculate_routing():
    """Evaluates the Living Memory state and determines the next instant routing action."""
    mem = global_state.living_memory
    mem["handoff_required"] = False
    mem["handoff_target"] = None

    if "safety" in mem["detected_flags"]:
        mem["recommended_next_action"] = "handoff_pharmacist"
        mem["handoff_required"] = True
        mem["handoff_target"] = "pharmacist"
    elif "pricing" in mem["detected_flags"]:
        mem["recommended_next_action"] = "handoff_billing"
        mem["handoff_required"] = True
        mem["handoff_target"] = "billing"
    elif "reschedule" in mem["detected_flags"]:
        mem["recommended_next_action"] = "handoff_scheduling"
        mem["handoff_required"] = True
        mem["handoff_target"] = "scheduling"
    elif mem["emotional_state"] == "distressed":
        mem["recommended_next_action"] = "empathy_probe"
    elif len(mem["questions_remaining"]) == 0:
        mem["recommended_next_action"] = "close_call"
    else:
        mem["recommended_next_action"] = "continue_interview"
        
    mem["coverage_pct"] = round(len(mem["questions_answered"]) / len(QUESTIONS), 2)

# ─────────────────────────────────────────────────────────────
# GHOST ANALYST (v5)
# ─────────────────────────────────────────────────────────────
def run_ghost_analyst_sync():
    """Runs a parallel LLM trace to generate real-time clinical observations."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        recent_tx = global_state.transcript[-6:]  # Last 3 exchanges
        if not recent_tx:
            return
            
        prompt = (
            "You are a senior clinical analyst watching this clinical AI call live. "
            "In ONE clear, punchy sentence, state the most important clinical observation or behavioral "
            "anomaly happening right now based on this recent transcript segment and current memory state.\n\n"
            f"Recent Transcript:\n{json.dumps(recent_tx, indent=2)}\n\n"
            f"Living Memory State:\n{json.dumps(global_state.living_memory, indent=2)}"
        )

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3
        )
        obs = res.choices[0].message.content.strip()
        alert = {
            "timestamp": time.time(),
            "message": obs
        }
        global_state.living_memory["ghost_alerts"].append(alert)
        print(f"[GHOST ANALYST] {obs}")
    except Exception as e:
        print(f"[GHOST ANALYST ERROR] {e}")

async def ghost_analyst_task():
    """Wrapper to prevent the async Ghost Analyst from blocking tool execution."""
    await asyncio.to_thread(run_ghost_analyst_sync)

def trigger_ghost_analyst():
    """Safely fires the Ghost Analyst on whatever event loop is active."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(ghost_analyst_task())
    except RuntimeError:
        threading.Thread(target=run_ghost_analyst_sync).start()


# ─────────────────────────────────────────────────────────────
# ELEVENLABS AGENT TOOLS (Exposed to HTTP endpoints)
# ─────────────────────────────────────────────────────────────

def get_memory_state(**kwargs) -> str:
    """
    [NEW v5 TOOL] Called by all agents before asking a question.
    Provides the single source of truth for the conversation state, preventing hallucination.
    """
    return json.dumps(global_state.living_memory)

def log_answer(question_topic: str = "", patient_answer: str = "",
               confidence: str = "clear", clinical_flag: str = "none", **kwargs):
    """Primary tool called by ElevenLabs to record patient responses."""
    print(f"[LOG] {question_topic} -> {patient_answer} [{clinical_flag}]")

    # Match slot
    idx = TOPIC_TO_INDEX.get(question_topic)
    if idx is not None and idx < len(global_state.responses):
        if not global_state.responses[idx]["answer"] or global_state.responses[idx]["auto_extracted"]:
            global_state.responses[idx]["answer"]         = patient_answer
            global_state.responses[idx]["flag"]           = clinical_flag
            global_state.responses[idx]["auto_extracted"] = False

    # Extract answers, score probes, accumulate
    extract_multi_answers(patient_answer, question_topic)
    acoustic_issue = score_probes_on_answer(patient_answer, clinical_flag, question_topic)
    accumulate_signal(clinical_flag)

    # ── LIVING MEMORY UPDATES (v5) ──
    mem = global_state.living_memory
    topic = question_topic.lower()
    
    if topic in mem["questions_remaining"]:
        mem["questions_remaining"].remove(topic)
    if topic not in mem["questions_answered"]:
        mem["questions_answered"].append(topic)

    # Map clinical_flag to emotional_state
    if clinical_flag in ["side_effect_severe", "safety_concern"]:
        mem["emotional_state"] = "distressed"
    elif clinical_flag in ["side_effect_moderate", "dissatisfied", "behavioral_uncertainty", "treatment_concerns"]:
        mem["emotional_state"] = "concerned"
    elif clinical_flag == "none" and mem["emotional_state"] != "distressed":
        mem["emotional_state"] = "engaged"

    # Map clinical_flag to detected_flags
    flag_keywords = {
        "safety_concern": "safety",
        "pricing_question": "pricing",
        "side_effect_severe": "safety",
        "side_effect_moderate": "nausea", 
    }
    mapped_flag = flag_keywords.get(clinical_flag)
    if mapped_flag and mapped_flag not in mem["detected_flags"]:
        mem["detected_flags"].append(mapped_flag)

    # Re-evaluate patient_risk_level
    if global_state.signal_score >= 7:
        mem["patient_risk_level"] = "HIGH"
    elif global_state.signal_score >= 3:
        mem["patient_risk_level"] = "MEDIUM"
    else:
        mem["patient_risk_level"] = "LOW"

    # Evaluate validation triggers
    validation_triggers = ["target weight", "days of medication", "harder to take", "change one thing", "pharmacy filled", "treatment plan"]
    if any(t in question_topic.lower() for t in validation_triggers):
        check_cross_validation()

    recalculate_routing()
    trigger_ghost_analyst()

    # Inform the AI agent if there was a data fidelity issue
    if acoustic_issue:
        print(f"[ACOUSTIC FLAG] {acoustic_issue}")
        return f"ACOUSTIC_FLAG: {acoustic_issue} Ask patient to confirm or correct."

    return (
        f"SUCCESS. State updated. "
        f"Handoff Required: {mem['handoff_required']}. "
        f"Target: {mem['handoff_target']}."
    )

def verify_identity(patient_confirmed_name: str = "", date_of_birth: str = "",
                    identity_match: str = "confirmed", **kwargs):
    print(f"[IDENTITY] {patient_confirmed_name} | {date_of_birth} | {identity_match}")
    if identity_match == "mismatch":
        global_state.probes["E_identity_signal"] = 0.0
    return f"Identity {identity_match}. Proceed accordingly."

def schedule_callback(preferred_day: str = "", preferred_time_window: str = "",
                      phone_confirmed: str = "confirmed", **kwargs):
    print(f"[CALLBACK] {preferred_day} {preferred_time_window}")
    if "reschedule" not in global_state.living_memory["detected_flags"]:
        global_state.living_memory["detected_flags"].append("reschedule")
    recalculate_routing()
    global_state.transcript.append({
        "role": "system",
        "message": f"[CALLBACK SCHEDULED] {preferred_day}, {preferred_time_window}",
        "timestamp": time.time()
    })
    return "Callback scheduled. End node sequence activated."

def escalate_to_pharmacist(symptom_description: str = "", urgency_level: str = "urgent",
                           patient_acknowledged_followup: str = "yes", **kwargs):
    print(f"[ESCALATION] {urgency_level}: {symptom_description}")
    global_state.probes["D_safety_compliance"] = 0.0
    accumulate_signal("safety_concern")
    
    if "safety" not in global_state.living_memory["detected_flags"]:
        global_state.living_memory["detected_flags"].append("safety")
    recalculate_routing()

    global_state.transcript.append({
        "role": "system",
        "message": f"[ESCALATION] {urgency_level.upper()}: {symptom_description}",
        "timestamp": time.time()
    })
    return "Escalation flagged. Handing off to pharmacist."

def capture_pricing_concern(concern_description: str = "", patient_wants_followup: str = "yes", **kwargs):
    print(f"[PRICING] {concern_description}")
    accumulate_signal("pricing_question")
    
    if "pricing" not in global_state.living_memory["detected_flags"]:
        global_state.living_memory["detected_flags"].append("pricing")
    recalculate_routing()
    trigger_ghost_analyst()

    return "Pricing concern logged. Handing off to billing specialist."

def flag_contradiction(field_a: str = "", answer_a: str = "", field_b: str = "", answer_b: str = "",
                       contradiction_summary: str = "", **kwargs):
    print(f"[CONTRADICTION TOOL] {field_a} vs {field_b}")
    c = {
        "field_a": field_a, "answer_a": answer_a,
        "field_b": field_b, "answer_b": answer_b,
        "summary": contradiction_summary
    }
    if c not in global_state.contradictions:
        global_state.contradictions.append(c)
        accumulate_signal("contradiction")
        global_state.probes["F_capture_accuracy"] = max(0.0, global_state.probes["F_capture_accuracy"] - 0.25)
    return "Contradiction logged. Probe gently."

def end_call(outcome: str = "completed", refill_approved: str = "yes", summary: str = "", **kwargs):
    answered = sum(1 for r in global_state.responses if r["answer"])
    global_state.probes["C_coverage_integrity"] = 1.0 if answered >= 14 else round(answered / 14, 2)
    return "Call marked for termination."

def detect_ai_inquiry(patient_question: str = "", response_given: str = "",
                      patient_accepted_continuation: str = "yes", **kwargs):
    print(f"[AI INQUIRY] Q: {patient_question} | Accepted: {patient_accepted_continuation}")
    global_state.ai_inquiry_logged = True
    global_state.transcript.append({
        "role": "system",
        "message": f"[AI INQUIRY] Patient asked: '{patient_question}'. Continuation accepted: {patient_accepted_continuation}",
        "timestamp": time.time()
    })
    if patient_accepted_continuation == "yes":
        accumulate_signal("behavioral_uncertainty")
    return "AI inquiry logged. Continue check-in if accepted, else trigger callback."

def log_emotional_distress(distress_type: str = "", patient_statement: str = "",
                           action_taken: str = "", urgency: str = "low", **kwargs):
    print(f"[EMOTIONAL] {distress_type} | urgency={urgency}")
    event = {
        "distress_type":     distress_type,
        "patient_statement": patient_statement,
        "action_taken":      action_taken,
        "urgency":           urgency,
        "timestamp":         time.time()
    }
    global_state.emotional_events.append(event)
    global_state.living_memory["emotional_state"] = "distressed"
    recalculate_routing()

    if urgency == "high":
        accumulate_signal("safety_concern")
        global_state.probes["G_behavioral_signal"] = 0.0
    elif urgency == "medium":
        accumulate_signal("behavioral_uncertainty")

    return f"Emotional event logged ({distress_type}). Adapt routing if necessary."

def handle_caregiver_proxy(relationship_to_patient: str = "", caregiver_name: str = "",
                           patient_available: str = "no", **kwargs):
    print(f"[PROXY] {relationship_to_patient} | available={patient_available}")
    note = f"Call answered by {relationship_to_patient} ({caregiver_name}). Available: {patient_available}."
    global_state.caregiver_proxy = note
    global_state.probes["E_identity_signal"] = 0.5
    return "Caregiver proxy logged. If unavailable, call schedule_callback."

def request_human_transfer(reason: str = "", urgency: str = "normal", **kwargs):
    print(f"[HUMAN TRANSFER] reason={reason} urgency={urgency}")
    global_state.human_transfer_req = True
    accumulate_signal("behavioral_uncertainty")
    
    if urgency == "urgent" and "safety" not in global_state.living_memory["detected_flags"]:
        global_state.living_memory["detected_flags"].append("safety")
    else:
        if "reschedule" not in global_state.living_memory["detected_flags"]:
            global_state.living_memory["detected_flags"].append("reschedule")
            
    recalculate_routing()
    return "Transfer requested. Living memory updated to enforce handoff."