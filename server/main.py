# server/main.py
# SPECTER v4 — Clinical Intelligence Voice Agent Backend
# Innovations added:
#   1. GLP-1 Churn Risk Matrix  (evidence-anchored risk factors)
#   2. Longitudinal Risk Delta   (cross-call trend detection)
#   3. MI Fidelity Score         (agent Motivational Interviewing quality — Probe H)
#   4. Hesitation Fingerprint    (patient linguistic evasion analysis)

import os
import time
import threading
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation, ClientTools, ConversationInitiationData
)
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from deliberation.judge import run_deliberation, run_final_llm_deliberation

load_dotenv()
app      = FastAPI(title="TrimRX SPECTER Backend v4")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")


# ─────────────────────────────────────────────────────────────
# 14 PRIMARY QUESTIONS + 5 HIDDEN VALIDATION PROBES = 19 TOTAL
# ─────────────────────────────────────────────────────────────
QUESTIONS = [
    # Primary 14 (matches hackathon requirement)
    "How have you been feeling overall?",
    "What's your current weight in pounds?",
    "How much weight have you lost this past month in pounds?",
    "Have you missed any doses this past month?",
    "Any side effects from your medication this month?",
    "Satisfied with your rate of weight loss?",
    "Have you started any new medications or supplements since last month?",
    "Any new allergies?",
    "Any surgeries since your last check-in?",
    "How would you rate your energy levels compared to when you started?",
    "Have you been following your recommended diet while on this medication?",
    "Do you have any concerns about continuing your treatment plan?",
    "Any questions for your doctor?",
    "Has your shipping address changed?",
    # Hidden validation probes (5)
    "How far do you feel you are from your target weight?",
    "How many days of medication supply do you have left?",
    "Has anything made it harder to take your medication consistently?",
    "If you could change one thing about your treatment, what would it be?",
    "Has your pharmacy filled anything new for you recently?",
]

QUESTION_LABELS = {
    "How have you been feeling overall?":                                     "Overall",
    "What's your current weight in pounds?":                                  "Weight",
    "How much weight have you lost this past month in pounds?":               "Weight Loss",
    "Have you missed any doses this past month?":                             "Missed Doses",
    "Any side effects from your medication this month?":                      "Side Effects",
    "Satisfied with your rate of weight loss?":                               "Satisfaction",
    "Have you started any new medications or supplements since last month?":   "New Meds",
    "Any new allergies?":                                                     "Allergies",
    "Any surgeries since your last check-in?":                                "Surgeries",
    "How would you rate your energy levels compared to when you started?":    "Energy Levels",
    "Have you been following your recommended diet while on this medication?": "Diet Adherence",
    "Do you have any concerns about continuing your treatment plan?":         "Treatment Concerns",
    "Any questions for your doctor?":                                         "Dr. Questions",
    "Has your shipping address changed?":                                     "Address",
    "How far do you feel you are from your target weight?":                   "[V] Target Gap",
    "How many days of medication supply do you have left?":                   "[V] Supply Days",
    "Has anything made it harder to take your medication consistently?":      "[V] Adherence",
    "If you could change one thing about your treatment, what would it be?":  "[V] Change Ask",
    "Has your pharmacy filled anything new for you recently?":                "[V] Pharmacy",
}

TOPIC_TO_INDEX = {
    "overall_feeling":      0,
    "current_weight":       1,
    "weight_lost":          2,
    "missed_doses":         3,
    "side_effects":         4,
    "satisfaction":         5,
    "new_medications":      6,
    "new_allergies":        7,
    "surgeries":            8,
    "energy_levels":        9,
    "diet_adherence":       10,
    "treatment_concerns":   11,
    "doctor_questions":     12,
    "address_change":       13,
    "target_gap":           14,
    "supply_days":          15,
    "adherence_difficulty": 16,
    "change_request":       17,
    "pharmacy_recent":      18,
}

PROBE_LABELS = {
    "A_acoustic_fidelity":  "Acoustic",
    "B_intent_alignment":   "Intent",
    "C_coverage_integrity": "Coverage",
    "D_safety_compliance":  "Safety",
    "E_identity_signal":    "Identity",
    "F_capture_accuracy":   "Capture",
    "G_behavioral_signal":  "Behavioral",
    "H_mi_fidelity":        "MI Fidelity",   # NEW v4
}

SIGNAL_WEIGHTS = {
    "none":                   0,
    "side_effect_mild":       1,
    "behavioral_uncertainty": 2,
    "dissatisfied":           2,
    "contradiction":          3,
    "safety_concern":         5,
    "side_effect_moderate":   3,
    "side_effect_severe":     8,
    "pricing_question":       1,
    "refused":                2,
}

EXTRACTION_HINTS = {
    "overall_feeling":    ["feeling", "doing well", "doing good", "been okay", "been great", "been rough", "been bad"],
    "weight_lost":        ["lost", "dropped", "down", "shed", "fewer pounds"],
    "missed_doses":       ["missed", "forgot", "skipped", "didn't take", "haven't taken"],
    "side_effects":       ["nausea", "nauseous", "vomiting", "headache", "dizzy", "fatigue", "tired", "constipated", "diarrhea", "side effect"],
    "satisfaction":       ["satisfied", "happy with", "not happy", "pleased", "not pleased", "disappointed", "working well", "not working"],
    "new_medications":    ["started taking", "new prescription", "doctor prescribed", "also taking", "blood pressure", "metformin", "new medication", "new supplement"],
    "new_allergies":      ["allergic", "allergy", "reaction to", "broke out"],
    "energy_levels":      ["energy", "more energy", "less energy", "tired", "fatigue", "energetic"],
    "diet_adherence":     ["diet", "eating", "food", "calories", "healthy eating", "meal plan", "cheating"],
    "treatment_concerns": ["concerned about", "worried about", "thinking about stopping", "not sure", "unsure about continuing"],
    "doctor_questions":   ["ask the doctor", "question for", "wondering about dosage", "want to know"],
}


# ─────────────────────────────────────────────────────────────
# CALL STATE
# ─────────────────────────────────────────────────────────────
class CallState:
    def __init__(self):
        self.responses = [
            {"question": q, "answer": "", "flag": "none", "auto_extracted": False}
            for q in QUESTIONS
        ]
        self.outcome              = None
        self.transcript           = []
        self.insights             = None
        self.active_conversation  = None
        self.start_time           = None
        self.call_duration        = 0
        self.behavior_mode        = "STANDBY"
        self._prev_g_score        = 1.0
        self.signal_score         = 0
        self.contradictions       = []
        self.interview_depth      = "standard"
        self.acoustic_flag        = None
        self.ai_inquiry_logged    = False
        self.emotional_events     = []
        self.caregiver_proxy      = None
        self.human_transfer_req   = False
        self.extracted_answers    = []
        # NEW v4 fields
        self.previous_call_context = ""
        self.hesitation_fingerprint = None
        self.probes = {
            "A_acoustic_fidelity":  1.0,
            "B_intent_alignment":   1.0,
            "C_coverage_integrity": 1.0,
            "D_safety_compliance":  1.0,
            "E_identity_signal":    1.0,
            "F_capture_accuracy":   1.0,
            "G_behavioral_signal":  1.0,
            "H_mi_fidelity":        1.0,   # NEW v4 — Motivational Interviewing probe
        }


global_state = CallState()


# ─────────────────────────────────────────────────────────────
# SIGNAL ACCUMULATION
# ─────────────────────────────────────────────────────────────
def accumulate_signal(flag: str):
    weight = SIGNAL_WEIGHTS.get(flag, 0)
    global_state.signal_score += weight
    if global_state.signal_score >= 7:
        global_state.interview_depth = "deep_probe"
    elif global_state.signal_score >= 3:
        global_state.interview_depth = "soft_probe"
    else:
        global_state.interview_depth = "standard"
    print(f"[SIGNAL] score={global_state.signal_score} flag={flag} depth={global_state.interview_depth}")


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION ENGINE (5 checks)
# ─────────────────────────────────────────────────────────────
def check_cross_validation():
    def get_answer(label_fragment):
        for r in global_state.responses:
            if label_fragment.lower() in r["question"].lower():
                return r["answer"].lower()
        return ""

    contradictions_found = []

    weight_lost  = get_answer("how much weight have you lost")
    satisfaction = get_answer("satisfied with your rate")
    target_gap   = get_answer("how far do you feel you are from your target")
    if weight_lost and satisfaction and target_gap:
        lost_positive = any(w in weight_lost for w in ["pound", "lb", "kilo"])
        sat_negative  = any(w in satisfaction for w in ["no", "not", "barely", "nothing"])
        if lost_positive and sat_negative:
            contradictions_found.append({
                "field_a": "weight_lost", "answer_a": weight_lost,
                "field_b": "satisfaction", "answer_b": satisfaction,
                "summary": "Patient reported weight loss but expressed dissatisfaction — "
                           "perceived vs actual progress mismatch."
            })

    missed_doses = get_answer("have you missed any doses")
    supply_days  = get_answer("how many days of medication supply")
    if supply_days:
        low_supply = any(w in supply_days for w in
                         ["few", "3", "4", "5", "running out", "almost", "low", "one week"])
        no_missed  = any(w in missed_doses for w in ["no", "none", "haven't", "i haven"])
        if low_supply and no_missed:
            contradictions_found.append({
                "field_a": "missed_doses", "answer_a": missed_doses,
                "field_b": "supply_remaining", "answer_b": supply_days,
                "summary": "Patient denied missing doses but reported very low remaining supply."
            })

    new_meds = get_answer("started any new medications")
    pharmacy = get_answer("has your pharmacy filled anything new")
    if new_meds and pharmacy:
        denied_new   = any(w in new_meds  for w in ["no", "none", "nothing", "haven"])
        pharmacy_new = any(w in pharmacy  for w in ["yes", "they did", "actually", "one", "new"])
        if denied_new and pharmacy_new:
            contradictions_found.append({
                "field_a": "new_medications", "answer_a": new_meds,
                "field_b": "pharmacy_recent", "answer_b": pharmacy,
                "summary": "Patient denied new medications but confirmed recent pharmacy fill."
            })

    side_effects = get_answer("any side effects")
    adherence    = get_answer("has anything made it harder")
    if side_effects and adherence:
        denied_se   = any(w in side_effects for w in ["no", "none", "nothing"])
        harder_true = any(w in adherence    for w in
                          ["yes", "nausea", "sick", "tired", "forget", "hard", "difficult"])
        if denied_se and harder_true:
            contradictions_found.append({
                "field_a": "side_effects", "answer_a": side_effects,
                "field_b": "adherence_difficulty", "answer_b": adherence,
                "summary": "Patient denied side effects but reported difficulty with adherence."
            })

    treatment_concerns = get_answer("do you have any concerns about continuing")
    if missed_doses and treatment_concerns:
        no_missed_d  = any(w in missed_doses for w in ["no", "none", "haven't"])
        has_concerns = any(w in treatment_concerns for w in
                           ["yes", "thinking about stopping", "not sure", "worried", "concerned"])
        if no_missed_d and has_concerns:
            contradictions_found.append({
                "field_a": "missed_doses", "answer_a": missed_doses,
                "field_b": "treatment_concerns", "answer_b": treatment_concerns,
                "summary": "Patient denied missing doses but expressed concerns about continuing — "
                           "possible early churn signal."
            })

    for c in contradictions_found:
        if c not in global_state.contradictions:
            global_state.contradictions.append(c)
            accumulate_signal("contradiction")
            global_state.probes["F_capture_accuracy"] = max(
                0.0, global_state.probes["F_capture_accuracy"] - 0.25
            )
            print(f"[CONTRADICTION] {c['summary']}")


# ─────────────────────────────────────────────────────────────
# MULTI-ANSWER EXTRACTION ENGINE
# ─────────────────────────────────────────────────────────────
def extract_multi_answers(patient_answer: str, active_topic: str):
    answer_lower = patient_answer.lower()
    word_count   = len(answer_lower.split())
    if word_count < 10:
        return

    extracted = []
    for topic, keywords in EXTRACTION_HINTS.items():
        if topic == active_topic:
            continue
        idx = TOPIC_TO_INDEX.get(topic)
        if idx is None:
            continue
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
                    start   = max(0, i - 2)
                    end     = min(len(words), i + 8)
                    snippet = " ".join(words[start:end])
                    global_state.responses[idx]["answer"]         = snippet
                    global_state.responses[idx]["flag"]           = "none"
                    global_state.responses[idx]["auto_extracted"]  = True
                    extracted.append({"topic": topic, "keyword": matched_keyword, "snippet": snippet})
                    print(f"[MULTI-EXTRACT] topic={topic} kw='{matched_keyword}' snippet='{snippet}'")
                    break

    if extracted:
        global_state.extracted_answers.extend(extracted)
        global_state.transcript.append({
            "role":      "system",
            "message":   f"[MULTI-EXTRACT] {len(extracted)} additional answer(s) captured: "
                         + ", ".join(e["topic"] for e in extracted),
            "timestamp": time.time()
        })
    return extracted


# ─────────────────────────────────────────────────────────────
# NUMERIC EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────
def _extract_number(text: str):
    digits = ''.join(c for c in text if c.isdigit() or c == '.')
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────
# PROBE SCORER (inline — 7 probes, runs on every answer)
# ─────────────────────────────────────────────────────────────
def score_probes_on_answer(patient_answer: str, flag: str, question_topic: str = ""):
    answer_lower  = patient_answer.lower()
    topic_lower   = question_topic.lower()
    acoustic_issue = None

    OPT_OUT = ["stop calling", "remove me", "don't call", "wrong number",
               "who is this", "not interested"]
    if any(t in answer_lower for t in OPT_OUT):
        global_state.probes["B_intent_alignment"] = 0.0

    SAFETY = ["should i take", "what should i do", "is it safe",
              "can i mix", "what is the dosage", "can i stop"]
    if any(t in answer_lower for t in SAFETY):
        global_state.probes["D_safety_compliance"] = 0.0
        accumulate_signal("safety_concern")

    prev_g = global_state.probes["G_behavioral_signal"]
    if flag == "behavioral_uncertainty":
        global_state.probes["G_behavioral_signal"] = 0.0
        global_state._prev_g_score = 0.0
    elif flag == "none" and prev_g == 0.0:
        global_state.probes["G_behavioral_signal"] = 1.0
        global_state._prev_g_score = 1.0

    answered = sum(1 for r in global_state.responses if r["answer"])
    global_state.probes["F_capture_accuracy"]   = round(answered / len(QUESTIONS), 2)
    global_state.probes["C_coverage_integrity"] = round(answered / len(QUESTIONS), 2)

    METRIC_TRIGGERS = ["kilo", " kg", "kilogram", "stone", " st "]
    if any(u in answer_lower for u in METRIC_TRIGGERS):
        acoustic_issue = (
            "patient gave weight in non-imperial units (kg or stone). "
            "Ask patient to provide weight in pounds."
        )
    elif ("current_weight" in topic_lower or
          ("weight" in topic_lower
           and "lost" not in topic_lower
           and "loss" not in topic_lower)):
        val = _extract_number(patient_answer)
        if val is not None:
            if val < 50 or val > 500:
                global_state.probes["A_acoustic_fidelity"] = 0.0
                acoustic_issue = (
                    f"reported weight of {int(val)} lbs is outside clinically plausible "
                    f"range (50–500 lbs). Likely STT mishearing or patient error."
                )
            else:
                global_state.probes["A_acoustic_fidelity"] = 1.0
    elif ("weight_lost" in topic_lower or
          ("lost" in topic_lower and "weight" in topic_lower)):
        val = _extract_number(patient_answer)
        if val is not None and val > 30:
            global_state.probes["A_acoustic_fidelity"] = 0.0
            acoustic_issue = (
                f"reported weight loss of {int(val)} lbs in one month is not clinically "
                f"plausible (clinical max ~30 lbs/month). Likely STT mishearing."
            )
    elif ("supply_days" in topic_lower or
          ("supply" in topic_lower and "days" in topic_lower) or
          ("days" in topic_lower and "medication" in topic_lower)):
        val = _extract_number(patient_answer)
        if val is not None and (val > 180 or val < 0):
            acoustic_issue = (
                f"reported supply of {int(val)} days is outside expected range (0–180 days)."
            )

    global_state.acoustic_flag = acoustic_issue
    return acoustic_issue


# ─────────────────────────────────────────────────────────────
# INNOVATION: MI FIDELITY COMPUTATION (Probe H)
# Computes Motivational Interviewing quality from agent transcript.
# Based on: JMIR 2025 MI-AI scoping review, OARS framework metrics.
# Grounded in: reflection-to-question ratio, open question usage,
#              empathy language, banned phrase avoidance.
# ─────────────────────────────────────────────────────────────
def compute_mi_fidelity(transcript: list) -> dict:
    """
    Computes MI fidelity score (0.0–1.0) from agent transcript turns.
    Returns both the score and structured metrics for the dossier.
    """
    agent_turns = [t["message"] for t in transcript if t.get("role") == "agent"]
    if not agent_turns:
        return {"score": 1.0, "metrics": {}}

    # R:Q Ratio — core MI metric (target: >1.0, ideal: 2.0+)
    questions   = sum(1 for t in agent_turns if t.strip().endswith("?"))
    reflections = sum(1 for t in agent_turns if not t.strip().endswith("?"))
    rq_ratio    = round(reflections / max(questions, 1), 2)
    rq_score    = min(1.0, rq_ratio / 2.0)

    # Open question ratio (OARS: Open questions > Closed questions)
    OPEN_STARTS = ["how ", "what ", "can you tell", "could you describe",
                   "tell me more", "help me understand", "walk me through"]
    open_qs   = sum(1 for t in agent_turns
                    if any(t.lower().startswith(o) or o in t.lower()[:40] for o in OPEN_STARTS)
                    and t.strip().endswith("?"))
    open_ratio = round(open_qs / max(questions, 1), 2)
    open_score = min(1.0, open_ratio / 0.6)  # 60% open = full score

    # Empathy language (OARS: Affirmations + Reflective listening)
    EMPATHY_PHRASES = [
        "i'm sorry to hear", "i hear you", "that must", "completely normal",
        "thank you for sharing", "i'll make a note", "i appreciate",
        "i understand", "that's really", "take all the time"
    ]
    empathy_count = sum(
        1 for t in agent_turns
        for ep in EMPATHY_PHRASES if ep in t.lower()
    )
    empathy_score = min(1.0, empathy_count / max(len(agent_turns) * 0.2, 1))

    # Banned phrases (OARS violation: confrontation / unsolicited advice)
    BANNED = ["great!", "awesome!", "absolutely!", "of course!", "no problem!"]
    banned_count = sum(1 for t in agent_turns for b in BANNED if b.lower() in t.lower())
    banned_penalty = min(0.4, banned_count * 0.12)

    # Compute overall MI score
    raw_score = (
        (rq_score    * 0.40) +
        (open_score  * 0.25) +
        (empathy_score * 0.35)
    ) - banned_penalty
    final_score = round(max(0.0, min(1.0, raw_score)), 2)

    metrics = {
        "overall_score":        final_score,
        "rq_ratio":             rq_ratio,
        "rq_target":            "≥ 1.0 (ideal ≥ 2.0)",
        "open_question_ratio":  open_ratio,
        "empathy_events":       empathy_count,
        "banned_phrase_count":  banned_count,
        "total_agent_turns":    len(agent_turns),
        "assessment":           (
            "Strong MI adherence" if final_score >= 0.75
            else "Adequate MI quality" if final_score >= 0.50
            else "MI improvement needed"
        )
    }
    print(f"[MI FIDELITY] score={final_score} R:Q={rq_ratio} open={open_ratio} "
          f"empathy={empathy_count} banned={banned_count}")
    return {"score": final_score, "metrics": metrics}


# ─────────────────────────────────────────────────────────────
# INNOVATION: HESITATION FINGERPRINT
# Detects patient linguistic evasion / avoidance patterns.
# Grounded in: voice biomarker research 2025, MI theory on
#              "sustain talk" and ambivalence detection.
# ─────────────────────────────────────────────────────────────
def analyze_hesitation(transcript: list) -> dict:
    """
    Analyzes patient transcript for hesitation and evasion patterns.
    Returns structured hesitation fingerprint for the dossier.
    """
    user_turns  = [t["message"] for t in transcript if t.get("role") == "user"]
    agent_turns = [t["message"] for t in transcript if t.get("role") == "agent"]

    if not user_turns:
        return None

    # 1. Short responses to open questions (evasion signal)
    OPEN_Q_WORDS = ["how", "what ", "can you", "could you", "tell me", "describe", "walk me"]
    short_to_open = 0
    for i, agent_msg in enumerate(agent_turns):
        if any(w in agent_msg.lower()[:50] for w in OPEN_Q_WORDS) and agent_msg.strip().endswith("?"):
            if i < len(user_turns):
                if len(user_turns[i].split()) < 5:
                    short_to_open += 1

    # 2. Hedge / uncertainty language (ambivalence signal)
    HEDGES = [
        "i think", "maybe", "kind of", "sort of", "i'm not sure",
        "i guess", "possibly", "probably", "i don't know", "not really",
        "i'm not certain", "i believe", "i suppose"
    ]
    hedge_count = sum(1 for t in user_turns for h in HEDGES if h in t.lower())

    # 3. Very short answers (deflection signal)
    very_short = sum(1 for t in user_turns if len(t.split()) < 4)

    # 4. Average response length
    avg_len = round(sum(len(t.split()) for t in user_turns) / len(user_turns), 1)

    # 5. Topic deflection — answers that don't contain words from the question
    deflection_count = 0
    for i, agent_msg in enumerate(agent_turns):
        if i < len(user_turns) and agent_msg.strip().endswith("?"):
            q_words = set(agent_msg.lower().split()) - {"the", "a", "an", "is", "are", "do", "did", "have", "you", "your", "i", "me"}
            u_words = set(user_turns[i].lower().split())
            if len(q_words) > 0 and len(q_words & u_words) == 0 and len(user_turns[i].split()) < 8:
                deflection_count += 1

    # Compute composite hesitation index (0 = none, 1 = maximum)
    index_raw = (
        min(short_to_open, 5) / 5 * 0.25 +
        min(hedge_count, 10)  / 10 * 0.30 +
        min(very_short, 5)    / 5  * 0.20 +
        max(0, (10 - avg_len)) / 10 * 0.15 +
        min(deflection_count, 4) / 4 * 0.10
    )
    hesitation_index = round(min(1.0, index_raw), 2)
    level = "HIGH" if hesitation_index >= 0.60 else "ELEVATED" if hesitation_index >= 0.35 else "LOW"

    # Detect notable patterns
    notable_patterns = []
    if short_to_open >= 2:
        notable_patterns.append(f"Short responses to {short_to_open} open questions — possible evasion")
    if hedge_count >= 4:
        notable_patterns.append(f"High uncertainty language: {hedge_count} hedging expressions")
    if very_short >= 3:
        notable_patterns.append(f"{very_short} very short answers (under 4 words)")
    if avg_len < 7:
        notable_patterns.append(f"Below-average response length ({avg_len} words/turn)")
    if deflection_count >= 2:
        notable_patterns.append(f"{deflection_count} potential topic deflections detected")

    result = {
        "hesitation_index":         hesitation_index,
        "level":                    level,
        "short_open_responses":     short_to_open,
        "hedge_count":              hedge_count,
        "very_short_answers":       very_short,
        "average_response_length":  avg_len,
        "topic_deflections":        deflection_count,
        "notable_patterns":         notable_patterns,
    }
    print(f"[HESITATION] index={hesitation_index} level={level} hedges={hedge_count} "
          f"short={very_short} avg_len={avg_len}")
    return result


# ─────────────────────────────────────────────────────────────
# GRAPH DATA
# ─────────────────────────────────────────────────────────────
def build_graph_data() -> dict:
    nodes = [{"id": "SPECTER", "group": 1}]
    links = []
    safety_failed   = global_state.probes["D_safety_compliance"] < 1.0
    behavioral_flag = global_state.probes["G_behavioral_signal"]  < 1.0
    contradicted_qs = {c["field_a"] for c in global_state.contradictions} | \
                      {c["field_b"] for c in global_state.contradictions}
    for item in global_state.responses:
        label        = QUESTION_LABELS.get(item["question"], item["question"][:14])
        answered     = bool(item["answer"])
        is_validation = label.startswith("[V]")
        if answered:
            topic      = item["question"].lower()
            is_flagged = (
                (safety_failed   and "side effect" in topic) or
                (behavioral_flag and "overall"     in topic) or
                any(cf in topic for cf in contradicted_qs)
            )
            group = 3 if is_flagged else (5 if is_validation else 2)
        else:
            group = 4 if not is_validation else 6
        nodes.append({"id": label, "group": group})
        links.append({"source": "SPECTER", "target": label})
    return {"nodes": nodes, "links": links}


# ─────────────────────────────────────────────────────────────
# BEHAVIOR MODE
# ─────────────────────────────────────────────────────────────
def compute_behavior_mode() -> str:
    if global_state.outcome:
        return "SYNTHESIS COMPLETE"
    if global_state.acoustic_flag:
        return "ACOUSTIC CLARIFICATION"
    if global_state.human_transfer_req:
        return "HUMAN TRANSFER PENDING"
    if global_state.emotional_events:
        latest = global_state.emotional_events[-1].get("distress_type", "")
        if latest in ["psychological_distress", "bereavement"]:
            return "EMPATHY CRITICAL"
        return "EMPATHY PROTOCOL"
    if global_state.ai_inquiry_logged:
        return "AI DISCLOSURE HANDLED"
    if global_state.probes["B_intent_alignment"] < 1.0:
        return "DE-ESCALATION"
    if global_state.probes["D_safety_compliance"] < 1.0:
        return "CLINICAL ADVISORY"
    if global_state.probes["G_behavioral_signal"] < 1.0:
        return "EMPATHY VALIDATION"
    # NEW v4: hesitation state
    if (global_state.hesitation_fingerprint and
            global_state.hesitation_fingerprint.get("level") == "HIGH"):
        return "HESITATION DETECTED"
    depth = global_state.interview_depth
    if depth == "deep_probe":
        return "DEEP PROBE ACTIVE"
    if depth == "soft_probe":
        return "SOFT PROBE ACTIVE"
    if global_state.contradictions:
        return "CONTRADICTION AUDIT"
    EMPATHY_WORDS = ["sorry to hear", "understand", "that must", "completely normal",
                     "side effect", "nausea", "vomiting", "dizzy", "headache", "fatigue"]
    recent = [t["message"].lower() for t in global_state.transcript[-6:]
              if t.get("role") == "agent"]
    for u in recent:
        if any(w in u for w in EMPATHY_WORDS):
            return "EMPATHY PROTOCOL"
    answered = sum(1 for r in global_state.responses if r["answer"])
    if answered == 0:              return "STANDARD CLINICAL"
    if answered >= len(QUESTIONS): return "COVERAGE COMPLETE"
    return "STANDARD CLINICAL"


# ─────────────────────────────────────────────────────────────
# CALL RUNNER
# ─────────────────────────────────────────────────────────────
def run_call(patient_name: str, medication: str, patient_context: str,
             previous_call_context: str = ""):
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    # Store previous call context on state for longitudinal delta
    global_state.previous_call_context = previous_call_context

    def trigger_auto_end(reason="completed"):
        if global_state.outcome:
            return
        print(f"[KILL SWITCH] {reason}")
        global_state.outcome = reason
        if reason == "wrong_number":
            global_state.probes["E_identity_signal"] = 0.0
        if global_state.active_conversation:
            try:
                global_state.active_conversation.end_session()
            except Exception:
                pass

        def run_llm_task():
            try:
                # ── Compute MI Fidelity (Probe H) from transcript ──
                mi_result = compute_mi_fidelity(global_state.transcript)
                global_state.probes["H_mi_fidelity"] = mi_result["score"]
                mi_metrics = mi_result["metrics"]
                print(f"[PROBE H] MI Fidelity = {mi_result['score']}")

                # ── Compute Hesitation Fingerprint ──
                hesitation = analyze_hesitation(global_state.transcript)
                global_state.hesitation_fingerprint = hesitation

                # ── Run full 4-round deliberation ──
                global_state.insights = run_final_llm_deliberation(
                    transcript=global_state.transcript,
                    responses=global_state.responses,
                    previous_call_context=global_state.previous_call_context,
                    mi_metrics=mi_metrics,
                    hesitation_data=hesitation,
                )

                # ── Append event logs to edge case report ──
                if global_state.contradictions:
                    c_text = "\n\n[CROSS-VALIDATION FLAGS]\n"
                    for i, c in enumerate(global_state.contradictions, 1):
                        c_text += f"{i}. {c['summary']}\n"
                    global_state.insights["edge_case_report"] += c_text
                if global_state.emotional_events:
                    e_text = "\n\n[EMOTIONAL DISTRESS EVENTS]\n"
                    for ev in global_state.emotional_events:
                        e_text += f"- {ev.get('distress_type','unknown')}: {ev.get('patient_statement','')}\n"
                    global_state.insights["edge_case_report"] += e_text
                if global_state.ai_inquiry_logged:
                    global_state.insights["edge_case_report"] += \
                        "\n\n[AI DISCLOSURE] Patient asked if Jessica is AI. Handled per protocol."
                if global_state.caregiver_proxy:
                    global_state.insights["edge_case_report"] += \
                        f"\n\n[CAREGIVER PROXY] {global_state.caregiver_proxy}"
                if global_state.extracted_answers:
                    ex_text = f"\n\n[MULTI-ANSWER EXTRACTION]\n"
                    ex_text += f"{len(global_state.extracted_answers)} additional data point(s) from incidental speech:\n"
                    for ex in global_state.extracted_answers:
                        ex_text += f"  - {ex['topic']}: '{ex['snippet']}'\n"
                    global_state.insights["edge_case_report"] += ex_text
                # NEW v4: append MI and hesitation to edge report
                if global_state.hesitation_fingerprint:
                    h = global_state.hesitation_fingerprint
                    global_state.insights["edge_case_report"] += (
                        f"\n\n[HESITATION FINGERPRINT]\n"
                        f"Index: {h.get('hesitation_index')} ({h.get('level')})\n"
                        + "\n".join(f"  - {p}" for p in h.get("notable_patterns", []))
                    )
            except Exception as e:
                print("LLM ERROR:", e)

        threading.Thread(target=run_llm_task).start()

    # ──────────────────────────────────────────────
    # ALL 11 TOOLS
    # ──────────────────────────────────────────────
    def log_answer(question_topic: str = "", patient_answer: str = "",
                   confidence: str = "clear", clinical_flag: str = "none", **kwargs):
        print(f"[LOG] {question_topic} -> {patient_answer} [{clinical_flag}]")

        # Match question slot by topic
        for item in global_state.responses:
            if (question_topic.lower() in item["question"].lower() or
                    item["question"].lower() in question_topic.lower()):
                item["answer"]        = patient_answer
                item["flag"]          = clinical_flag
                item["auto_extracted"] = False
                break

        # Topic-key based matching (more precise)
        idx = TOPIC_TO_INDEX.get(question_topic)
        if idx is not None and idx < len(global_state.responses):
            if not global_state.responses[idx]["answer"] or global_state.responses[idx]["auto_extracted"]:
                global_state.responses[idx]["answer"]        = patient_answer
                global_state.responses[idx]["flag"]          = clinical_flag
                global_state.responses[idx]["auto_extracted"] = False

        extract_multi_answers(patient_answer, question_topic)
        acoustic_issue = score_probes_on_answer(patient_answer, clinical_flag, question_topic)
        accumulate_signal(clinical_flag)

        validation_triggers = [
            "target weight", "days of medication", "harder to take",
            "change one thing", "pharmacy filled", "treatment plan",
        ]
        if any(t in question_topic.lower() for t in validation_triggers):
            check_cross_validation()

        try:
            ans_dict = {i["question"]: i["answer"]
                        for i in global_state.responses if i["answer"]}
            global_state.insights = run_deliberation(ans_dict, global_state.transcript, [], True)
        except Exception:
            pass

        if acoustic_issue:
            print(f"[ACOUSTIC FLAG] {acoustic_issue}")
            return (
                f"ACOUSTIC_FLAG: {acoustic_issue} "
                f"DO NOT proceed. Ask patient to confirm or correct. "
                f"Re-call log_answer once corrected."
            )

        auto_count = sum(1 for r in global_state.responses if r.get("auto_extracted"))
        return (
            f"SUCCESS. Signal score: {global_state.signal_score}. "
            f"depth: {global_state.interview_depth}. "
            f"Contradictions: {len(global_state.contradictions)}. "
            f"Auto-extracted: {auto_count}. Proceed."
        )

    def verify_identity(patient_confirmed_name: str = "",
                        date_of_birth: str = "",
                        identity_match: str = "confirmed", **kwargs):
        print(f"[IDENTITY] {patient_confirmed_name} | {date_of_birth} | {identity_match}")
        if identity_match == "mismatch":
            global_state.probes["E_identity_signal"] = 0.0
        return f"Identity {identity_match}. Proceed accordingly."

    def schedule_callback(preferred_day: str = "",
                          preferred_time_window: str = "",
                          phone_confirmed: str = "confirmed", **kwargs):
        print(f"[CALLBACK] {preferred_day} {preferred_time_window}")
        global_state.transcript.append({
            "role": "system",
            "message": f"[CALLBACK SCHEDULED] {preferred_day}, {preferred_time_window}",
            "timestamp": time.time()
        })
        return "Callback scheduled. Proceed to end_call with outcome: rescheduled."

    def escalate_to_pharmacist(symptom_description: str = "",
                                urgency_level: str = "urgent",
                                patient_acknowledged_followup: str = "yes", **kwargs):
        print(f"[ESCALATION] {urgency_level}: {symptom_description}")
        global_state.probes["D_safety_compliance"] = 0.0
        global_state.signal_score += 8
        global_state.interview_depth = "deep_probe"
        global_state.transcript.append({
            "role": "system",
            "message": f"[ESCALATION] {urgency_level.upper()}: {symptom_description}",
            "timestamp": time.time()
        })
        return "Escalation flagged. End call immediately after safety messaging."

    def capture_pricing_concern(concern_description: str = "",
                                 patient_wants_followup: str = "yes", **kwargs):
        print(f"[PRICING] {concern_description}")
        accumulate_signal("pricing_question")
        return "Pricing concern logged. Provide billing contact and continue."

    def flag_contradiction(field_a: str = "", answer_a: str = "",
                            field_b: str = "", answer_b: str = "",
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
            global_state.probes["F_capture_accuracy"] = max(
                0.0, global_state.probes["F_capture_accuracy"] - 0.25
            )
        return "Contradiction logged. Probe gently before continuing."

    def end_call(outcome: str = "completed",
                 refill_approved: str = "yes",
                 summary: str = "", **kwargs):
        answered = sum(1 for r in global_state.responses if r["answer"])
        global_state.probes["C_coverage_integrity"] = (
            1.0 if answered >= 14 else round(answered / 14, 2)
        )
        trigger_auto_end(outcome)
        return "ended"

    def detect_ai_inquiry(patient_question: str = "",
                           response_given: str = "",
                           patient_accepted_continuation: str = "yes", **kwargs):
        print(f"[AI INQUIRY] Q: {patient_question} | Accepted: {patient_accepted_continuation}")
        global_state.ai_inquiry_logged = True
        global_state.transcript.append({
            "role": "system",
            "message": f"[AI INQUIRY] Patient asked: '{patient_question}'. "
                       f"Continuation accepted: {patient_accepted_continuation}",
            "timestamp": time.time()
        })
        if patient_accepted_continuation == "yes":
            accumulate_signal("behavioral_uncertainty")
        return (
            "AI inquiry logged. "
            + ("Continue with check-in." if patient_accepted_continuation == "yes"
               else "Patient declined. Call schedule_callback then end_call(outcome: rescheduled).")
        )

    def log_emotional_distress(distress_type: str = "",
                                patient_statement: str = "",
                                action_taken: str = "",
                                urgency: str = "low", **kwargs):
        print(f"[EMOTIONAL] {distress_type} | urgency={urgency}")
        event = {
            "distress_type":     distress_type,
            "patient_statement": patient_statement,
            "action_taken":      action_taken,
            "urgency":           urgency,
            "timestamp":         time.time()
        }
        global_state.emotional_events.append(event)
        global_state.transcript.append({
            "role": "system",
            "message": f"[EMOTIONAL DISTRESS] {distress_type.upper()}: {patient_statement}",
            "timestamp": time.time()
        })
        if urgency == "high":
            accumulate_signal("safety_concern")
            global_state.probes["G_behavioral_signal"] = 0.0
        elif urgency == "medium":
            accumulate_signal("behavioral_uncertainty")
        instructions = {
            "bereavement":              "Express condolences. Shorten call.",
            "psychological_distress":   "Flag care team. Call escalate_to_pharmacist(urgent).",
            "frustration_with_company": "Acknowledge. Do not argue. Continue if patient allows.",
            "emotional_distress":       "Pause. Wait for patient. Offer callback.",
            "crisis":                   "Provide 988 lifeline. Escalate emergency. End call.",
        }
        return f"Emotional event logged ({distress_type}, urgency={urgency}). " + \
               instructions.get(distress_type, "Acknowledge with empathy. Continue when patient is ready.")

    def handle_caregiver_proxy(relationship_to_patient: str = "",
                                caregiver_name: str = "",
                                patient_available: str = "no", **kwargs):
        print(f"[PROXY] {relationship_to_patient} | available={patient_available}")
        note = f"Call answered by {relationship_to_patient} ({caregiver_name}). Available: {patient_available}."
        global_state.caregiver_proxy = note
        global_state.probes["E_identity_signal"] = 0.5
        global_state.transcript.append({
            "role": "system",
            "message": f"[CAREGIVER PROXY] {note}",
            "timestamp": time.time()
        })
        if patient_available == "shortly":
            return "Caregiver proxy logged. Patient coming shortly. Re-attempt identity gate when on line."
        return "Caregiver proxy logged. Patient not available. Call schedule_callback, then end_call(rescheduled)."

    def request_human_transfer(reason: str = "", urgency: str = "normal", **kwargs):
        print(f"[HUMAN TRANSFER] reason={reason} urgency={urgency}")
        global_state.human_transfer_req = True
        global_state.transcript.append({
            "role": "system",
            "message": f"[HUMAN TRANSFER REQUESTED] Reason: {reason} | Urgency: {urgency}",
            "timestamp": time.time()
        })
        accumulate_signal("behavioral_uncertainty")
        if urgency == "urgent":
            return "URGENT transfer. Call escalate_to_pharmacist then end_call."
        return "Transfer logged. Call schedule_callback then end_call(rescheduled)."

    # ── Register all 11 tools ──
    tools = ClientTools()
    tools.register("log_answer",              log_answer)
    tools.register("verify_identity",         verify_identity)
    tools.register("schedule_callback",       schedule_callback)
    tools.register("escalate_to_pharmacist",  escalate_to_pharmacist)
    tools.register("capture_pricing_concern", capture_pricing_concern)
    tools.register("flag_contradiction",      flag_contradiction)
    tools.register("end_call",                end_call)
    tools.register("detect_ai_inquiry",       detect_ai_inquiry)
    tools.register("log_emotional_distress",  log_emotional_distress)
    tools.register("handle_caregiver_proxy",  handle_caregiver_proxy)
    tools.register("request_human_transfer",  request_human_transfer)

    def agent_cb(text):
        if text.strip():
            global_state.transcript.append({
                "role": "agent", "message": text, "timestamp": time.time()
            })
            lower = text.lower()
            if ("goodbye" in lower or
                    "everything i need" in lower or
                    "finish another time" in lower):
                trigger_auto_end("completed_via_transcript")

    def user_cb(text):
        if text.strip() and text != "...":
            global_state.transcript.append({
                "role": "user", "message": text, "timestamp": time.time()
            })

    convo = Conversation(
        client=client,
        agent_id=os.getenv("ELEVENLABS_AGENT_ID"),
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=agent_cb,
        callback_user_transcript=user_cb,
        client_tools=tools,
        config=ConversationInitiationData(
            dynamic_variables={
                "patient_name":    patient_name,
                "medication":      medication,
                "patient_context": patient_context
            }
        )
    )

    global_state.active_conversation = convo
    global_state.start_time = time.time()
    try:
        convo.start_session()
        convo.wait_for_session_end()
    except Exception as e:
        print("CONVERSATION ERROR:", e)
    global_state.call_duration = int(time.time() - global_state.start_time)


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return FileResponse(os.path.join(DEMO_DIR, "index.html"))


@app.get("/call-state")
def get_state():
    insights = global_state.insights or {}
    return {
        "responses":              global_state.responses,
        "outcome":                global_state.outcome,
        "transcript":             global_state.transcript,
        "insights":               global_state.insights,
        "call_duration":          global_state.call_duration,
        "probes":                 global_state.probes,
        "probe_labels":           PROBE_LABELS,
        "graph_data":             build_graph_data(),
        "agent_behavior_mode":    compute_behavior_mode(),
        "signal_score":           global_state.signal_score,
        "interview_depth":        global_state.interview_depth,
        "contradictions":         global_state.contradictions,
        "acoustic_flag":          global_state.acoustic_flag,
        "emotional_events":       global_state.emotional_events,
        "ai_inquiry_logged":      global_state.ai_inquiry_logged,
        "caregiver_proxy":        global_state.caregiver_proxy,
        "human_transfer_req":     global_state.human_transfer_req,
        "extracted_answers":      global_state.extracted_answers,
        # NEW v4 fields
        "glp1_risk_factors":      insights.get("glp1_risk_factors", []),
        "longitudinal_delta":     insights.get("longitudinal_delta"),
        "hesitation_fingerprint": global_state.hesitation_fingerprint,
        "mi_score":               insights.get("mi_score"),
        "mi_assessment":          insights.get("mi_assessment", ""),
        "previous_call_provided": bool(global_state.previous_call_context.strip()),
    }


class CallRequest(BaseModel):
    patient_name:          str
    medication:            str
    patient_context:       str
    previous_call_context: str = ""   # NEW v4 — for longitudinal delta


@app.post("/api/start-call")
def start(req: CallRequest, bg: BackgroundTasks):
    global global_state
    global_state = CallState()
    bg.add_task(
        run_call,
        req.patient_name,
        req.medication,
        req.patient_context,
        req.previous_call_context,
    )
    return {"status": "started"}


@app.post("/api/end-call")
def end():
    if not global_state.outcome:
        global_state.outcome = "human_intercepted"
        if global_state.active_conversation:
            try:
                global_state.active_conversation.end_session()
            except Exception:
                pass

        def run_llm_task():
            try:
                mi_result  = compute_mi_fidelity(global_state.transcript)
                global_state.probes["H_mi_fidelity"] = mi_result["score"]
                hesitation = analyze_hesitation(global_state.transcript)
                global_state.hesitation_fingerprint = hesitation
                global_state.insights = run_final_llm_deliberation(
                    transcript=global_state.transcript,
                    responses=global_state.responses,
                    previous_call_context=global_state.previous_call_context,
                    mi_metrics=mi_result["metrics"],
                    hesitation_data=hesitation,
                )
            except Exception as e:
                print("LLM ERROR:", e)

        threading.Thread(target=run_llm_task).start()
    return {"status": "ended"}