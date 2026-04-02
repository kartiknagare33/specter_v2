# server/main.py
# SPECTER v5 — Clinical Intelligence Voice Agent Backend
#
# v4 innovations (kept):
#   1. GLP-1 Churn Risk Matrix
#   2. Longitudinal Risk Delta
#   3. MI Fidelity Score — Probe H
#   4. Hesitation Fingerprint
#
# v5 NEW innovations:
#   5. Living Memory — shared structured brain across all Workflow agents
#   6. Pre-Call AI Briefing — Groq reads patient history, generates today's strategy
#   7. Ghost Analyst — async parallel Groq commentary on every tool call
#   8. Closed Loop — Round 5 deliberation writes next-call strategy to patient DB
#   9. Patient SQLite DB — longitudinal patient tracking across calls
#  10. Dynamic Prompt Injection — Jessica's ElevenLabs prompt overridden per-call
#      with patient-specific strategy. Permanent agent config never touched.


import os
import time
import json
import threading
import asyncio
from datetime import date

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation, ClientTools, ConversationInitiationData
)
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from deliberation.judge import run_deliberation, run_final_llm_deliberation
from db.patient_store import init_db, get_patient, save_call_summary, update_next_strategy, upsert_patient

load_dotenv()

app      = FastAPI(title="TrimRX SPECTER Backend v5")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init patient DB on startup
init_db()


# ─────────────────────────────────────────────────────────────
# 14 PRIMARY QUESTIONS + 5 HIDDEN VALIDATION PROBES = 19 TOTAL
# ─────────────────────────────────────────────────────────────
QUESTIONS = [
    # Primary 14
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

PRIMARY_TOPIC_KEYS = [
    "overall_feeling", "current_weight", "weight_lost", "missed_doses",
    "side_effects", "satisfaction", "new_medications", "new_allergies",
    "surgeries", "energy_levels", "diet_adherence", "treatment_concerns",
    "doctor_questions", "address_change",
]

PROBE_LABELS = {
    "A_acoustic_fidelity":  "Acoustic",
    "B_intent_alignment":   "Intent",
    "C_coverage_integrity": "Coverage",
    "D_safety_compliance":  "Safety",
    "E_identity_signal":    "Identity",
    "F_capture_accuracy":   "Capture",
    "G_behavioral_signal":  "Behavioral",
    "H_mi_fidelity":        "MI Fidelity",
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

FLAG_KEYWORD_MAP = {
    "pricing":      ["expensive", "afford", "cost", "price", "insurance", "billing", "money", "can't keep"],
    "nausea":       ["nausea", "nauseous", "vomiting", "sick", "queasy"],
    "safety":       ["chest pain", "shortness of breath", "emergency", "hospital", "severe", "allergic reaction", "pancreatitis"],
    "reschedule":   ["call back", "driving", "busy", "bad time", "reschedule", "call me later", "call tomorrow"],
    "missed_doses": ["missed", "forgot", "skipped", "didn't take"],
    "distress":     ["depressed", "anxious", "suicidal", "hopeless", "crying", "can't cope"],
}

EMOTIONAL_KEYWORDS = {
    "distressed": ["depressed", "anxious", "suicidal", "hopeless", "crying", "can't cope",
                   "struggling", "really bad", "terrible", "awful"],
    "concerned":  ["worried", "concerned", "not sure", "unsure", "thinking about stopping",
                   "scared", "nervous", "not happy"],
    "engaged":    ["great", "wonderful", "amazing", "really good", "doing well", "happy",
                   "excited", "pleased", "love it"],
}


# ─────────────────────────────────────────────────────────────
# JESSICA BASE PROMPT  (v5 — NEW)
# Stored in Python so we can prepend patient-specific context
# and inject it into ElevenLabs per-call via conversation_config_override.
# The permanent ElevenLabs agent config is never modified.
# ─────────────────────────────────────────────────────────────
JESSICA_BASE_PROMPT = """You are Jessica, a compassionate and professional clinical care coordinator
for TrimRX, a GLP-1 weight-loss medication company. You are conducting a
routine medication check-in call with a patient on Semaglutide.

=== LIVING MEMORY PROTOCOL ===

RULE 0 — MEMORY CHECK: Before formulating ANY question, call get_memory_state().
Read questions_remaining to know exactly what hasn't been asked.
Read questions_answered — NEVER re-ask these.
Read emotional_state — calibrate your tone to match.
Read handoff_required — if true, do NOT ask another question.
Read recommended_next_action and follow it exactly.

RULE 0b — HANDOFF PROTOCOL:
  If handoff_required is true:
  - handoff_target "billing": acknowledge the concern warmly, say "Let me
    connect you with Maya from our billing team who can help with that
    directly." Call capture_pricing_concern(), then stop speaking.
  - handoff_target "pharmacist": say "I want to make sure we address this
    properly. Let me get Dr. Patel, our clinical pharmacist."
    Call escalate_to_pharmacist(), then stop speaking.
  - handoff_target "scheduling": say "Absolutely, no problem at all."
    Call schedule_callback(), then stop speaking.

RULE 0c — QUESTION ORDER: Work through questions_remaining in the order
the memory returns them. If a key is in questions_answered, skip it entirely.

RULE 0d — COVERAGE LOCK: Do NOT call end_call(completed) if
questions_remaining has more than 0 items, unless the patient has
explicitly refused further questions or opted out.

=== CLINICAL RULES ===

RULE 1 — IDENTITY: The patient's identity has already been verified.
Do not re-verify. Greet them by name warmly and proceed directly.

RULE 2 — 14 QUESTIONS: You must ask all 14 primary clinical questions.
Call log_answer() after EVERY confirmed patient response. No exceptions.

RULE 3 — NATURAL FLOW: Ask one question at a time. Acknowledge what the
patient said before moving to the next question. Never sound like a checklist.

RULE 4 — EMPATHY: When a patient shares something difficult (pain, side
effects, financial stress, emotional distress), acknowledge it genuinely
before continuing. Example: "I'm really sorry to hear that — thank you
for letting me know."

RULE 5 — NO MEDICAL ADVICE: If asked about dosage changes, stopping
medication, or drug interactions, say: "That's a great question for
Dr. Patel on our clinical team — I'll make sure it gets flagged for them."

RULE 6 — CONTRADICTION HANDLING: If two answers seem inconsistent, call
flag_contradiction() and probe gently: "Just to make sure I have this
right..."

RULE 7 — MULTI-EXTRACTION: When a patient volunteers information that
answers an unasked question, call log_answer() for that topic immediately.

RULE 8 — AI DISCLOSURE: If asked if you are an AI or a robot, call
detect_ai_inquiry(). Say: "I'm a care coordinator assistant — would you
like to continue your check-in?" Do not deny being AI.

RULE 9 — EMOTIONAL DISTRESS: If the patient shows signs of distress,
call log_emotional_distress() with appropriate urgency level.

RULE 10 — CLOSING: Once questions_remaining is empty, say:
"That covers everything I needed today. Your care team will review this
and be in touch if anything needs follow-up. Take care, {{patient_name}}."
Then call end_call(outcome: completed).

=== CALL FLOW ===

STEP 1 — WARM OPENING: "Hi {{patient_name}}, this is Jessica from TrimRX.
I'm calling for your monthly medication check-in — is now a good time?"

STEP 2 — MEMORY CHECK: Call get_memory_state() to load clinical context
before asking anything.

STEP 3 — 14 QUESTIONS: Work through questions_remaining. After each
answer, call log_answer() before moving on.

STEP 4 — EDGE CASES: Trust the routing. When handoff_required fires,
transfer immediately. Do not attempt to handle pricing or safety yourself.

STEP 5 — CLOSE: Confirm refill, thank patient, call end_call(completed).

You are warm, professional, clinically competent, and genuinely care
about each patient. You never rush. You never skip questions."""


# ─────────────────────────────────────────────────────────────
# CALL STATE
# ─────────────────────────────────────────────────────────────
class CallState:
    def __init__(self):
        self.responses = [
            {"question": q, "answer": "", "flag": "none", "auto_extracted": False}
            for q in QUESTIONS
        ]
        self.outcome               = None
        self.transcript            = []
        self.insights              = None
        self.active_conversation   = None
        self.start_time            = None
        self.call_duration         = 0
        self.behavior_mode         = "STANDBY"
        self._prev_g_score         = 1.0
        self.signal_score          = 0
        self.contradictions        = []
        self.interview_depth       = "standard"
        self.acoustic_flag         = None
        self.ai_inquiry_logged     = False
        self.emotional_events      = []
        self.caregiver_proxy       = None
        self.human_transfer_req    = False
        self.extracted_answers     = []
        self.previous_call_context = ""
        self.hesitation_fingerprint = None
        self.patient_id            = ""
        self.patient_name          = ""

        # ── PROBE SCORES ──
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

        # ── LIVING MEMORY (v5) ──
        self.living_memory = {
            "questions_remaining":     list(PRIMARY_TOPIC_KEYS),
            "questions_answered":      [],
            "emotional_state":         "neutral",
            "detected_flags":          [],
            "patient_risk_level":      "LOW",
            "recommended_next_action": "continue_interview",
            "handoff_required":        False,
            "handoff_target":          None,
            "ghost_alerts":            [],
            "pre_call_brief":          "",
            "coverage_pct":            0.0,
            "call_number":             1,
        }


global_state = CallState()


# ─────────────────────────────────────────────────────────────
# LIVING MEMORY HELPERS
# ─────────────────────────────────────────────────────────────
def update_living_memory(topic_key: str, patient_answer: str, clinical_flag: str):
    lm        = global_state.living_memory
    ans_lower = patient_answer.lower()

    if topic_key in lm["questions_remaining"]:
        lm["questions_remaining"].remove(topic_key)
    if topic_key not in lm["questions_answered"]:
        lm["questions_answered"].append(topic_key)

    lm["coverage_pct"] = round(
        len(lm["questions_answered"]) / len(PRIMARY_TOPIC_KEYS), 2
    )

    for flag, keywords in FLAG_KEYWORD_MAP.items():
        if any(kw in ans_lower for kw in keywords):
            if flag not in lm["detected_flags"]:
                lm["detected_flags"].append(flag)

    if clinical_flag in ("safety_concern", "side_effect_severe"):
        if "safety" not in lm["detected_flags"]:
            lm["detected_flags"].append("safety")
    if clinical_flag == "pricing_question":
        if "pricing" not in lm["detected_flags"]:
            lm["detected_flags"].append("pricing")

    for state_label, keywords in EMOTIONAL_KEYWORDS.items():
        if any(kw in ans_lower for kw in keywords):
            lm["emotional_state"] = state_label
            break

    avg_probe = sum(global_state.probes.values()) / len(global_state.probes)
    if avg_probe < 0.5 or global_state.signal_score > 15:
        lm["patient_risk_level"] = "HIGH"
    elif avg_probe < 0.75 or global_state.signal_score > 7:
        lm["patient_risk_level"] = "MEDIUM"
    else:
        lm["patient_risk_level"] = "LOW"

    flags = lm["detected_flags"]
    es    = lm["emotional_state"]
    rem   = lm["questions_remaining"]

    if "safety" in flags:
        lm["recommended_next_action"] = "handoff_pharmacist"
        lm["handoff_required"]        = True
        lm["handoff_target"]          = "pharmacist"
    elif "pricing" in flags:
        lm["recommended_next_action"] = "handoff_billing"
        lm["handoff_required"]        = True
        lm["handoff_target"]          = "billing"
    elif "reschedule" in flags:
        lm["recommended_next_action"] = "handoff_scheduling"
        lm["handoff_required"]        = True
        lm["handoff_target"]          = "scheduling"
    elif es == "distressed":
        lm["recommended_next_action"] = "empathy_probe"
        lm["handoff_required"]        = False
        lm["handoff_target"]          = None
    elif len(rem) == 0:
        lm["recommended_next_action"] = "close_call"
        lm["handoff_required"]        = False
        lm["handoff_target"]          = None
    else:
        lm["recommended_next_action"] = "continue_interview"
        lm["handoff_required"]        = False
        lm["handoff_target"]          = None


def run_ghost_analyst(topic_key: str, patient_answer: str):
    try:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        recent_tx = global_state.transcript[-6:] if len(global_state.transcript) >= 6 \
                    else global_state.transcript
        recent_text = "\n".join(
            f"{t['role'].upper()}: {t['message']}"
            for t in recent_tx if t.get("message")
        )

        mem_summary = {
            "questions_remaining_count": len(global_state.living_memory["questions_remaining"]),
            "emotional_state":           global_state.living_memory["emotional_state"],
            "detected_flags":            global_state.living_memory["detected_flags"],
            "patient_risk_level":        global_state.living_memory["patient_risk_level"],
            "recommended_next":          global_state.living_memory["recommended_next_action"],
            "coverage_pct":              global_state.living_memory["coverage_pct"],
        }

        prompt = f"""You are a senior clinical analyst watching a GLP-1 medication check-in call live.

Recent transcript:
{recent_text}

Current clinical state:
{json.dumps(mem_summary, indent=2)}

Last answer logged: [{topic_key}] "{patient_answer}"

In ONE sentence only, state the single most important clinical observation right now.
Be specific. Focus on risk, behaviour, or what Jessica should do next.
Do not start with "I". Do not include any preamble."""

        r = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
        )
        observation = r.choices[0].message.content.strip()

        global_state.living_memory["ghost_alerts"].append({
            "timestamp":   round(time.time(), 2),
            "observation": observation,
            "topic":       topic_key,
            "risk":        global_state.living_memory["patient_risk_level"],
        })
        print(f"[GHOST] {observation}")

    except Exception as e:
        print(f"[GHOST ERROR] {e}")


# ─────────────────────────────────────────────────────────────
# PROBE SCORING
# ─────────────────────────────────────────────────────────────
def score_probes_on_answer(patient_answer: str, clinical_flag: str,
                           question_topic: str) -> str | None:
    answer_lower  = patient_answer.lower()
    acoustic_issue = None

    if question_topic in ("current_weight", "weight_lost"):
        try:
            val = float("".join(c for c in patient_answer if c.isdigit() or c == "."))
            if question_topic == "current_weight" and (val < 50 or val > 500):
                acoustic_issue = f"Weight '{patient_answer}' is physiologically implausible."
                global_state.probes["A_acoustic_fidelity"] = 0.0
            elif question_topic == "weight_lost" and val > 100:
                acoustic_issue = f"Weight loss '{patient_answer}' lbs in one month is implausible."
                global_state.probes["A_acoustic_fidelity"] = 0.0
            else:
                global_state.probes["A_acoustic_fidelity"] = 1.0
                global_state.acoustic_flag = None
        except ValueError:
            global_state.probes["A_acoustic_fidelity"] = 1.0

    opt_out_triggers = ["stop calling", "remove me", "shut up", "don't call me",
                        "wrong number", "who is this"]
    if any(t in answer_lower for t in opt_out_triggers):
        global_state.probes["B_intent_alignment"] = 0.0
    else:
        global_state.probes["B_intent_alignment"] = max(
            0.5, global_state.probes["B_intent_alignment"]
        )

    safety_triggers = ["should i take", "what should i do", "is it safe to",
                       "can i mix", "what is the dosage"]
    if any(t in answer_lower for t in safety_triggers):
        global_state.probes["D_safety_compliance"] = 0.0
    else:
        global_state.probes["D_safety_compliance"] = max(
            0.5, global_state.probes["D_safety_compliance"]
        )

    if clinical_flag == "behavioral_uncertainty":
        global_state.probes["G_behavioral_signal"] = 0.0
    elif clinical_flag == "none" and global_state.probes["G_behavioral_signal"] == 0.0:
        global_state.probes["G_behavioral_signal"] = 1.0

    return acoustic_issue


def accumulate_signal(clinical_flag: str):
    weight = SIGNAL_WEIGHTS.get(clinical_flag, 0)
    global_state.signal_score += weight
    if global_state.signal_score >= 15:
        global_state.interview_depth = "deep_probe"
    elif global_state.signal_score >= 7:
        global_state.interview_depth = "soft_probe"
    else:
        global_state.interview_depth = "standard"


def extract_multi_answers(patient_answer: str, current_topic: str):
    answer_lower = patient_answer.lower()
    for topic, hints in EXTRACTION_HINTS.items():
        if topic == current_topic:
            continue
        idx = TOPIC_TO_INDEX.get(topic)
        if idx is None:
            continue
        if global_state.responses[idx]["answer"]:
            continue
        for hint in hints:
            if hint in answer_lower:
                start   = max(0, answer_lower.find(hint) - 20)
                end     = min(len(patient_answer), answer_lower.find(hint) + 60)
                snippet = patient_answer[start:end].strip()
                global_state.responses[idx]["answer"]        = snippet
                global_state.responses[idx]["auto_extracted"] = True
                global_state.extracted_answers.append({
                    "topic":   topic,
                    "snippet": snippet,
                    "source":  current_topic,
                })
                break


def check_cross_validation():
    checks = [
        ("missed_doses",    "supply_days",
         "Patient said no missed doses but supply is running very low"),
        ("satisfaction",    "weight_lost",
         "Patient expressed dissatisfaction but reported significant weight loss"),
        ("side_effects",    "satisfaction",
         "Patient reported side effects but also said they are fully satisfied"),
        ("new_medications", "pharmacy_recent",
         "Denied new medications but pharmacy filled new prescription"),
        ("missed_doses",    "treatment_concerns",
         "Reported missed doses and also has treatment concerns — churn risk elevated"),
    ]
    for key_a, key_b, summary in checks:
        idx_a = TOPIC_TO_INDEX.get(key_a)
        idx_b = TOPIC_TO_INDEX.get(key_b)
        if idx_a is None or idx_b is None:
            continue
        ans_a = global_state.responses[idx_a]["answer"]
        ans_b = global_state.responses[idx_b]["answer"]
        if not ans_a or not ans_b:
            continue
        c = {"field_a": key_a, "answer_a": ans_a,
             "field_b": key_b, "answer_b": ans_b, "summary": summary}
        if c not in global_state.contradictions:
            global_state.contradictions.append(c)
            accumulate_signal("contradiction")
            global_state.probes["F_capture_accuracy"] = max(
                0.0, global_state.probes["F_capture_accuracy"] - 0.2
            )


def compute_behavior_mode() -> str:
    lm = global_state.living_memory
    if global_state.outcome:
        return "CALL COMPLETE"
    if lm["handoff_required"]:
        target = lm["handoff_target"] or "specialist"
        return f"HANDOFF → {target.upper()}"
    if lm["emotional_state"] == "distressed":
        return "DISTRESS DETECTED"
    if "safety" in lm["detected_flags"]:
        return "SAFETY FLAG — ESCALATING"
    if global_state.interview_depth == "deep_probe":
        return "DEEP PROBE ACTIVE"
    if global_state.interview_depth == "soft_probe":
        return "SOFT PROBE ACTIVE"
    if lm["coverage_pct"] > 0:
        return f"INTERVIEW IN PROGRESS — {int(lm['coverage_pct']*100)}% COVERED"
    return "AWAITING RESPONSE"


def compute_mi_fidelity(transcript: list) -> dict:
    agent_turns = [t for t in transcript if t.get("role") == "agent"]
    user_turns  = [t for t in transcript if t.get("role") == "user"]
    if not agent_turns:
        return {"r_q_ratio": 0, "open_q_ratio": 0, "empathy_events": 0, "mi_score": 50}

    open_markers    = ["how", "what", "tell me", "describe", "can you share", "help me understand"]
    empathy_markers = ["i understand", "that makes sense", "i hear you", "i'm sorry",
                       "that sounds", "thank you for sharing"]

    open_count    = sum(1 for t in agent_turns
                        if any(m in t.get("message","").lower() for m in open_markers))
    empathy_count = sum(1 for t in agent_turns
                        if any(m in t.get("message","").lower() for m in empathy_markers))

    r_q_ratio    = len(user_turns) / max(1, len(agent_turns))
    open_q_ratio = open_count / max(1, len(agent_turns))
    mi_score     = min(100, int((r_q_ratio * 20) + (open_q_ratio * 50) + (empathy_count * 5)))
    global_state.probes["H_mi_fidelity"] = round(mi_score / 100, 2)
    return {
        "r_q_ratio":      round(r_q_ratio, 2),
        "open_q_ratio":   round(open_q_ratio, 2),
        "empathy_events": empathy_count,
        "mi_score":       mi_score,
    }


def analyze_hesitation(transcript: list) -> dict:
    user_turns = [t for t in transcript if t.get("role") == "user" and t.get("message")]
    if not user_turns:
        return {"hesitation_index": 0.0, "level": "LOW", "notable_patterns": []}

    hedge_words = ["um", "uh", "well", "i think", "i guess", "maybe", "kind of",
                   "sort of", "i'm not sure", "i don't know", "probably"]
    hedge_count = sum(
        sum(1 for hw in hedge_words if hw in t["message"].lower())
        for t in user_turns
    )
    short_open = sum(1 for t in user_turns if len(t["message"].split()) < 5)
    avg_len    = sum(len(t["message"].split()) for t in user_turns) / len(user_turns)
    raw_index  = min(1.0, (hedge_count * 0.05) + (short_open * 0.07) + (max(0, 15 - avg_len) * 0.02))
    level      = "HIGH" if raw_index > 0.6 else "ELEVATED" if raw_index > 0.3 else "LOW"

    patterns = []
    if hedge_count > 3:
        patterns.append(f"High hedge word frequency ({hedge_count} instances)")
    if short_open > 2:
        patterns.append(f"Unusually short responses ({short_open} turns under 5 words)")
    if avg_len < 8:
        patterns.append(f"Low average response length ({avg_len:.1f} words)")

    return {
        "hesitation_index":  round(raw_index, 2),
        "level":             level,
        "hedge_count":       hedge_count,
        "short_responses":   short_open,
        "avg_response_len":  round(avg_len, 1),
        "notable_patterns":  patterns,
    }


def build_graph_data() -> list:
    nodes    = []
    edges    = []
    answered = [r for r in global_state.responses if r["answer"]]
    for i, r in enumerate(answered):
        nodes.append({"id": i, "label": r["question"][:30], "answer": r["answer"][:40]})
        if i > 0:
            edges.append({"source": i - 1, "target": i})
    return {"nodes": nodes, "edges": edges}


# ─────────────────────────────────────────────────────────────
# PRE-CALL BRIEFING + DYNAMIC PROMPT INJECTION  (v5 — UPDATED)
# ─────────────────────────────────────────────────────────────
def generate_pre_call_brief(patient: dict) -> dict:
    """
    Calls Groq to generate a patient-specific clinical strategy.
    Returns:
      strategy_summary       — shown on dashboard pre-call brief card
      agent_instruction      — 2-sentence patient-specific note for Jessica
      custom_system_prompt   — full Jessica prompt with patient context
                               prepended. Injected into ElevenLabs via
                               conversation_config_override for this call only.
      risk_prediction        — LOW / MEDIUM / HIGH
      priority_focus         — single most important topic today
      sensitivities_reminder — comma-separated sensitive topics
    """
    try:
        groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        history_text  = json.dumps(patient.get("call_history", []), indent=2)
        sensitivities = ", ".join(patient.get("known_sensitivities", [])) or "none noted"
        last_strategy = patient.get("next_call_strategy", "") or "No prior strategy."
        call_count    = len(patient.get("call_history", [])) + 1
        risk_trend    = patient.get("risk_trend", [])

        trend_text = ""
        if len(risk_trend) >= 2:
            last_two  = risk_trend[-2:]
            direction = "rising" if last_two[-1]["churn"] > last_two[0]["churn"] else "falling"
            trend_text = (
                f"Churn risk trend: {direction} "
                f"({last_two[0]['churn']}% → {last_two[-1]['churn']}%)"
            )

        prompt = f"""You are a senior clinical strategist at TrimRX preparing for call #{call_count}
with patient: {patient['name']} (ID: {patient['patient_id']}).

CALL HISTORY:
{history_text}

KNOWN SENSITIVITIES: {sensitivities}
{trend_text}
STRATEGY FROM LAST CALL: {last_strategy}

Generate a pre-call brief. Return ONLY valid JSON:
{{
  "strategy_summary": "4-line brief: (1) emotional approach, (2) first topic priority, (3) sensitive topics, (4) specialist standby",
  "risk_prediction": "LOW or MEDIUM or HIGH",
  "priority_focus": "single most important question topic key to address today",
  "agent_instruction": "2-sentence instruction for Jessica about THIS patient — what to know before saying hello",
  "sensitivities_reminder": "comma-separated list of topics to approach carefully",
  "opening_tone": "warm_and_checking_in or empathetic_about_concerns or encouraging_progress",
  "watch_flags": ["flag1", "flag2"]
}}"""

        r = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        brief = json.loads(r.choices[0].message.content)

        # ── BUILD CUSTOM SYSTEM PROMPT ──────────────────────────────
        # Patient-specific block prepended to JESSICA_BASE_PROMPT.
        # This is what gets injected into ElevenLabs per-call.
        patient_block = f"""=== PATIENT-SPECIFIC STRATEGY FOR THIS CALL ===
Patient: {patient['name']} | Call #{call_count} | Predicted Risk: {brief.get('risk_prediction', 'MEDIUM')}

CLINICAL BRIEFING (read before speaking):
{brief.get('strategy_summary', 'Standard check-in.')}

JESSICA — YOUR INSTRUCTION FOR THIS PATIENT:
{brief.get('agent_instruction', 'Proceed with standard clinical interview.')}

SENSITIVE TOPICS THIS CALL: {brief.get('sensitivities_reminder', 'None flagged')}
OPENING TONE:               {brief.get('opening_tone', 'warm_and_checking_in')}
PRIORITY FOCUS TODAY:       {brief.get('priority_focus', 'overall_feeling')}
FLAGS TO WATCH:             {', '.join(brief.get('watch_flags', [])) or 'None'}

PRIOR CALL STRATEGY:
{last_strategy}
=== END PATIENT STRATEGY — STANDARD RULES FOLLOW ===

"""
        brief["custom_system_prompt"] = patient_block + JESSICA_BASE_PROMPT

        print(f"[PRE-CALL BRIEF] {patient['patient_id']}: "
              f"risk={brief.get('risk_prediction')} | "
              f"focus={brief.get('priority_focus')} | "
              f"prompt_len={len(brief['custom_system_prompt'])} chars")
        return brief

    except Exception as e:
        print(f"[PRE-CALL BRIEF ERROR] {e}")
        # Fallback — base prompt unmodified
        return {
            "strategy_summary":       "Standard check-in. No prior history available.",
            "risk_prediction":        "MEDIUM",
            "priority_focus":         "overall_feeling",
            "agent_instruction":      "Proceed with standard clinical interview.",
            "sensitivities_reminder": "",
            "opening_tone":           "warm_and_checking_in",
            "watch_flags":            [],
            "custom_system_prompt":   JESSICA_BASE_PROMPT,
        }


# ─────────────────────────────────────────────────────────────
# POST-CALL PIPELINE
# ─────────────────────────────────────────────────────────────
def trigger_auto_end(outcome: str = "completed"):
    global_state.outcome = outcome

    def run_llm_task():
        try:
            mi_metrics = compute_mi_fidelity(global_state.transcript)
            hesitation = analyze_hesitation(global_state.transcript)
            global_state.hesitation_fingerprint = hesitation

            answered = sum(1 for r in global_state.responses if r["answer"])
            global_state.probes["C_coverage_integrity"] = (
                1.0 if answered >= 14 else round(answered / 14, 2)
            )

            global_state.insights = run_final_llm_deliberation(
                transcript=global_state.transcript,
                responses=global_state.responses,
                previous_call_context=global_state.previous_call_context,
                mi_metrics=mi_metrics,
                hesitation_data=hesitation,
                patient_id=global_state.patient_id,
            )

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
                ex_text  = f"\n\n[MULTI-ANSWER EXTRACTION]\n"
                ex_text += f"{len(global_state.extracted_answers)} extra data point(s):\n"
                for ex in global_state.extracted_answers:
                    ex_text += f"  - {ex['topic']}: '{ex['snippet']}'\n"
                global_state.insights["edge_case_report"] += ex_text
            if global_state.hesitation_fingerprint:
                h = global_state.hesitation_fingerprint
                global_state.insights["edge_case_report"] += (
                    f"\n\n[HESITATION FINGERPRINT]\n"
                    f"Index: {h.get('hesitation_index')} ({h.get('level')})\n"
                    + "\n".join(f"  - {p}" for p in h.get("notable_patterns", []))
                )

            global_state.living_memory["ghost_alerts"].append({
                "timestamp":   round(time.time(), 2),
                "observation": (
                    f"Call complete. Deliberation running. "
                    f"Final churn: {global_state.insights.get('churn_score','?')}%. "
                    f"Risk: {global_state.insights.get('risk_level','?')}."
                ),
                "topic": "post_call",
                "risk":  global_state.insights.get("risk_level", "MEDIUM"),
            })

            if global_state.patient_id:
                summary = {
                    "date":        str(date.today()),
                    "churn_score": global_state.insights.get("churn_score", 0),
                    "risk_level":  global_state.insights.get("risk_level", "UNKNOWN"),
                    "key_flags":   list(global_state.living_memory["detected_flags"]),
                    "outcome":     outcome,
                }
                save_call_summary(global_state.patient_id, summary)

                next_strategy = global_state.insights.get("next_call_strategy", "")
                brief_used    = global_state.living_memory.get("pre_call_brief", "")
                if next_strategy:
                    update_next_strategy(global_state.patient_id, next_strategy, brief_used)

        except Exception as e:
            print("LLM ERROR:", e)

    threading.Thread(target=run_llm_task, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# MAIN CALL RUNNER
# ─────────────────────────────────────────────────────────────
def run_call(patient_name: str, medication: str, patient_context: str,
             previous_call_context: str = "", patient_id: str = ""):

    global_state.previous_call_context = previous_call_context
    global_state.patient_id            = patient_id
    global_state.patient_name          = patient_name

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    # ── ALL 12 TOOLS ──
    def log_answer(question_topic: str = "", patient_answer: str = "",
                   confidence: str = "clear", clinical_flag: str = "none", **kwargs):
        print(f"[LOG] {question_topic} -> {patient_answer} [{clinical_flag}]")

        for item in global_state.responses:
            if (question_topic.lower() in item["question"].lower() or
                    item["question"].lower() in question_topic.lower()):
                item["answer"]         = patient_answer
                item["flag"]           = clinical_flag
                item["auto_extracted"] = False
                break

        idx = TOPIC_TO_INDEX.get(question_topic)
        if idx is not None and idx < len(global_state.responses):
            if not global_state.responses[idx]["answer"] or \
                    global_state.responses[idx]["auto_extracted"]:
                global_state.responses[idx]["answer"]         = patient_answer
                global_state.responses[idx]["flag"]           = clinical_flag
                global_state.responses[idx]["auto_extracted"] = False

        extract_multi_answers(patient_answer, question_topic)
        acoustic_issue = score_probes_on_answer(patient_answer, clinical_flag, question_topic)
        accumulate_signal(clinical_flag)
        update_living_memory(question_topic, patient_answer, clinical_flag)

        threading.Thread(
            target=run_ghost_analyst,
            args=(question_topic, patient_answer),
            daemon=True
        ).start()

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
            global_state.acoustic_flag = acoustic_issue
            return (
                f"ACOUSTIC_FLAG: {acoustic_issue} "
                f"DO NOT proceed. Ask patient to confirm or correct. "
                f"Re-call log_answer once corrected."
            )

        lm         = global_state.living_memory
        auto_count = sum(1 for r in global_state.responses if r.get("auto_extracted"))
        return (
            f"SUCCESS. Signal score: {global_state.signal_score}. "
            f"depth: {global_state.interview_depth}. "
            f"Questions remaining: {len(lm['questions_remaining'])}. "
            f"Contradictions: {len(global_state.contradictions)}. "
            f"Auto-extracted: {auto_count}. "
            f"Handoff required: {lm['handoff_required']}. "
            f"Recommended next: {lm['recommended_next_action']}. Proceed."
        )

    def get_memory_state(**kwargs) -> str:
        lm = global_state.living_memory
        return json.dumps({
            "questions_remaining":     lm["questions_remaining"],
            "questions_answered":      lm["questions_answered"],
            "emotional_state":         lm["emotional_state"],
            "detected_flags":          lm["detected_flags"],
            "patient_risk_level":      lm["patient_risk_level"],
            "recommended_next_action": lm["recommended_next_action"],
            "handoff_required":        lm["handoff_required"],
            "handoff_target":          lm["handoff_target"],
            "coverage_pct":            lm["coverage_pct"],
            "pre_call_brief":          lm["pre_call_brief"],
            "contradictions_count":    len(global_state.contradictions),
            "signal_score":            global_state.signal_score,
            "interview_depth":         global_state.interview_depth,
        })

    def verify_identity(patient_confirmed_name: str = "",
                        date_of_birth: str = "",
                        identity_match: str = "confirmed", **kwargs):
        print(f"[IDENTITY] {patient_confirmed_name} | {date_of_birth} | {identity_match}")
        if identity_match == "mismatch":
            global_state.probes["E_identity_signal"] = 0.0
            global_state.living_memory["detected_flags"].append("wrong_number")
        else:
            global_state.living_memory["ghost_alerts"].append({
                "timestamp":   round(time.time(), 2),
                "observation": f"Identity verified: {patient_confirmed_name}. Clinical interview can begin.",
                "topic":       "identity",
                "risk":        "LOW",
            })
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
        if "safety" not in global_state.living_memory["detected_flags"]:
            global_state.living_memory["detected_flags"].append("safety")
        global_state.living_memory["handoff_required"]        = True
        global_state.living_memory["handoff_target"]          = "pharmacist"
        global_state.living_memory["recommended_next_action"] = "handoff_pharmacist"
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
        if "pricing" not in global_state.living_memory["detected_flags"]:
            global_state.living_memory["detected_flags"].append("pricing")
        global_state.living_memory["handoff_required"]        = True
        global_state.living_memory["handoff_target"]          = "billing"
        global_state.living_memory["recommended_next_action"] = "handoff_billing"
        return "Pricing concern logged. Billing specialist standby activated. Provide billing@trimrx.com and continue."

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
        answered  = sum(1 for r in global_state.responses if r["answer"])
        global_state.probes["C_coverage_integrity"] = (
            1.0 if answered >= 14 else round(answered / 14, 2)
        )
        remaining = global_state.living_memory["questions_remaining"]
        if len(remaining) > 0 and outcome == "completed":
            print(f"[COVERAGE LOCK WARNING] {len(remaining)} questions unanswered: {remaining}")
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
        global_state.living_memory["emotional_state"] = "distressed"
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
        return (
            f"Emotional event logged ({distress_type}, urgency={urgency}). "
            + instructions.get(distress_type, "Acknowledge with empathy. Continue when ready.")
        )

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

    # ── Register all 12 tools ──
    tools = ClientTools()
    tools.register("log_answer",              log_answer)
    tools.register("get_memory_state",        get_memory_state)
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

    # ── DYNAMIC PROMPT INJECTION (v5 — NEW) ──────────────────
    # Build patient-specific prompt. Inject via conversation_config_override.
    # ElevenLabs applies this for THIS call only — permanent config untouched.
    custom_prompt  = None
    brief_summary  = "Standard check-in."
    dynamic_vars   = {
        "patient_name":    patient_name,
        "medication":      medication,
        "patient_context": patient_context,
        "pre_call_brief":  brief_summary,   # fallback dynamic variable
    }

    if patient_id:
        patient = get_patient(patient_id)
        if patient:
            brief         = generate_pre_call_brief(patient)
            custom_prompt = brief.get("custom_system_prompt")
            brief_summary = brief.get("strategy_summary", "Standard check-in.")

            # Seed Living Memory with brief
            global_state.living_memory["pre_call_brief"]     = brief_summary
            global_state.living_memory["patient_risk_level"] = brief.get("risk_prediction", "MEDIUM")
            history = patient.get("call_history", [])
            global_state.living_memory["call_number"] = len(history) + 1

            # Update dynamic variable with real brief
            dynamic_vars["pre_call_brief"] = brief_summary

            # Auto-populate previous_call_context if not already set
            if not global_state.previous_call_context and history:
                last = history[-1]
                global_state.previous_call_context = (
                    f"Last call ({last.get('date','')}): "
                    f"churn={last.get('churn_score','?')}%, "
                    f"risk={last.get('risk_level','?')}, "
                    f"flags={','.join(last.get('key_flags',[]))}, "
                    f"outcome={last.get('outcome','?')}"
                )

    # Build ConversationInitiationData with or without prompt override
    if custom_prompt:
        # Full dynamic prompt injection — patient-specific prompt for this call
        config = ConversationInitiationData(
            dynamic_variables=dynamic_vars,
            conversation_config_override={
                "agent": {
                    "prompt": {
                        "prompt": custom_prompt
                    },
                    "first_message": (
                        f"Hi {patient_name}, this is Jessica from TrimRX. "
                        f"I'm calling for your monthly medication check-in — "
                        f"is now a good time?"
                    ),
                }
            }
        )
        print(f"[PROMPT INJECTION] Custom prompt injected for {patient_id} "
              f"({len(custom_prompt)} chars)")
    else:
        # No patient found — use ElevenLabs default prompt
        config = ConversationInitiationData(dynamic_variables=dynamic_vars)
        print("[PROMPT INJECTION] No patient found — using default ElevenLabs prompt")

    convo = Conversation(
        client=client,
        agent_id=os.getenv("ELEVENLABS_AGENT_ID"),
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=agent_cb,
        callback_user_transcript=user_cb,
        client_tools=tools,
        config=config,
    )

    global_state.active_conversation = convo
    global_state.start_time          = time.time()
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
    lm       = global_state.living_memory
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
        # v4 fields
        "glp1_risk_factors":      insights.get("glp1_risk_factors", []),
        "longitudinal_delta":     insights.get("longitudinal_delta"),
        "hesitation_fingerprint": global_state.hesitation_fingerprint,
        "mi_score":               insights.get("mi_score"),
        "mi_assessment":          insights.get("mi_assessment", ""),
        "previous_call_provided": bool(global_state.previous_call_context.strip()),
        # v5 fields
        "living_memory":          lm,
        "ghost_alerts":           lm["ghost_alerts"],
        "pre_call_brief":         lm["pre_call_brief"],
        "coverage_pct":           lm["coverage_pct"],
        "handoff_required":       lm["handoff_required"],
        "handoff_target":         lm["handoff_target"],
        "patient_id":             global_state.patient_id,
    }


@app.get("/api/memory-state")
def get_memory_state_endpoint():
    lm = global_state.living_memory
    return {
        "questions_remaining":     lm["questions_remaining"],
        "questions_answered":      lm["questions_answered"],
        "emotional_state":         lm["emotional_state"],
        "detected_flags":          lm["detected_flags"],
        "patient_risk_level":      lm["patient_risk_level"],
        "recommended_next_action": lm["recommended_next_action"],
        "handoff_required":        lm["handoff_required"],
        "handoff_target":          lm["handoff_target"],
        "ghost_alerts":            lm["ghost_alerts"],
        "coverage_pct":            lm["coverage_pct"],
        "pre_call_brief":          lm["pre_call_brief"],
        "signal_score":            global_state.signal_score,
        "behavior_mode":           compute_behavior_mode(),
    }


class PreCallBriefRequest(BaseModel):
    patient_id: str


@app.post("/api/pre-call-brief")
def pre_call_brief_endpoint(req: PreCallBriefRequest):
    """
    Called by ElevenLabs Workflow HTTP node OR dashboard button.
    Fetches patient, generates Groq brief, seeds Living Memory.
    Returns brief for dashboard display.
    """
    patient = get_patient(req.patient_id)
    if not patient:
        brief = {
            "strategy_summary":       "New patient. No prior history. Standard check-in approach.",
            "risk_prediction":        "MEDIUM",
            "priority_focus":         "overall_feeling",
            "agent_instruction":      "Introduce yourself warmly. Proceed with standard 14-question interview.",
            "sensitivities_reminder": "",
            "custom_system_prompt":   JESSICA_BASE_PROMPT,
        }
    else:
        brief = generate_pre_call_brief(patient)

    global_state.living_memory["pre_call_brief"]     = brief.get("strategy_summary", "")
    global_state.living_memory["patient_risk_level"] = brief.get("risk_prediction", "MEDIUM")

    if patient:
        history = patient.get("call_history", [])
        global_state.living_memory["call_number"] = len(history) + 1

    # Don't return custom_system_prompt in API response (too large, not needed by dashboard)
    brief_for_response = {k: v for k, v in brief.items() if k != "custom_system_prompt"}

    return {
        "patient_id":          req.patient_id,
        "patient_name":        patient["name"] if patient else "Unknown",
        "brief":               brief_for_response,
        "living_memory_seeded": True,
    }


class CallRequest(BaseModel):
    patient_name:          str
    medication:            str
    patient_context:       str
    previous_call_context: str = ""
    patient_id:            str = ""


@app.post("/api/start-call")
def start(req: CallRequest, bg: BackgroundTasks):
    global global_state
    global_state = CallState()

    # Store previous_call_context on state so run_call can access it
    global_state.previous_call_context = req.previous_call_context

    bg.add_task(
        run_call,
        req.patient_name,
        req.medication,
        req.patient_context,
        req.previous_call_context,
        req.patient_id,
    )
    return {"status": "started"}


@app.post("/api/end-call")
def end():
    trigger_auto_end("completed")
    return {"status": "ended"}


@app.get("/api/patient/{patient_id}")
def get_patient_info(patient_id: str):
    patient = get_patient(patient_id)
    if not patient:
        return JSONResponse(status_code=404, content={"error": "Patient not found"})
    return patient