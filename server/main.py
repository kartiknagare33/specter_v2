# server/main.py

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
app = FastAPI(title="TrimRX SPECTER Backend")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")

# ── 10 primary questions + 5 hidden validation probes ──────
QUESTIONS = [
    "How have you been feeling overall?",
    "What's your current weight in pounds?",
    "How much weight have you lost this past month in pounds?",
    "Any side effects from your medication this month?",
    "Satisfied with your rate of weight loss?",
    "Have you started any new medications or supplements since last month?",
    "Any new allergies?",
    "Any surgeries since your last check-in?",
    "Any questions for your doctor?",
    "Has your shipping address changed?",
    # Hidden validation probes (cross-check earlier answers)
    "How far do you feel you are from your target weight?",
    "How many days of medication supply do you have left?",
    "Has anything made it harder to take your medication consistently?",
    "If you could change one thing about your treatment, what would it be?",
    "Has your pharmacy filled anything new for you recently?"
]

QUESTION_LABELS = {
    "How have you been feeling overall?":                                   "Overall",
    "What's your current weight in pounds?":                                "Weight",
    "How much weight have you lost this past month in pounds?":             "Weight Loss",
    "Any side effects from your medication this month?":                    "Side Effects",
    "Satisfied with your rate of weight loss?":                             "Satisfaction",
    "Have you started any new medications or supplements since last month?": "New Meds",
    "Any new allergies?":                                                   "Allergies",
    "Any surgeries since your last check-in?":                              "Surgeries",
    "Any questions for your doctor?":                                       "Dr. Questions",
    "Has your shipping address changed?":                                   "Address",
    "How far do you feel you are from your target weight?":                 "[V] Target Gap",
    "How many days of medication supply do you have left?":                 "[V] Supply Days",
    "Has anything made it harder to take your medication consistently?":    "[V] Adherence",
    "If you could change one thing about your treatment, what would it be?":"[V] Change Ask",
    "Has your pharmacy filled anything new for you recently?":              "[V] Pharmacy"
}

PROBE_LABELS = {
    "A_acoustic_fidelity":   "Acoustic",
    "B_intent_alignment":    "Intent",
    "C_coverage_integrity":  "Coverage",
    "D_safety_compliance":   "Safety",
    "E_identity_signal":     "Identity",
    "F_capture_accuracy":    "Capture",
    "G_behavioral_signal":   "Behavioral"
}

# Signal weights — how much each flag adds to accumulated score
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


# ═══════════════════════════════════════════════════════════
# CALL STATE
# ═══════════════════════════════════════════════════════════
class CallState:
    def __init__(self):
        self.responses = [{"question": q, "answer": "", "flag": "none"}
                          for q in QUESTIONS]
        self.outcome             = None
        self.transcript          = []
        self.insights            = None
        self.active_conversation = None
        self.start_time          = None
        self.call_duration       = 0
        self.behavior_mode       = "STANDBY"
        self._prev_g_score       = 1.0

        # Signal accumulation for adaptive depth
        self.signal_score        = 0
        self.contradictions      = []   # list of dicts
        self.interview_depth     = "standard"  # standard / soft_probe / deep_probe

        # 7 SPECTER probes
        self.probes = {
            "A_acoustic_fidelity":  1.0,
            "B_intent_alignment":   1.0,
            "C_coverage_integrity": 1.0,
            "D_safety_compliance":  1.0,
            "E_identity_signal":    1.0,
            "F_capture_accuracy":   1.0,
            "G_behavioral_signal":  1.0
        }

global_state = CallState()


# ═══════════════════════════════════════════════════════════
# SIGNAL ACCUMULATION
# ═══════════════════════════════════════════════════════════
def accumulate_signal(flag: str):
    """Add flag weight to signal score and recalculate interview depth."""
    weight = SIGNAL_WEIGHTS.get(flag, 0)
    global_state.signal_score += weight

    if global_state.signal_score >= 7:
        global_state.interview_depth = "deep_probe"
    elif global_state.signal_score >= 3:
        global_state.interview_depth = "soft_probe"
    else:
        global_state.interview_depth = "standard"

    print(f"[SIGNAL] score={global_state.signal_score} "
          f"flag={flag} depth={global_state.interview_depth}")


# ═══════════════════════════════════════════════════════════
# CROSS-VALIDATION — detect contradictions between answers
# ═══════════════════════════════════════════════════════════
def check_cross_validation():
    """
    Compare primary answers against their hidden validation probes.
    Called after each validation probe answer is logged.
    """
    def get_answer(label_fragment):
        for r in global_state.responses:
            if label_fragment.lower() in r["question"].lower():
                return r["answer"].lower()
        return ""

    contradictions_found = []

    # Check 1: Weight loss satisfaction vs stated weight loss
    weight_lost   = get_answer("how much weight have you lost")
    satisfaction  = get_answer("satisfied with your rate")
    target_gap    = get_answer("how far do you feel you are from your target")

    if weight_lost and satisfaction and target_gap:
        lost_positive = any(w in weight_lost for w in ["pound", "lb", "kilo"])
        sat_negative  = any(w in satisfaction for w in ["no", "not", "barely", "nothing"])
        if lost_positive and sat_negative:
            contradictions_found.append({
                "field_a": "weight_lost",
                "answer_a": weight_lost,
                "field_b": "satisfaction",
                "answer_b": satisfaction,
                "summary": "Patient reported weight loss but expressed dissatisfaction, suggesting perceived vs actual progress mismatch."
            })

    # Check 2: No missed doses vs low supply
    missed_doses = get_answer("missed doses") or get_answer("how have you been feeling")
    supply_days  = get_answer("how many days of medication supply")

    if supply_days:
        low_supply = any(w in supply_days for w in
                        ["few", "3", "4", "5", "running out", "almost", "low", "one week"])
        no_missed  = any(w in missed_doses for w in ["no", "none", "haven't", "i haven"])
        if low_supply and no_missed:
            contradictions_found.append({
                "field_a": "missed_doses",
                "answer_a": missed_doses,
                "field_b": "supply_remaining",
                "answer_b": supply_days,
                "summary": "Patient denied missing doses but reported very low remaining supply, which may be inconsistent."
            })

    # Check 3: New medications direct vs pharmacy probe
    new_meds    = get_answer("started any new medications")
    pharmacy    = get_answer("has your pharmacy filled anything new")

    if new_meds and pharmacy:
        denied_new  = any(w in new_meds for w in ["no", "none", "nothing", "haven"])
        pharmacy_new = any(w in pharmacy for w in ["yes", "they did", "actually", "one", "new"])
        if denied_new and pharmacy_new:
            contradictions_found.append({
                "field_a": "new_medications",
                "answer_a": new_meds,
                "field_b": "pharmacy_recent",
                "answer_b": pharmacy,
                "summary": "Patient denied new medications but confirmed recent pharmacy fill, suggesting unreported medication."
            })

    # Check 4: Side effects vs adherence probe
    side_effects = get_answer("any side effects")
    adherence    = get_answer("has anything made it harder")

    if side_effects and adherence:
        denied_se   = any(w in side_effects for w in ["no", "none", "nothing"])
        harder_true = any(w in adherence for w in
                         ["yes", "nausea", "sick", "tired", "forget", "hard", "difficult"])
        if denied_se and harder_true:
            contradictions_found.append({
                "field_a": "side_effects",
                "answer_a": side_effects,
                "field_b": "adherence_difficulty",
                "answer_b": adherence,
                "summary": "Patient denied side effects but reported difficulty with consistent adherence, which may be related."
            })

    # Log new contradictions and update signal score
    for c in contradictions_found:
        if c not in global_state.contradictions:
            global_state.contradictions.append(c)
            accumulate_signal("contradiction")
            global_state.probes["F_capture_accuracy"] = max(
                0.0,
                global_state.probes["F_capture_accuracy"] - 0.25
            )
            print(f"[CONTRADICTION] {c['summary']}")


# ═══════════════════════════════════════════════════════════
# PROBE SCORING
# ═══════════════════════════════════════════════════════════
def score_probes_on_answer(patient_answer: str, flag: str):
    answer_lower = patient_answer.lower()

    # Probe B — opt-out / wrong number
    OPT_OUT = ["stop calling","remove me","don't call","wrong number",
               "who is this","not interested"]
    if any(t in answer_lower for t in OPT_OUT):
        global_state.probes["B_intent_alignment"] = 0.0

    # Probe D — medical advice solicitation
    SAFETY = ["should i take","what should i do","is it safe",
              "can i mix","what is the dosage","can i stop"]
    if any(t in answer_lower for t in SAFETY):
        global_state.probes["D_safety_compliance"] = 0.0
        accumulate_signal("safety_concern")

    # Probe G — behavioral signal
    prev_g = global_state.probes["G_behavioral_signal"]
    if flag == "behavioral_uncertainty":
        global_state.probes["G_behavioral_signal"] = 0.0
        global_state._prev_g_score = 0.0
    elif flag == "none" and prev_g == 0.0:
        global_state.probes["G_behavioral_signal"] = 1.0
        global_state._prev_g_score = 1.0

    # Probe F — capture ratio
    answered = sum(1 for r in global_state.responses if r["answer"])
    global_state.probes["F_capture_accuracy"]   = round(answered / len(QUESTIONS), 2)
    global_state.probes["C_coverage_integrity"] = round(answered / len(QUESTIONS), 2)

    # Probe A — acoustic (implausible numeric weight)
    WEIGHT_Q = ["weight", "pounds"]
    if any(w in answer_lower for w in WEIGHT_Q):
        try:
            val = float(''.join(c for c in patient_answer if c.isdigit() or c == '.'))
            global_state.probes["A_acoustic_fidelity"] = 0.0 if (val < 50 or val > 500) else 1.0
        except (ValueError, TypeError):
            pass


# ═══════════════════════════════════════════════════════════
# GRAPH DATA
# ═══════════════════════════════════════════════════════════
def build_graph_data() -> dict:
    nodes = [{"id": "SPECTER", "group": 1}]
    links = []

    safety_failed   = global_state.probes["D_safety_compliance"] < 1.0
    behavioral_flag = global_state.probes["G_behavioral_signal"]  < 1.0
    contradicted_qs = {c["field_a"] for c in global_state.contradictions} | \
                      {c["field_b"] for c in global_state.contradictions}

    for item in global_state.responses:
        label    = QUESTION_LABELS.get(item["question"], item["question"][:14])
        answered = bool(item["answer"])
        is_validation = label.startswith("[V]")

        if answered:
            topic = item["question"].lower()
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


# ═══════════════════════════════════════════════════════════
# BEHAVIOR MODE
# ═══════════════════════════════════════════════════════════
def compute_behavior_mode() -> str:
    if global_state.outcome:
        return "SYNTHESIS COMPLETE"

    if global_state.probes["B_intent_alignment"] < 1.0:
        return "DE-ESCALATION"
    if global_state.probes["D_safety_compliance"] < 1.0:
        return "CLINICAL ADVISORY"
    if global_state.probes["G_behavioral_signal"] < 1.0:
        return "EMPATHY VALIDATION"

    depth = global_state.interview_depth
    if depth == "deep_probe":
        return "DEEP PROBE ACTIVE"
    if depth == "soft_probe":
        return "SOFT PROBE ACTIVE"

    if global_state.contradictions:
        return "CONTRADICTION AUDIT"

    EMPATHY_WORDS = ["sorry to hear","understand","that must","completely normal",
                     "side effect","nausea","vomiting","dizzy","headache","fatigue"]
    recent = [t["message"].lower() for t in global_state.transcript[-6:]
              if t.get("role") == "agent"]
    for u in recent:
        if any(w in u for w in EMPATHY_WORDS):
            return "EMPATHY PROTOCOL"

    answered = sum(1 for r in global_state.responses if r["answer"])
    if answered == 0:          return "STANDARD CLINICAL"
    if answered >= len(QUESTIONS): return "COVERAGE COMPLETE"
    return "STANDARD CLINICAL"


# ═══════════════════════════════════════════════════════════
# RUN CALL
# ═══════════════════════════════════════════════════════════
def run_call(patient_name: str, medication: str, patient_context: str):
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    def trigger_auto_end(reason="completed"):
        if global_state.outcome:
            return
        print(f"🛑 KILL SWITCH: {reason}")
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
                global_state.insights = run_final_llm_deliberation(
                    global_state.transcript,
                    global_state.responses
                )
                # Inject contradiction data into edge case report
                if global_state.contradictions:
                    c_text = "\n\n[CROSS-VALIDATION FLAGS]\n"
                    for i, c in enumerate(global_state.contradictions, 1):
                        c_text += f"{i}. {c['summary']}\n"
                    global_state.insights["edge_case_report"] += c_text
            except Exception as e:
                print("LLM ERROR:", e)

        threading.Thread(target=run_llm_task).start()

    # ── TOOL: log_answer ────────────────────────────────────
    def log_answer(question_topic: str = "", patient_answer: str = "",
                   confidence: str = "clear", clinical_flag: str = "none", **kwargs):
        print(f"[LOG] {question_topic} → {patient_answer} [{clinical_flag}]")

        # Match and store
        for item in global_state.responses:
            if (question_topic.lower() in item["question"].lower() or
                    item["question"].lower() in question_topic.lower()):
                item["answer"] = patient_answer
                item["flag"]   = clinical_flag
                break

        # Update probes and signal score
        score_probes_on_answer(patient_answer, clinical_flag)
        accumulate_signal(clinical_flag)

        # Run cross-validation after each validation probe
        validation_triggers = [
            "target weight", "days of medication", "harder to take",
            "change one thing", "pharmacy filled"
        ]
        if any(t in question_topic.lower() for t in validation_triggers):
            check_cross_validation()

        # Live churn meter update
        try:
            ans_dict = {i["question"]: i["answer"]
                        for i in global_state.responses if i["answer"]}
            global_state.insights = run_deliberation(
                ans_dict, global_state.transcript, [], True
            )
        except Exception:
            pass

        return (
            f"SUCCESS. Signal score: {global_state.signal_score}. "
            f"Interview depth: {global_state.interview_depth}. "
            f"Contradictions: {len(global_state.contradictions)}. "
            f"Proceed to next step."
        )

    # ── TOOL: verify_identity ───────────────────────────────
    def verify_identity(patient_confirmed_name: str = "",
                        date_of_birth: str = "",
                        identity_match: str = "confirmed", **kwargs):
        print(f"[IDENTITY] {patient_confirmed_name} | {date_of_birth} | {identity_match}")
        if identity_match == "mismatch":
            global_state.probes["E_identity_signal"] = 0.0
        return f"Identity {identity_match}. Proceed accordingly."

    # ── TOOL: schedule_callback ─────────────────────────────
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

    # ── TOOL: escalate_to_pharmacist ────────────────────────
    def escalate_to_pharmacist(symptom_description: str = "",
                                urgency_level: str = "urgent",
                                patient_acknowledged_followup: str = "yes", **kwargs):
        print(f"[ESCALATION] {urgency_level}: {symptom_description}")
        global_state.probes["D_safety_compliance"] = 0.0
        global_state.signal_score += 8
        global_state.interview_depth = "deep_probe"
        global_state.transcript.append({
            "role": "system",
            "message": f"[ESCALATION TRIGGERED] {urgency_level.upper()}: {symptom_description}",
            "timestamp": time.time()
        })
        return "Escalation flagged. End call immediately after safety messaging."

    # ── TOOL: capture_pricing_concern ───────────────────────
    def capture_pricing_concern(concern_description: str = "",
                                 patient_wants_followup: str = "yes", **kwargs):
        print(f"[PRICING] {concern_description}")
        accumulate_signal("pricing_question")
        return "Pricing concern logged. Provide billing contact and continue."

    # ── TOOL: flag_contradiction ────────────────────────────
    def flag_contradiction(field_a: str = "", answer_a: str = "",
                            field_b: str = "", answer_b: str = "",
                            contradiction_summary: str = "", **kwargs):
        print(f"[CONTRADICTION] {field_a} vs {field_b}: {contradiction_summary}")
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

    # ── TOOL: end_call ──────────────────────────────────────
    def end_call(outcome: str = "completed",
                 refill_approved: str = "yes",
                 summary: str = "", **kwargs):
        answered = sum(1 for r in global_state.responses if r["answer"])
        global_state.probes["C_coverage_integrity"] = (
            1.0 if answered >= 10  # primary questions only
            else round(answered / 10, 2)
        )
        trigger_auto_end(outcome)
        return "ended"

    # ── REGISTER TOOLS ──────────────────────────────────────
    tools = ClientTools()
    tools.register("log_answer",              log_answer)
    tools.register("verify_identity",         verify_identity)
    tools.register("schedule_callback",       schedule_callback)
    tools.register("escalate_to_pharmacist",  escalate_to_pharmacist)
    tools.register("capture_pricing_concern", capture_pricing_concern)
    tools.register("flag_contradiction",      flag_contradiction)
    tools.register("end_call",                end_call)

    # ── CALLBACKS ───────────────────────────────────────────
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

    # ── START SESSION ───────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════
@app.get("/")
def home():
    return FileResponse(os.path.join(DEMO_DIR, "index.html"))


@app.get("/call-state")
def get_state():
    return {
        "responses":           global_state.responses,
        "outcome":             global_state.outcome,
        "transcript":          global_state.transcript,
        "insights":            global_state.insights,
        "call_duration":       global_state.call_duration,
        "probes":              global_state.probes,
        "probe_labels":        PROBE_LABELS,
        "graph_data":          build_graph_data(),
        "agent_behavior_mode": compute_behavior_mode(),
        # New fields for UI
        "signal_score":        global_state.signal_score,
        "interview_depth":     global_state.interview_depth,
        "contradictions":      global_state.contradictions,
    }


class CallRequest(BaseModel):
    patient_name: str
    medication: str
    patient_context: str


@app.post("/api/start-call")
def start(req: CallRequest, bg: BackgroundTasks):
    global global_state
    global_state = CallState()
    bg.add_task(run_call, req.patient_name, req.medication, req.patient_context)
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
                global_state.insights = run_final_llm_deliberation(
                    global_state.transcript, global_state.responses
                )
            except Exception as e:
                print("LLM ERROR:", e)

        threading.Thread(target=run_llm_task).start()
    return {"status": "ended"}
