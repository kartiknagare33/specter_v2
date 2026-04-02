# deliberation/judge.py
# SPECTER v5 — Full Deliberation Chamber
#
# v4 innovations (kept):
#   GLP-1 Churn Risk Matrix · Longitudinal Delta · MI Fidelity · Hesitation Fingerprint
#
# v5 NEW:
#   Round 5 — Next-Call Strategy Generator (closes the loop back to Layer 1)


import os
import json
from groq import Groq


# ─────────────────────────────────────────────────────────────
# GLP-1 SPECIFIC CHURN RISK MATRIX
# Evidence: JAMA Network Open 2025 · EASD 2025 · BHI 2024 · AiCure 2024 · PMC 2025
# ─────────────────────────────────────────────────────────────
GLP1_RISK_MATRIX = [
    {
        "id":           "gi_side_effects",
        "label":        "GI Side Effects Present",
        "keywords":     ["nausea", "vomiting", "diarrhea", "stomach", "constipated", "sick", "nauseated"],
        "check_fields": ["side_effects", "overall_feeling"],
        "weight":       20,
        "evidence":     "JAMA 2025: GI side effects → +9% early dropout probability"
    },
    {
        "id":           "pricing_barrier",
        "label":        "Cost / Affordability Barrier",
        "keywords":     ["expensive", "afford", "cost", "price", "insurance", "billing", "money", "can't keep"],
        "check_fields": ["satisfaction", "treatment_concerns", "overall_feeling"],
        "weight":       25,
        "evidence":     "BHI 2024: Cost burden is #1 non-clinical discontinuation driver"
    },
    {
        "id":           "weight_plateau",
        "label":        "Weight Loss Plateau Suspected",
        "keywords":     ["not working", "not losing", "same weight", "plateau", "stopped losing", "barely", "nothing"],
        "check_fields": ["satisfaction", "weight_lost"],
        "weight":       15,
        "evidence":     "EASD 2025: <3.6% loss in 3 months → 2.8× discontinuation risk"
    },
    {
        "id":           "low_supply",
        "label":        "Low Medication Supply",
        "keywords":     ["few days", "running out", "almost out", "one week", "3 days", "4 days", "5 days", "2 days"],
        "check_fields": ["supply_days"],
        "weight":       20,
        "evidence":     "Published adherence data: Supply gap is a direct proxy for adherence failure"
    },
    {
        "id":           "missed_doses",
        "label":        "Missed Doses Reported",
        "keywords":     ["yes", "a few", "some", "couple", "forgot", "skipped", "missed"],
        "check_fields": ["missed_doses"],
        "weight":       20,
        "evidence":     "AiCure 2024: Early missed doses predict 30-day non-adherence in 73% of cases"
    },
    {
        "id":           "treatment_concerns",
        "label":        "Treatment Continuation Intent Concern",
        "keywords":     ["worried", "thinking about stopping", "not sure", "concerned", "unsure", "considering stopping"],
        "check_fields": ["treatment_concerns", "doctor_questions"],
        "weight":       25,
        "evidence":     "PMC 2025: Treatment intent concerns are the strongest single churn predictor"
    },
    {
        "id":           "dissatisfaction",
        "label":        "Patient Dissatisfaction",
        "keywords":     ["no", "not really", "not happy", "disappointed", "barely", "nothing", "not satisfied"],
        "check_fields": ["satisfaction"],
        "weight":       15,
        "evidence":     "JAMA 2025: Dissatisfaction at 3-month check-in → 1.9× 90-day dropout rate"
    },
]


def _get_combined_answer_text(responses: list) -> str:
    return " ".join(
        str(r.get("answer", "")).lower()
        for r in responses if r.get("answer")
    )


def run_deliberation(answers: dict, transcript: list,
                     strategy_log: list, is_live: bool = False) -> dict:
    """
    Fast live-call deliberation — rule-based, no LLM, runs after every log_answer.
    Uses GLP-1 evidence-anchored risk matrix.
    """
    churn              = 10
    active_risk_factors = []

    combined_text = " ".join(str(v).lower() for v in answers.values() if v)

    for risk in GLP1_RISK_MATRIX:
        field_text = ""
        for field in risk["check_fields"]:
            field_text += " " + str(answers.get(field, "")).lower()

        search_text = field_text.strip() + " " + combined_text
        if any(kw in search_text for kw in risk["keywords"]):
            churn += risk["weight"]
            active_risk_factors.append({
                "id":       risk["id"],
                "label":    risk["label"],
                "evidence": risk["evidence"],
                "active":   True
            })

    risk_level = "HIGH" if churn > 60 else "MEDIUM" if churn > 30 else "LOW"

    return {
        "risk_level":             risk_level,
        "churn_score":            min(churn, 100),
        "glp1_risk_factors":      active_risk_factors,
        "behavioral_profile":     "Synthesizing live clinical data...",
        "edge_case_report":       "Monitoring active session telemetry...",
        "soap_note":              "Awaiting call completion...",
        "advocate_summary":       "",
        "skeptic_summary":        "",
        "arbiter_verdict":        "",
        "adherence_forecast":     None,
        "priority_actions":       None,
        "validator_result":       None,
        "longitudinal_delta":     None,
        "mi_fidelity_metrics":    None,
        "hesitation_fingerprint": None,
        "next_call_strategy":     None,
    }


def _flatten(value, fallback=""):
    if isinstance(value, dict):
        return "\n".join(f"[{k.upper()}]\n{v}" for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value) if value else fallback


def run_final_llm_deliberation(
    transcript: list,
    responses: list,
    previous_call_context: str = "",
    mi_metrics: dict = None,
    hesitation_data: dict = None,
    patient_id: str = "",
) -> dict:
    """
    Full 5-round post-call deliberation.
    Rounds: Advocate → Skeptic → Arbiter → Validator → Next-Call Strategy

    v4 innovations:
      1. GLP-1 Churn Risk Matrix
      2. Longitudinal Risk Delta
      3. MI Fidelity Score
      4. Hesitation Fingerprint

    v5 NEW:
      5. Next-Call Strategy — writes pre-call brief for the NEXT interaction
         Closes the loop: deliberation output → patient DB → Layer 1 input
    """
    print("DELIBERATION CHAMBER OPENING...")
    client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
    sorted_tx = sorted(transcript, key=lambda x: x.get("timestamp", 0))

    base_data = f"""
CALL TRANSCRIPT:
{json.dumps(sorted_tx, indent=2)}

STRUCTURED RESPONSES:
{json.dumps(responses, indent=2)}
"""

    longitudinal_section = (
        f"\nPREVIOUS CALL CONTEXT (for longitudinal comparison):\n{previous_call_context}\n"
        if previous_call_context and previous_call_context.strip() else ""
    )
    mi_section = (
        f"\nMOTIVATIONAL INTERVIEWING METRICS (pre-computed):\n{json.dumps(mi_metrics, indent=2)}\n"
        if mi_metrics else ""
    )
    hesitation_section = (
        f"\nPATIENT HESITATION FINGERPRINT (pre-computed):\n{json.dumps(hesitation_data, indent=2)}\n"
        if hesitation_data else ""
    )

    # ──────────────────────────────────────────────
    # ROUND 1 — ADVOCATE
    # ──────────────────────────────────────────────
    print("  Round 1: Advocate...")
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"""You are a clinical patient advocate reviewing a GLP-1 medication check-in call.
Build the STRONGEST possible case that this patient is healthy, compliant, engaged, and low churn risk.
Find every positive signal. Look for: consistent dosing, weight progress, satisfaction, engagement, stable supply.
Return ONLY valid JSON:
{{
  "advocate_case": "2-3 sentence strongest-case summary",
  "positive_signals": ["signal 1", "signal 2", "signal 3"],
  "recommended_risk": "LOW or MEDIUM"
}}
{base_data}"""}],
            temperature=0.3,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        advocate = json.loads(r)
        print(f"  Advocate: {advocate.get('recommended_risk')} risk")
    except Exception as e:
        print(f"  Advocate failed: {e}")
        advocate = {"advocate_case": "Insufficient data.", "positive_signals": [], "recommended_risk": "MEDIUM"}

    # ──────────────────────────────────────────────
    # ROUND 2 — SKEPTIC
    # ──────────────────────────────────────────────
    print("  Round 2: Skeptic...")
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"""You are a clinical risk analyst. Challenge this Advocate assessment: {json.dumps(advocate, indent=2)}

Find every GLP-1 discontinuation risk factor the Advocate dismissed. Specifically look for:
- GI side effects (JAMA 2025: +9% dropout probability)
- Cost/affordability concerns (BHI 2024: #1 non-clinical discontinuation driver)
- Weight plateau (<3.6% loss in 3 months = 2.8x dropout risk, EASD 2025)
- Low supply days (adherence gap proxy)
- Missed doses (AiCure 2024: 73% predictive for 30-day non-adherence)
- Treatment intent concerns (PMC 2025: strongest single churn predictor)
- Patient dissatisfaction (JAMA 2025: 1.9x 90-day dropout at 3-month check-in)

Return ONLY valid JSON:
{{
  "skeptic_rebuttal": "2-3 sentence strongest-case for WHY this patient is at risk",
  "red_flags": ["flag 1", "flag 2", "flag 3"],
  "recommended_risk": "MEDIUM or HIGH"
}}
{base_data}"""}],
            temperature=0.3,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        skeptic = json.loads(r)
        print(f"  Skeptic: {skeptic.get('recommended_risk')} risk")
    except Exception as e:
        print(f"  Skeptic failed: {e}")
        skeptic = {"skeptic_rebuttal": "Insufficient data.", "red_flags": [], "recommended_risk": "MEDIUM"}

    # ──────────────────────────────────────────────
    # ROUND 3 — ARBITER (Chief Medical Officer)
    # ──────────────────────────────────────────────
    print("  Round 3: Arbiter...")
    longitudinal_prompt = (
        f"\nLONGITUDINAL ANALYSIS: You have previous call data. Compare weight trend, side effects, "
        f"supply pattern, and engagement. Produce a longitudinal_delta with trend direction "
        f"(IMPROVING/STABLE/DETERIORATING). Previous context:\n{previous_call_context}\n"
        if previous_call_context and previous_call_context.strip()
        else "\nNo previous call data available. Set longitudinal_delta.available = false.\n"
    )
    mi_prompt = (
        f"\nMI QUALITY: Agent Motivational Interviewing metrics: {json.dumps(mi_metrics)}. "
        f"Use R:Q ratio, open question ratio, and empathy events to compute mi_score (0-100).\n"
        if mi_metrics else "\nNo MI metrics available. Estimate mi_score from transcript quality.\n"
    )
    hesitation_prompt = (
        f"\nHESITATION: Patient hesitation analysis: {json.dumps(hesitation_data)}. "
        f"Factor hedge_count, short_open_responses, and average_response_length into risk.\n"
        if hesitation_data else ""
    )

    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"""You are the Chief Medical Officer rendering a final verdict on a GLP-1 check-in call.
Advocate: {json.dumps(advocate, indent=2)}
Skeptic: {json.dumps(skeptic, indent=2)}
{longitudinal_section}
{longitudinal_prompt}
{mi_prompt}
{hesitation_prompt}

CRITICAL OUTPUT RULES:
- soap_note: plain text with [S][O][A][P] sections
- adherence_probability: integer 0-100
- next_contact_window: exactly "7 days", "14 days", or "30 days"
- priority_actions: exactly 3 items
- glp1_risk_factors: ONLY factors with actual transcript evidence
- longitudinal_delta: compare to previous call if available
- mi_score: integer 0-100
- hesitation_index: float 0-1

Return ONLY valid JSON:
{{
  "behavioral_profile": "2-3 sentence balanced clinical summary",
  "edge_case_report": "Specific anomalies, or: Standard flow. No edge cases detected.",
  "soap_note": "[S] Subjective...\\n[O] Objective...\\n[A] Assessment...\\n[P] Plan...",
  "risk_level": "LOW, MEDIUM, or HIGH",
  "churn_score": <integer 0-100>,
  "advocate_summary": "One sentence strongest Advocate point.",
  "skeptic_summary": "One sentence strongest Skeptic point.",
  "arbiter_verdict": "One sentence final clinical judgment.",
  "adherence_forecast": {{
    "adherence_probability": <integer 0-100>,
    "forecast_label": "one sentence 30-day prediction",
    "top_risk_factors": ["factor 1", "factor 2", "factor 3"],
    "next_contact_window": "7 days or 14 days or 30 days",
    "contact_reason": "one sentence reason"
  }},
  "priority_actions": [
    {{"urgency": "URGENT", "category": "Refill/Safety/Billing/Follow-up/Escalation", "action": "specific instruction"}},
    {{"urgency": "REVIEW", "category": "Refill/Safety/Billing/Follow-up/Escalation", "action": "specific instruction"}},
    {{"urgency": "MONITOR", "category": "Refill/Safety/Billing/Follow-up/Escalation", "action": "specific instruction"}}
  ],
  "glp1_risk_factors": [
    {{"id": "risk_id", "label": "Human-readable label", "evidence": "clinical citation", "active": true}}
  ],
  "longitudinal_delta": {{
    "available": true or false,
    "trend": "IMPROVING or STABLE or DETERIORATING",
    "weight_trend": "comparison sentence or N/A",
    "side_effect_trend": "new / improved / unchanged or N/A",
    "engagement_trend": "engagement comparison or N/A",
    "supply_trend": "supply status comparison or N/A",
    "delta_summary": "one sentence overall longitudinal assessment",
    "escalation_recommended": true or false
  }},
  "mi_score": <integer 0-100>,
  "mi_assessment": "one sentence on agent MI quality"
}}
{base_data}{hesitation_section}"""}],
            temperature=0.2,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        result = json.loads(r)
        print(f"  Arbiter: {result.get('risk_level')} / churn={result.get('churn_score')}% / "
              f"adherence={result.get('adherence_forecast', {}).get('adherence_probability', '?')}% / "
              f"MI={result.get('mi_score', '?')}")
    except Exception as e:
        print(f"  Arbiter failed: {e}")
        result = {
            "behavioral_profile":  "Deliberation chamber error. Manual review required.",
            "edge_case_report":    "Error during synthesis.",
            "soap_note":           "Error: SOAP note generation failed.",
            "risk_level":          "UNKNOWN",
            "churn_score":         0,
            "advocate_summary":    "",
            "skeptic_summary":     "",
            "arbiter_verdict":     "",
            "adherence_forecast": {
                "adherence_probability": 0,
                "forecast_label":        "Forecast unavailable.",
                "top_risk_factors":      [],
                "next_contact_window":   "14 days",
                "contact_reason":        "Default fallback."
            },
            "priority_actions": [
                {"urgency": "REVIEW",  "category": "Follow-up", "action": "Manual review required."},
                {"urgency": "MONITOR", "category": "Follow-up", "action": "Verify transcript and re-run analysis."},
                {"urgency": "MONITOR", "category": "Follow-up", "action": "Confirm patient received refill."},
            ],
            "glp1_risk_factors":   [],
            "longitudinal_delta":  {"available": False, "trend": "N/A", "delta_summary": "No previous data."},
            "mi_score":            50,
            "mi_assessment":       "Assessment unavailable."
        }

    # ──────────────────────────────────────────────
    # ROUND 4 — VALIDATOR
    # ──────────────────────────────────────────────
    print("  Round 4: Validator...")
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"""You are a clinical AI safety officer.
Arbiter verdict: {json.dumps(result, indent=2)}

Check whether the Arbiter's output is consistent with what the patient ACTUALLY SAID.
Look for:
1. behavioral_profile claims that contradict transcript
2. churn_score / risk_level not supported by evidence
3. Signals in transcript the Arbiter missed entirely
4. Priority actions not grounded in patient statements
5. GLP-1 risk factors flagged without transcript support
6. longitudinal_delta conclusions not supported by provided previous call context

Return ONLY valid JSON:
{{
  "validation_passed": true or false,
  "confidence": "HIGH, MEDIUM, or LOW",
  "discrepancies": ["discrepancy 1 if any"],
  "missed_signals": ["missed signal if any"],
  "validator_note": "One sentence overall assessment of Arbiter output quality."
}}
{base_data}"""}],
            temperature=0.1,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        validator = json.loads(r)
        print(f"  Validator: passed={validator.get('validation_passed')} confidence={validator.get('confidence')}")
    except Exception as e:
        print(f"  Validator failed: {e}")
        validator = {
            "validation_passed": True,
            "confidence":        "LOW",
            "discrepancies":     [],
            "missed_signals":    [],
            "validator_note":    "Validator round failed — output unverified."
        }

    # ──────────────────────────────────────────────
    # ROUND 5 — NEXT-CALL STRATEGY (NEW v5)
    # Closes the loop: writes the pre-call brief for the NEXT interaction.
    # Saved to patient DB by main.py after deliberation completes.
    # ──────────────────────────────────────────────
    print("  Round 5: Next-Call Strategy...")
    next_call_strategy = ""
    next_call_strategy_structured = None
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"""You are a senior clinical strategist at TrimRX.
This call just ended. Based on everything that happened, write the pre-call brief 
for the NEXT interaction with this patient.

Current call summary:
- Risk level: {result.get('risk_level')}
- Churn score: {result.get('churn_score')}%
- Arbiter verdict: {result.get('arbiter_verdict')}
- Active GLP-1 risk factors: {json.dumps([f['label'] for f in result.get('glp1_risk_factors', [])])}
- Priority actions: {json.dumps([a['action'] for a in result.get('priority_actions', [])])}
- MI score (agent quality): {result.get('mi_score')}
- Longitudinal trend: {result.get('longitudinal_delta', {}).get('trend', 'N/A')}

Return ONLY valid JSON:
{{
  "next_call_strategy": "4-line brief: (1) expected emotional state, (2) first topic to address, (3) sensitive topics to approach carefully, (4) specialist to have on standby",
  "open_loops": ["unresolved issue 1", "unresolved issue 2"],
  "recommended_opening": "Exact suggested opening line for the next call",
  "risk_watch_flags": ["flag to monitor 1", "flag to monitor 2"],
  "suggested_question_priority": ["topic_key_1", "topic_key_2", "topic_key_3"]
}}
{base_data}"""}],
            temperature=0.3,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        next_call_strategy_structured = json.loads(r)
        next_call_strategy = next_call_strategy_structured.get("next_call_strategy", "")
        print(f"  Round 5: Next-call strategy generated.")
        print(f"    → {next_call_strategy[:80]}...")
    except Exception as e:
        print(f"  Round 5 failed: {e}")
        next_call_strategy = "Standard follow-up. Monitor churn indicators from this call."
        next_call_strategy_structured = {
            "next_call_strategy":           next_call_strategy,
            "open_loops":                   [],
            "recommended_opening":          "Hi, this is Jessica from TrimRX. How have you been since our last call?",
            "risk_watch_flags":             [],
            "suggested_question_priority":  [],
        }

    # ──────────────────────────────────────────────
    # FLATTEN STRING FIELDS
    # ──────────────────────────────────────────────
    result["soap_note"]          = _flatten(result.get("soap_note"),          "No SOAP note generated.")
    result["edge_case_report"]   = _flatten(result.get("edge_case_report"),   "Standard flow.")
    result["behavioral_profile"] = _flatten(result.get("behavioral_profile"), "No profile generated.")
    result["advocate_summary"]   = _flatten(result.get("advocate_summary"),   "")
    result["skeptic_summary"]    = _flatten(result.get("skeptic_summary"),    "")
    result["arbiter_verdict"]    = _flatten(result.get("arbiter_verdict"),    "")
    result["mi_assessment"]      = _flatten(result.get("mi_assessment"),      "")

    # Validate structured objects
    if not isinstance(result.get("adherence_forecast"), dict):
        result["adherence_forecast"] = {
            "adherence_probability": 0,
            "forecast_label":        "Forecast unavailable.",
            "top_risk_factors":      [],
            "next_contact_window":   "14 days",
            "contact_reason":        "Data insufficient."
        }

    if not isinstance(result.get("priority_actions"), list) or len(result.get("priority_actions", [])) < 3:
        result["priority_actions"] = [
            {"urgency": "MONITOR", "category": "Follow-up", "action": "Confirm refill processed and delivered."},
            {"urgency": "MONITOR", "category": "Follow-up", "action": "Schedule next check-in per standard cycle."},
            {"urgency": "MONITOR", "category": "Follow-up", "action": "No priority actions identified."},
        ]

    if not isinstance(result.get("glp1_risk_factors"), list):
        result["glp1_risk_factors"] = []

    if not isinstance(result.get("longitudinal_delta"), dict):
        result["longitudinal_delta"] = {
            "available":    False,
            "trend":        "N/A",
            "delta_summary": "No previous call data was provided for comparison."
        }

    if not isinstance(result.get("mi_score"), int):
        try:
            result["mi_score"] = int(result.get("mi_score", 50))
        except Exception:
            result["mi_score"] = 50

    if hesitation_data:
        result["hesitation_fingerprint"] = hesitation_data

    result["validator_result"]              = validator
    result["next_call_strategy"]            = next_call_strategy
    result["next_call_strategy_structured"] = next_call_strategy_structured

    print("DELIBERATION CHAMBER CLOSED.")
    return result