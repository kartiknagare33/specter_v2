# deliberation/judge.py

import os
import json
from groq import Groq


# ═══════════════════════════════════════════════════════════
# LIVE DELIBERATION — fast, runs during call for churn meter
# ═══════════════════════════════════════════════════════════
def run_deliberation(answers: dict, transcript: list,
                     strategy_log: list, is_live: bool = False) -> dict:
    churn = 10
    flags = []

    satisfied = str(answers.get("Satisfied with your rate of weight loss?", "")).lower()
    if "no" in satisfied or "not really" in satisfied:
        churn += 30
        flags.append("Dissatisfied")

    side_effects = str(answers.get("Any side effects from your medication this month?", "")).lower()
    if side_effects and side_effects not in ["no", "none", "nothing", "not really"]:
        churn += 25
        flags.append("Side Effects")

    risk = "HIGH" if churn > 60 else "MEDIUM" if churn > 30 else "LOW"

    return {
        "risk_level": risk,
        "churn_score": churn,
        "behavioral_profile": "Synthesizing live clinical data...",
        "edge_case_report":   "Monitoring active session telemetry...",
        "soap_note":          "Awaiting call completion...",
        "advocate_summary":   "",
        "skeptic_summary":    "",
        "arbiter_verdict":    ""
    }


# ═══════════════════════════════════════════════════════════
# SAFE FLATTEN — prevents [object Object] in UI
# ═══════════════════════════════════════════════════════════
def _flatten(value, fallback=""):
    if isinstance(value, dict):
        return "\n".join(f"[{k.upper()}]\n{v}" for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value) if value else fallback


# ═══════════════════════════════════════════════════════════
# DELIBERATION CHAMBER — 3 sequential Groq calls
# ═══════════════════════════════════════════════════════════
def run_final_llm_deliberation(transcript: list, responses: list) -> dict:
    print("🏛️  DELIBERATION CHAMBER OPENING...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    sorted_tx = sorted(transcript, key=lambda x: x.get("timestamp", 0))
    base_data = f"""
CALL TRANSCRIPT:
{json.dumps(sorted_tx, indent=2)}

STRUCTURED RESPONSES:
{json.dumps(responses, indent=2)}
"""

    # ── ROUND 1: ADVOCATE ──────────────────────────────────
    print("  ⚖️  Round 1: Advocate building case...")
    try:
        advocate_raw = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""You are a clinical patient advocate reviewing a medication check-in call.

Your job: Build the STRONGEST possible case that this patient is healthy,
compliant, engaged, and low churn risk.

Find every positive signal:
- Consistent medication adherence
- Positive attitude or engagement
- Good health indicators
- Signs of trust in the care team
- Evidence of lifestyle compliance

Return ONLY valid JSON, nothing else:
{{
  "advocate_case": "2-3 sentence strongest-case summary for this patient",
  "positive_signals": ["signal 1", "signal 2", "signal 3"],
  "recommended_risk": "LOW or MEDIUM"
}}

{base_data}"""
            }],
            temperature=0.3,
            response_format={"type": "json_object"}
        ).choices[0].message.content

        advocate = json.loads(advocate_raw)
        print(f"  ✅ Advocate: {advocate.get('recommended_risk')} risk")
    except Exception as e:
        print(f"  ❌ Advocate failed: {e}")
        advocate = {
            "advocate_case": "Insufficient data for advocate assessment.",
            "positive_signals": [],
            "recommended_risk": "MEDIUM"
        }

    # ── ROUND 2: SKEPTIC ───────────────────────────────────
    print("  ⚖️  Round 2: Skeptic building rebuttal...")
    try:
        skeptic_raw = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""You are a clinical risk analyst reviewing a medication check-in call.

The Advocate has made this case for the patient:
{json.dumps(advocate, indent=2)}

Your job: CHALLENGE this assessment. Find every red flag the Advocate dismissed.

Look for:
- Contradictions between answers (e.g., "no missed doses" but "supply almost gone")
- Hedging language ("I think", "maybe", "kind of")
- Avoided or vague answers
- Side effects that may be understated
- Signs of frustration, disengagement, or distrust
- Answers that changed when probed
- Anything clinically inconsistent

Return ONLY valid JSON, nothing else:
{{
  "skeptic_rebuttal": "2-3 sentence strongest-case for WHY this patient is at risk",
  "red_flags": ["flag 1", "flag 2", "flag 3"],
  "recommended_risk": "MEDIUM or HIGH"
}}

{base_data}"""
            }],
            temperature=0.3,
            response_format={"type": "json_object"}
        ).choices[0].message.content

        skeptic = json.loads(skeptic_raw)
        print(f"  ✅ Skeptic: {skeptic.get('recommended_risk')} risk")
    except Exception as e:
        print(f"  ❌ Skeptic failed: {e}")
        skeptic = {
            "skeptic_rebuttal": "Insufficient data for skeptic assessment.",
            "red_flags": [],
            "recommended_risk": "MEDIUM"
        }

    # ── ROUND 3: ARBITER ───────────────────────────────────
    print("  ⚖️  Round 3: Arbiter rendering final verdict...")
    try:
        arbiter_raw = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""You are the Chief Medical Officer rendering a final clinical verdict.

You have read two assessments of the same patient call:

ADVOCATE'S CASE (arguing patient is low risk):
{json.dumps(advocate, indent=2)}

SKEPTIC'S REBUTTAL (arguing patient is higher risk):
{json.dumps(skeptic, indent=2)}

Your job: Synthesize both perspectives into a BALANCED, accurate clinical verdict.
Do not simply average them. Weigh the evidence. Make a judgment call.

CRITICAL: 'soap_note' MUST be a single plain text string.
Use [S], [O], [A], [P] headers within the string. Use \\n for line breaks.
Do NOT make it a nested object.

Return ONLY valid JSON, nothing else:
{{
  "behavioral_profile": "2-3 sentence balanced clinical summary of patient mood, adherence, and engagement. Draw on BOTH the advocate and skeptic perspectives.",
  "edge_case_report": "Specific anomalies, contradictions, or edge cases detected. Write 'Standard flow. No edge cases detected.' if none.",
  "soap_note": "[S] Subjective findings...\\n[O] Objective findings...\\n[A] Assessment...\\n[P] Plan...",
  "risk_level": "LOW, MEDIUM, or HIGH",
  "churn_score": <integer 0-100>,
  "advocate_summary": "One sentence: the strongest point the Advocate made.",
  "skeptic_summary": "One sentence: the strongest point the Skeptic made.",
  "arbiter_verdict": "One sentence: your final clinical judgment on this patient."
}}

{base_data}"""
            }],
            temperature=0.2,
            response_format={"type": "json_object"}
        ).choices[0].message.content

        result = json.loads(arbiter_raw)
        print(f"  ✅ Arbiter verdict: {result.get('risk_level')} / {result.get('churn_score')}%")

    except Exception as e:
        print(f"  ❌ Arbiter failed: {e}")
        result = {
            "behavioral_profile": "Deliberation chamber error. Manual review required.",
            "edge_case_report":   "Error during synthesis.",
            "soap_note":          "Error: SOAP note generation failed.",
            "risk_level":         "UNKNOWN",
            "churn_score":        0,
            "advocate_summary":   "",
            "skeptic_summary":    "",
            "arbiter_verdict":    ""
        }

    # ── FLATTEN all fields defensively ─────────────────────
    result["soap_note"]          = _flatten(result.get("soap_note"),          "No SOAP note generated.")
    result["edge_case_report"]   = _flatten(result.get("edge_case_report"),   "Standard flow.")
    result["behavioral_profile"] = _flatten(result.get("behavioral_profile"), "No profile generated.")
    result["advocate_summary"]   = _flatten(result.get("advocate_summary"),   "")
    result["skeptic_summary"]    = _flatten(result.get("skeptic_summary"),    "")
    result["arbiter_verdict"]    = _flatten(result.get("arbiter_verdict"),    "")

    print("✅ DELIBERATION CHAMBER CLOSED. Verdict rendered.")
    return result
