# deliberation/judge.py

import datetime
import os
import json
from groq import Groq

# =========================
# LIVE (FAST)
# =========================
def run_deliberation(answers: dict, transcript: list, strategy_log: list, is_live: bool = False) -> dict:
    churn = 10
    flags = []

    # Safe extraction for live churn scoring
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
        "edge_case_report": "Monitoring active session telemetry...",
        "soap_note": "Awaiting call completion..."
    }

# =========================
# FINAL (LLM)
# =========================
def run_final_llm_deliberation(transcript: list, responses: list) -> dict:
    print("🚀 SENDING TRANSCRIPT TO GROQ (LLAMA 3.1)...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Sort transcript chronologically
    sorted_transcript = sorted(transcript, key=lambda x: x.get("timestamp", 0))

    prompt = f"""
You are an expert clinical AI system for TrimRX. 
Analyze the following call transcript and the structured patient responses. 

Return STRICT JSON matching this exact schema. Do not output anything outside of the JSON.
CRITICAL: 'soap_note' MUST be a single plain text string. DO NOT make it a nested JSON object. Use \\n for line breaks.

{{
  "behavioral_profile": "A 2-to-3 sentence clinical summary of the patient's mood, adherence, and frustration levels.",
  "edge_case_report": "A short note on any interruptions, side effects, or early hang-ups. (Say 'Standard flow maintained. No edge cases detected.' if none).",
  "soap_note": "A formal, structured clinical SOAP note string based on the data gathered. Format with [S], [O], [A], [P] within this single string.",
  "risk_level": "LOW, MEDIUM, or HIGH",
  "churn_score": <integer between 0 and 100>
}}

TRANSCRIPT:
{json.dumps(sorted_transcript)}

RESPONSES:
{json.dumps(responses)}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        
        # --- CRITICAL FIX: PREVENT [object Object] IN UI ---
        # If Llama disobeys and returns a dict/object for SOAP note, flatten it into a readable string
        if isinstance(result.get("soap_note"), dict):
            formatted_soap = ""
            for k, v in result["soap_note"].items():
                formatted_soap += f"[{k.upper()}]\n{v}\n\n"
            result["soap_note"] = formatted_soap.strip()
        elif isinstance(result.get("soap_note"), list):
            result["soap_note"] = "\n".join(str(x) for x in result["soap_note"])
        else:
            result["soap_note"] = str(result.get("soap_note", "No SOAP note generated."))

        # Same failsafe for Edge Case report
        if isinstance(result.get("edge_case_report"), dict):
            formatted_edge = ""
            for k, v in result["edge_case_report"].items():
                formatted_edge += f"{k}: {v}\n"
            result["edge_case_report"] = formatted_edge.strip()
        elif isinstance(result.get("edge_case_report"), list):
            result["edge_case_report"] = "\n".join(str(x) for x in result["edge_case_report"])
        else:
            result["edge_case_report"] = str(result.get("edge_case_report", "Standard flow maintained."))
            
        print("✅ LLM RESPONSE RECEIVED & PUSHED TO UI")
        return result
    except Exception as e:
        print(f"❌ GROQ ERROR: {e}")
        return {
            "behavioral_profile": "Error: Could not synthesize final profile.",
            "edge_case_report": "Error: Audit unavailable.",
            "soap_note": "Error: SOAP note generation failed.",
            "risk_level": "UNKNOWN",
            "churn_score": 0
        }