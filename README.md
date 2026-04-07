# SPECTER — Clinical Intelligence Voice Agent
### VoiceHack 2026 · Team Blitzkrieg

[SPECTER Master Doc Link (for all info)](https://docs.google.com/document/d/1b4BnjpXAxS7vS9YRUrm1x5pAROAKvhysqrb_qub4p9k/edit?usp=sharing)
---

> **⚠️ Active Development Notice**
> We are currently working on integrating Twilio for outbound calling support, and are actively fixing other bugs and issues identified. The codebase will be updated as these are resolved.

---

## What is SPECTER?

**SPECTER** stands for **S**urveillance **P**robe-based **E**valuation **C**linical **T**elemetry **E**xtraction **R**easoning.

It is a clinical intelligence system built for [VoiceHack 2026](https://carecaller.ai), organised by CareCaller.ai. The challenge: build a voice agent that conducts medication refill check-in calls — asking 14 health questionnaire questions, handling edge cases, and producing structured patient data.

Most teams will build a chatbot. SPECTER is a **clinical reasoning system that happens to use voice.**

The voice agent patients hear is named **Jessica**. The fictional GLP-1 weight-loss medication company is **TrimRX**. The system was built and submitted solo under the team name **Team Blitzkrieg**.

---

## The One-Line Pitch

> *"While other teams built a voice agent that reacts to patients — SPECTER thinks about them before, during, and after every call."*

---

## Architecture Overview

SPECTER operates in three layers:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — BEFORE THE CALL (Pre-Call Intelligence)          │
│                                                             │
│  Patient SQLite DB → get_patient()                          │
│       ↓                                                     │
│  generate_pre_call_brief() via Groq (llama-3.1-8b-instant)  │
│       ↓                                                     │
│  Custom system prompt injected into ElevenLabs agent        │
│  via conversation_config_override before call connects.     │
│  Jessica walks in knowing the patient's history.            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — DURING THE CALL (Living Memory + Ghost Analyst)  │
│                                                             │
│  Patient speaks → ElevenLabs fires log_answer tool          │
│       ↓                                                     │
│  update_living_memory() — zero-latency rule-based routing   │
│  8 quality probes scored in real time                       │
│  5 cross-validation contradiction checks                    │
│  Multi-answer extraction catches incidental disclosures     │
│  Outcome prediction bars updated (v6)                       │
│       ↓ (async, no latency penalty)                         │
│  Ghost Analyst — parallel Groq call after every answer      │
│  One-sentence clinical observation → dashboard amber feed   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — AFTER THE CALL (5-Round Deliberation Chamber)    │
│                                                             │
│  Round 1: Advocate   — builds strongest positive case       │
│  Round 2: Skeptic    — challenges Advocate, finds risk      │
│  Round 3: Arbiter    — CMO renders final verdict            │
│  Round 4: Validator  — checks Arbiter claims vs transcript  │
│  Round 5: Strategist — writes pre-call brief for NEXT call  │
│       ↓                                                     │
│  Saved to SQLite → feeds Layer 1 of the next call.          │
│  The loop closes. Every call makes the next one smarter.    │
└─────────────────────────────────────────────────────────────┘
```

---

## What Makes SPECTER Different

### 1. It thinks before the call
Every other system is reactive. SPECTER reads the patient's full longitudinal history before the call connects, generates a clinical strategy using Groq, and injects a patient-specific system prompt into ElevenLabs. Jessica doesn't start from a blank slate — she starts knowing Elena's cost sensitivity, GI history, and missed dose pattern.

### 2. It maintains a living brain during the call
Instead of relying on raw conversation history (which degrades as the call grows), SPECTER maintains a single structured JSON object — the **Living Memory** — updated after every patient response. Jessica reads this before each question. She never loses track because the memory never loses track.

### 3. A second intelligence watches in parallel
The **Ghost Analyst** fires an async Groq call after every `log_answer`. A parallel AI watches the call in real time and generates one-sentence clinical observations. These appear on the dashboard as the patient speaks. Two AIs working simultaneously on one call.

### 4. It shows what's going to happen before it happens
Real-time **Outcome Prediction** bars update after every answer. Judges watch "completed: 72%" drop to "54%" as Elena raises the pricing concern, while "billing_concern: 5%" climbs to "36%". No other team shows live call outcome probability.

### 5. The loop closes
Round 5 of the Deliberation Chamber writes the pre-call brief for the **next** call with this patient. Layer 3 feeds Layer 1. The system gets smarter across every interaction.

---

## Feature List

### Core (v1–v4)
- ElevenLabs Conversational AI as voice backbone (Jessica)
- FastAPI backend intercepting all tool calls
- 14 primary clinical questions + 5 hidden validation probes = **19 total data points**
- 8 SPECTER quality probes (A–H) scored after every answer, visualised on D3.js hexagonal radar chart
- **Multi-Answer Extraction Engine** — detects when patient answers unasked questions incidentally (teal ◈ AUTO badge on dashboard)
- **5-Round Adversarial Deliberation Chamber** — Advocate → Skeptic → Arbiter → Validator → Strategist (Groq llama-3.1-8b-instant)
- **GLP-1 Churn Risk Matrix** — 7 evidence-anchored risk factors (JAMA 2025, EASD 2025, BHI 2024, AiCure 2024, PMC 2025)
- **Longitudinal Risk Delta** — IMPROVING / STABLE / DETERIORATING trend across calls
- **MI Fidelity Score** — Probe H, measures Motivational Interviewing quality from transcript (R:Q ratio, open question ratio, empathy events)
- **Hesitation Fingerprint** — linguistic evasion analysis post-call
- **30-Day Adherence Forecast** — probability, label, top risk factors, next contact window
- **Priority Action Board** — 3 urgency levels (URGENT / REVIEW / MONITOR)
- Full edge case handling: wrong number, opt-out, reschedule, pricing, safety escalation, AI disclosure, caregiver proxy, human transfer, emotional distress

### v5 Additions
- **Dynamic Prompt Injection** via `conversation_config_override` — custom per-patient system prompt on every call
- **Living Memory** — structured JSON brain, updated in real time, polled at 400ms
- **Ghost Analyst** — async parallel Groq track, amber alert feed on dashboard
- **Round 5** — Next-Call Strategy Generator (closes the loop)
- **Patient SQLite Database** — longitudinal call history, risk trend, known sensitivities
- **Pre-Call Brief card** on dashboard (purple)

### v6 Additions (current)
- **Structured JSON Output** — `/api/structured-output` returns all 14 Q&As in `transcript_samples.json` schema (directly satisfies the problem statement's explicit output format requirement)
- **`correct_answer` Tool (#13)** — patient self-corrections handled explicitly; amber CORRECTED badge on response grid; original vs corrected values logged
- **Live Outcome Prediction Bars** — 5 outcome probabilities normalised to 100%, updated every 400ms
- **Pricing Severity Gate** — 2-strike system: Level 1 = note and continue, Level 2+ = escalate to billing. Prevents robotic immediate escalation on first pricing keyword.

---

## File Structure

```
SPECTER-V2/
│
├── agent/
│   ├── __init__.py
│   ├── call_state.py          # CallState class + Living Memory
│   └── tools.py               # log_answer, end_call tool implementations
│
├── db/
│   ├── patient_store.py       # SQLite patient database (longitudinal)
│   └── patients.db            # Auto-created on first run
│
├── deliberation/
│   ├── __init__.py
│   └── judge.py               # 5-round adversarial deliberation chamber
│
├── probes/
│   ├── __init__.py
│   ├── probe_a_acoustic.py    # Acoustic fidelity
│   ├── probe_b_intent.py      # Intent alignment
│   ├── probe_c_coverage.py    # Coverage integrity
│   ├── probe_d_safety.py      # Safety compliance
│   ├── probe_e_identity.py    # Identity signal
│   ├── probe_f_capture.py     # Capture accuracy / contradiction detection
│   └── probe_g_behavioral.py  # Behavioral signal
│
├── demo/
│   └── index.html             # Full real-time dashboard (D3.js + Chart.js)
│
├── server/
│   └── main.py                # FastAPI server — THE CORE FILE
│
├── .env                       # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## The Dashboard

The dashboard at `http://localhost:8000` provides a real-time clinical intelligence view of the call:

| Panel | Description |
|---|---|
| **Signal Score + Radar** | Accumulated clinical signal score; 8-spoke D3.js radar chart for probes A–H (Probe H in gold = MI Fidelity) |
| **Response Grid** | 14 primary Q&A slots, colour-coded by flag. Teal ◈ = auto-extracted. Amber = CORRECTED. |
| **Pre-Call Brief** | Purple card — Groq-generated patient strategy seeded before call |
| **Living Memory** | Blue card — live JSON brain: coverage %, questions remaining, emotional state, detected flags, recommended next action, outcome prediction bars |
| **Ghost Analyst Feed** | Amber card — real-time one-sentence observations from parallel AI |
| **GLP-1 Risk Matrix** | Post-call: 7 evidence-anchored risk factors |
| **Deliberation Chamber** | Post-call: Advocate / Skeptic / Arbiter / Validator summaries |
| **30-Day Forecast** | Adherence probability, top risk factors, next contact window |
| **Priority Actions** | 3 urgency-ranked actions for the care team |
| **Structured Output** | v6: full transcript in `transcript_samples.json` schema with Export button |
| **Next-Call Strategy** | Round 5 output: opening line, open loops, risk watch flags, question priority |

---

## Prerequisites

- Python 3.11+
- An [ElevenLabs](https://elevenlabs.io) account with a Conversational AI agent configured (see ElevenLabs Configuration below)
- A [Groq](https://console.groq.com) API key
- The ElevenLabs Python SDK v1.0+

---

## Environment Variables

Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_AGENT_ID=your_agent_id_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/specter-v2.git
cd specter-v2

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your .env file (see above)
```

### requirements.txt

```
fastapi
uvicorn
python-dotenv
elevenlabs>=1.0.0
groq
pydantic
```

---

## Running the Server

```bash
# From the project root
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

On first run, the server will:
1. Initialise the SQLite database at `db/patients.db`
2. Seed the demo patient `elena_vance` if not already present
3. Print `[DB] Demo patient 'elena_vance' seeded.`
4. Start listening on `http://localhost:8000`

Open the dashboard: **`http://localhost:8000`**

---

## Starting a Call

### Via Dashboard (Recommended for Demo)
1. Open `http://localhost:8000`
2. Enter patient ID: `elena_vance`
3. Click **Start Call** — SPECTER will:
   - Load Elena's history from SQLite
   - Generate a patient-specific system prompt via Groq
   - Inject the prompt into ElevenLabs via `conversation_config_override`
   - Connect the call with Jessica
4. Speak as Elena. Watch the dashboard update in real time.

### Via API

```bash
# Start a call for a patient
curl -X POST http://localhost:8000/api/start-call \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "elena_vance"}'

# Check live call state
curl http://localhost:8000/call-state

# Get Living Memory snapshot
curl http://localhost:8000/api/memory-state

# Get structured output (v6)
curl http://localhost:8000/api/structured-output

# Get pre-call brief
curl http://localhost:8000/api/pre-call-brief

# Manually trigger end-of-call deliberation
curl -X POST http://localhost:8000/api/end-call
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the dashboard (`demo/index.html`) |
| `POST` | `/api/start-call` | Initialises call state, generates pre-call brief, connects ElevenLabs |
| `GET` | `/call-state` | Full call state snapshot (probes, responses, signal score, etc.) |
| `POST` | `/api/end-call` | Triggers post-call deliberation chamber (also called via ElevenLabs webhook) |
| `GET` | `/api/memory-state` | Living Memory JSON (polled every 400ms by dashboard) |
| `GET` | `/api/pre-call-brief` | Returns the Groq-generated pre-call brief for current patient |
| `GET` | `/api/structured-output` | Returns all 14 Q&As in `transcript_samples.json` schema |
| `GET` | `/api/insights` | Full post-call deliberation results |

---

## ElevenLabs Configuration Summary

The ElevenLabs agent (Jessica) requires these settings:

**Agent Tab:**
- Voice: Jessica, Stability 0.55, Similarity 0.80
- Max duration: 600 seconds
- Backchanneling: ON, Smart endpointing: ON, Silence threshold: 600ms
- Dynamic variables: `{{patient_name}}`, `{{pre_call_brief}}`, `{{call_number}}`

**13 Tools configured:**
1. `verify_identity` — confirms patient DOB
2. `log_answer` — logs each Q&A, triggers Living Memory update + Ghost Analyst
3. `schedule_callback` — handles reschedule requests
4. `escalate_to_pharmacist` — safety escalation
5. `capture_pricing_concern` — pricing barrier handling
6. `flag_contradiction` — cross-validation conflict detection
7. `end_call` — terminates call, triggers deliberation
8. `log_emotional_distress` — distress event logging
9. `detect_ai_inquiry` — handles "are you a bot?" questions
10. `handle_caregiver_proxy` — third-party caller handling
11. `request_human_transfer` — warm transfer trigger
12. `get_memory_state` — Jessica reads Living Memory before each question
13. `correct_answer` *(v6)* — handles patient self-corrections

**Advanced Tab:**
- Post-call webhook: `http://your-server-ip:8000/api/end-call`
- Silence timeout: 10 seconds
- Custom vocabulary: TrimRX, Semaglutide, GLP-1, Ozempic, Wegovy

---

## Demo Scenario (Test 05 — Pricing Concern)

This is the recommended scenario for judges. It exercises: pricing concern handling, empathy branching, Ghost Analyst observations, 5+ tool calls, and the full deliberation chamber — all in under 4 minutes.

**Simulated patient behaviour:**
> Answer Q1–Q4 normally as Elena Vance. When asked Q5 (satisfaction), say:
> *"Honestly not really — the medication is really expensive and I'm not sure I can keep affording it. I don't know if it's worth the cost."*

**What to watch on the dashboard:**
1. Pre-Call Brief card (purple) — populated before the call starts
2. Ghost Analyst feed (amber) — one-sentence observations appear after each answer
3. Living Memory — `pricing` flag appears in `detected_flags`, `recommended_next_action` shifts to `handoff_billing`
4. Outcome Prediction bars — `completed` drops, `billing_concern` climbs
5. After call ends — full deliberation chamber runs (5 rounds visible in console), dashboard populates with GLP-1 Risk Matrix, Priority Actions, SOAP Note, 30-Day Forecast, Next-Call Strategy

---

## The 16 Innovations At a Glance

| # | Innovation | Version |
|---|---|---|
| 1 | Dual-Track Cross-Validation (5 hidden probes) | v4 |
| 2 | Adaptive Interview Depth (standard / soft_probe / deep_probe) | v4 |
| 3 | 5-Round Adversarial Deliberation Chamber | v4 |
| 4 | Multi-Answer Extraction Engine | v4 |
| 5 | 30-Day Adherence Forecast | v4 |
| 6 | Priority Action Board + Validator Round | v4 |
| 7 | GLP-1 Churn Risk Matrix (7 evidence-anchored factors) | v4 |
| 8 | Longitudinal Risk Delta (IMPROVING/STABLE/DETERIORATING) | v4 |
| 9 | MI Fidelity Score (Probe H, gold on radar) | v4 |
| 10 | Hesitation Fingerprint | v4 |
| 11 | Dynamic Prompt Injection (pre-call Groq → ElevenLabs) | v5 |
| 12 | Living Memory (structured JSON brain) | v5 |
| 13 | Ghost Analyst (async parallel Groq track) | v5 |
| 14 | Round 5 / Next-Call Strategy (closed loop) | v5 |
| 15 | Patient SQLite DB (longitudinal tracking) | v5 |
| 16 | Structured JSON Output (`/api/structured-output`) | v6 |
| 17 | `correct_answer` Tool — patient self-corrections | v6 |
| 18 | Live Outcome Prediction Bars | v6 |
| 19 | Pricing Severity Gate (2-strike, not immediate escalation) | v6 |

---

## Known Issues & Edge Cases

- **ElevenLabs SDK version:** `conversation_config_override` requires `elevenlabs>=1.0.0`. If the call starts but the custom prompt is ignored: `pip install --upgrade elevenlabs`
- **Groq rate limits:** Ghost Analyst fires after every answer. On long calls this can hit rate limits. The `run_ghost_analyst()` function has a `try/except` that silently suppresses failures — the call continues normally.
- **Pricing flag race condition:** `capture_pricing_concern()` and `update_living_memory()` both increment `pricing_concern_level`. This means the 2-strike threshold may be reached slightly faster than expected on calls with heavy pricing language. This is intentional — it ensures billing handoff fires reliably.
- **`correct_answer` + unknown topic:** Falls back to substring matching. If no match found, the correction event is logged but the response slot is not updated. Ghost alert fires to surface the attempted correction.

---

## Built With

- [ElevenLabs Conversational AI](https://elevenlabs.io) — voice backbone
- [FastAPI](https://fastapi.tiangolo.com) — backend API server
- [Groq](https://groq.com) — LLM inference (llama-3.1-8b-instant)
- [D3.js](https://d3js.org) — hexagonal radar chart
- [Chart.js](https://chartjs.org) — outcome prediction charts
- [SQLite](https://sqlite.org) — patient database

---

## Team

**Team Blitzkrieg** — VoiceHack 2026, organised by CareCaller.ai  
Solo submission.

---

*SPECTER v6 · April 2026*
