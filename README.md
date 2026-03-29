# 🎙️ SPECTER: Autonomous Clinical Voice Agent

**SPECTER** (System for Patient Evaluation & Clinical Telehealth Routing) is an enterprise-grade, real-time Voice AI system designed to automate telehealth triage, medication refill check-ins, and clinical documentation. 

Built for modern healthcare compliance, SPECTER interacts with patients using ultra-realistic voice, dynamically adapts to conversational edge cases (interruptions, symptom reports, early hang-ups), and instantly synthesizes call transcripts into structured clinical insights.

---

## ✨ Key Features

* **Real-Time Conversational AI:** Powered by ElevenLabs, the agent (Jessica) conducts natural, human-like check-ins, strictly following a 10-point clinical questionnaire.
* **Invisible Tool Execution:** Advanced prompt engineering ensures the AI silently logs data and triggers backend actions without ever "leaking" JSON or tool syntax into the audio stream.
* **Dynamic Edge-Case Handling:** Built-in behavioral safeguards instantly override the script if a patient reports pain (injecting empathy) or needs to hang up early (graceful termination).
* **Live Telemetry Dashboard:** A glassmorphism command center featuring a real-time **D3.js Knowledge Graph**, live transcript streaming, and active churn-risk prediction.
* **Instant Clinical Synthesis:** Upon call completion, the system utilizes **Groq (Llama-3.1-8b-instant)** to instantly generate a professional Behavioral Profile, Edge Case Audit, and structured SOAP Note.
* **Structured Data Export:** 1-click export of the complete interaction dossier to strict JSON format for EHR integration.

---

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **Voice Engine:** ElevenLabs Conversational AI API
* **LLM / Reasoning Engine:** Groq API (Llama-3.1-8b-instant)
* **Frontend:** Vanilla JS, HTML/CSS (Glassmorphism UI)
* **Data Visualization:** D3.js

---

## 🚀 Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/kartiknagare33/specter_v2.git](https://github.com/kartiknagare33/specter_v2.git)
cd specter_v2
```

**2. Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install fastapi uvicorn python-dotenv elevenlabs groq pydantic
```

**4. Configure Environment Variables**
Create a `.env` file in the root directory and add your API keys:
```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_AGENT_ID=your_elevenlabs_agent_id_here
GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the Application**
```bash
uvicorn server.main:app --reload
```
Navigate to `http://localhost:8000` in your browser to access the SPECTER Command Center.

---

## 🧠 System Architecture & Workflow

1.  **Initiation:** The UI sends patient context (Name, Medication, EHR notes) to the FastAPI backend.
2.  **Voice Session:** FastAPI establishes a secure websocket with the ElevenLabs agent.
3.  **Silent Logging:** As the patient answers questions, the agent triggers a `log_answer` tool. The Python backend intercepts this, updates the UI's Knowledge Graph, and calculates live drop-off risk.
4.  **Auto-Kill Switch:** If the patient completes the questionnaire or asks to leave early, the agent speaks a goodbye phrase. The backend "listens" to the transcript and forcefully severs the connection to ensure a perfect cut-off.
5.  **LLM Deliberation:** A background thread spins up the Groq API, feeding the entire transcript to Llama 3.1 to generate the final SOAP note without freezing the main server thread.

---

## ⚠️ Hackathon Note: Prompt Engineering & Tool Safety
A major technical achievement of this project is the **sterilization of tool calls**. LLMs frequently hallucinate and read JSON code out loud when executing background tools. SPECTER mitigates this by abstracting code-syntax completely out of the system prompt and enforcing strict "silent execution" directives within the individual tool descriptions, resulting in a flawless, human-only audio stream.
