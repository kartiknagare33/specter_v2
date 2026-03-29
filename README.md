# SPECTER — Clinical Intelligence Voice Agent

**VoiceHack 2026 Grand Finale Project**  
**Team: Blitzkrieg**  
**Status: Fully Functional v4 (Final Hackathon Build)**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![ElevenLabs](https://img.shields.io/badge/ElevenLabs-Conversational%20AI-orange)
![Groq](https://img.shields.io/badge/Groq-Llama%203.1-purple)

---

## 🎯 Project Overview

**SPECTER** (Surveillance, Probe-based, Evaluation, Clinical, Telemetry, Extraction, and Reasoning) is an **enterprise-grade AI voice agent** designed for medication refill check-in calls for **TrimRX** (a GLP-1 weight-loss medication company).

The agent, named **Jessica**, conducts natural full-duplex voice conversations using **ElevenLabs Conversational AI**, while a sophisticated backend clinical reasoning system (**SPECTER**) performs real-time analysis, cross-validation, and post-call deliberation.

### Key Differentiators

- Cross-validates answers in real time  
- Extracts multiple answers from incidental speech  
- Adapts interview depth based on clinical signals  
- Measures Motivational Interviewing (MI) fidelity  
- Detects hesitation/evasion patterns  
- Computes **GLP-1 Churn Risk**  
- Tracks **longitudinal risk deltas**  
- Runs a **4-round AI deliberation chamber**  
- Generates actionable **Priority Action Board**

---

## ✨ Core Innovations

1. Dual-Track Cross-Validation  
2. Adaptive Interview Depth  
3. 4-Round Deliberation Chamber  
4. Multi-Answer Extraction Engine  
5. 30-Day Adherence Forecast  
6. Priority Action Board  
7. GLP-1 Risk Matrix  
8. Longitudinal Risk Delta  
9. MI Fidelity Scoring  
10. Hesitation Fingerprint  

---

## 🏗️ System Architecture


Patient → Voice (WebRTC)
↓
ElevenLabs AI Agent (Jessica)
↓
FastAPI Backend
↓
Signal Processing + Risk Analysis
↓
Groq LLM (Deliberation)
↓
Clinical Dashboard + Dossier


---

## 📁 Project Structure

```bash
specter-v4/
├── server/
│   └── main.py
├── deliberation/
│   └── judge.py
├── demo/
│   └── index.html
├── .env
├── requirements.txt
└── README.md
🚀 Quick Start
1. Clone Repo
git clone https://github.com/yourusername/specter-v4.git
cd specter-v4
2. Setup Environment
python -m venv venv

# Activate:
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install fastapi uvicorn python-dotenv elevenlabs groq
3. Configure .env
ELEVENLABS_API_KEY=your_key
ELEVENLABS_AGENT_ID=your_agent_id
GROQ_API_KEY=your_key
4. Run Server
uvicorn server.main:app --reload --port 8000

Open: http://localhost:8000

🎮 Demo Guide
Enter sample patient data
Start call
Speak naturally
Observe live dashboard updates
End call → wait for AI analysis
📊 Dashboard Highlights
Real-time transcript
Behavioral radar (8 probes)
GLP-1 Risk Matrix
Multi-answer extraction
Hesitation fingerprint
30-day adherence forecast
SOAP notes + clinical summary
🔑 Tech Stack
Voice AI: ElevenLabs
Backend: FastAPI (Python 3.12)
LLM: Groq (Llama 3.x)
Frontend: HTML + JS + D3.js
📋 Requirements
Python 3.12+
ElevenLabs API key
Groq API key
Microphone
🏆 Why SPECTER?

SPECTER is not just a chatbot — it is a clinical intelligence system that:

Thinks like a medical professional
Validates patient responses
Detects subtle behavioral signals
Provides actionable healthcare insights
📜 License

MIT License

🙌 Team Blitzkrieg

Built for VoiceHack 2026

"Clinical Reasoning, Powered by Voice"

Version: v4 Final | March 2026
