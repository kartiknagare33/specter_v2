# server/main.py

import os
import time
import threading
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools, ConversationInitiationData
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

from deliberation.judge import run_deliberation, run_final_llm_deliberation

load_dotenv()
app = FastAPI(title="TrimRX SPECTER Backend")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")

# EXACTLY ALIGNED TO THE ELEVENLABS PROMPT
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
    "Has your shipping address changed?"
]

class CallState:
    def __init__(self):
        self.responses = [{"question": q, "answer": ""} for q in QUESTIONS]
        self.outcome = None
        self.transcript = []
        self.insights = None
        self.active_conversation = None
        self.start_time = None
        self.call_duration = 0

global_state = CallState()

class CallRequest(BaseModel):
    patient_name: str
    medication: str
    patient_context: str

def run_call(patient_name: str, medication: str, patient_context: str):
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    def trigger_auto_end(reason="completed"):
        """Master Kill Switch: Severs connection and triggers Groq Llama 3.1 in background"""
        if global_state.outcome: 
            return # Prevent double execution
            
        print(f"🛑 AUTO-KILL SWITCH ACTIVATED: {reason}")
        global_state.outcome = reason

        # 1. Force sever the ElevenLabs connection
        if global_state.active_conversation:
            try: 
                global_state.active_conversation.end_session()
            except Exception: 
                pass

        # 2. Run Groq LLM asynchronously so it doesn't freeze FastAPI
        def run_llm_task():
            try:
                global_state.insights = run_final_llm_deliberation(global_state.transcript, global_state.responses)
            except Exception as e: 
                print("LLM ERROR:", e)
                
        threading.Thread(target=run_llm_task).start()

    # Tool 1: Log Answer
    def log_answer(question_topic: str = "", patient_answer: str = "", **kwargs):
        print(f"[LOG] {question_topic} -> {patient_answer}")
        
        # Robust string matching to find the correct question slot
        for item in global_state.responses:
            if question_topic.lower() in item["question"].lower() or item["question"].lower() in question_topic.lower():
                item["answer"] = patient_answer
                
        # Run fast live deliberation for the UI churn meter
        try:
            ans_dict = {i["question"]: i["answer"] for i in global_state.responses if i["answer"]}
            global_state.insights = run_deliberation(ans_dict, global_state.transcript, [], True)
        except Exception: 
            pass
            
        return "SUCCESS: Answer logged. Immediately proceed to next conversational step."

    # Tool 2: End Call
    def end_call(outcome: str = "completed", **kwargs):
        trigger_auto_end(outcome)
        return "ended"

    tools = ClientTools()
    tools.register("log_answer", log_answer)
    tools.register("end_call", end_call)

    def agent_cb(text):
        if text.strip():
            global_state.transcript.append({"role": "agent", "message": text, "timestamp": time.time()})
            
            # TRANSCRIPT LISTENER: Proactively sever the call if she speaks the goodbye phrases
            lower_text = text.lower()
            if "goodbye" in lower_text or "everything i need" in lower_text or "finish another time" in lower_text:
                trigger_auto_end("completed_via_transcript")

    def user_cb(text):
        if text.strip() and text != "...":
            global_state.transcript.append({"role": "user", "message": text, "timestamp": time.time()})

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
                "patient_name": patient_name, 
                "medication": medication, 
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

@app.get("/")
def home(): 
    return FileResponse(os.path.join(DEMO_DIR, "index.html"))

@app.get("/call-state")
def state(): 
    return {
        "responses": global_state.responses, 
        "outcome": global_state.outcome, 
        "transcript": global_state.transcript, 
        "insights": global_state.insights, 
        "call_duration": global_state.call_duration
    }

@app.post("/api/start-call")
def start(req: CallRequest, bg: BackgroundTasks):
    global global_state
    global_state = CallState()
    bg.add_task(run_call, req.patient_name, req.medication, req.patient_context)
    return {"status": "started"}

@app.post("/api/end-call")
def end():
    # If the user clicks the "Force End Call" button on the UI
    if not global_state.outcome: 
        global_state.outcome = "human_intercepted"
        
        if global_state.active_conversation:
            try: global_state.active_conversation.end_session()
            except Exception: pass
            
        def run_llm_task():
            try:
                global_state.insights = run_final_llm_deliberation(global_state.transcript, global_state.responses)
            except Exception as e: print("LLM ERROR:", e)
            
        threading.Thread(target=run_llm_task).start()
        
    return {"status": "ended"}