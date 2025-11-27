"""
Day 6 – Fraud Detection Agent
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Annotated
from pydantic import Field
from datetime import datetime

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    RoomInputOptions,
    function_tool,
    cli
)

from livekit.plugins import google, deepgram, murf, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

FRAUD_DB_FILE = "shared-data/fraud_db.json"


# ---------------------------
# Helpers
# ---------------------------
def load_db():
    if not os.path.exists(FRAUD_DB_FILE):
        return []
    with open(FRAUD_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data):
    with open(FRAUD_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# User / Case state
# ---------------------------
@dataclass
class ActiveCase:
    record: Optional[dict] = None
    verified: bool = False


@dataclass
class UserState:
    active_case: ActiveCase = field(default_factory=ActiveCase)
    agent_session: Optional[AgentSession] = None


# ---------------------------
# Tools
# ---------------------------
@function_tool
async def lookup_customer(ctx: RunContext[UserState], name: Annotated[str, Field(description="Customer name")]) -> str:
    db = load_db()
    found = next((entry for entry in db if entry["userName"].lower() == name.lower()), None)

    if not found:
        return "User not found. Please ask them to check their name."

    ctx.userdata.active_case.record = found
    return f"Found user {found['userName']}. Ask for the security identifier."


@function_tool
async def verify_security(ctx: RunContext[UserState], identifier: Annotated[str, Field(description="Security identifier")]) -> str:
    case = ctx.userdata.active_case.record
    if not case:
        return "No case loaded."

    if identifier.strip() == case["securityIdentifier"]:
        ctx.userdata.active_case.verified = True
        return f"Security identifier correct. Now ask the security question: {case['securityQuestion']}"
    else:
        return "Invalid security identifier. Politely end the call."


@function_tool
async def verify_security_answer(ctx: RunContext[UserState], user_answer: Annotated[str, Field(description="Security question answer")]) -> str:
    case = ctx.userdata.active_case.record
    if not case:
        return "No case loaded."

    if user_answer.lower().strip() == case["securityAnswer"].lower().strip():
        ctx.userdata.active_case.verified = True
        return "Identity verified. Now proceed to explain the suspicious transaction."
    else:
        return "Security answer incorrect. End the call."


@function_tool
async def resolve_fraud_case(
    ctx: RunContext[UserState],
    status: Annotated[str, Field(description="'confirmed_fraud' or 'safe_transaction'")]
) -> str:
    case = ctx.userdata.active_case.record
    if not case:
        return "No case loaded."

    db = load_db()
    for entry in db:
        if entry["userName"] == case["userName"]:
            entry["case_status"] = status
            entry["resolved_at"] = datetime.utcnow().isoformat() + "Z"

    save_db(db)

    if status == "confirmed_fraud":
        return "Case marked as FRAUD. Assure user their card will be blocked and new card issued."
    else:
        return "Case marked SAFE. Transaction confirmed."

# ---------------------------
# Agent
# ---------------------------
class FraudAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a calm, professional Fraud Detection Specialist from Global Bank.

PHASE 1 – Greeting  
• Introduce yourself. Ask for the customer's full name.

PHASE 2 – Case Lookup  
• Call lookup_customer(name).

PHASE 3 – Security Verification  
• Ask for the security identifier.  
• Call verify_security(identifier).  
• If correct → Ask the stored security question.  
• Call verify_security_answer(answer).  
• If incorrect → end call politely.

PHASE 4 – Suspicious Transaction Review  
• Read the stored suspicious transaction: name, amount, time, source.  
• Ask: “Did you make this transaction?”

PHASE 5 – Handle Response  
If YES → call resolve_fraud_case("safe_transaction")  
If NO → call resolve_fraud_case("confirmed_fraud")

PHASE 6 – Close Call  
• Thank them.  
• Reassure next steps.

— VERY IMPORTANT —  
• Never ask for PIN, CVV, OTP, full card number, or any sensitive info.  
• Only use the fields provided in the database.  
"""
            ,
            tools=[lookup_customer, verify_security, verify_security_answer, resolve_fraud_case]
        )


# ---------------------------
# Prewarm
# ---------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except:
        proc.userdata["vad"] = None


# ---------------------------
# Entrypoint
# ---------------------------
async def entrypoint(ctx: JobContext):
    userdata = UserState()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Conversational"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    userdata.agent_session = session

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
