"""
Day 10 – Voice Improv Battle

This agent acts as a host for a voice-only improv game.
Updates:
- Safe environment variable parsing.
- Robust exception logging in prewarm and entrypoint.
- Secure randomness for reaction logic.
- Graceful handling of empty input.
- Signal handling for cleanup.
"""

import logging
import os
import asyncio
import uuid
import secrets
import re
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Configuration & Logging
# -------------------------
load_dotenv(".env.local")

logger = logging.getLogger("voice_improv_battle")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# 1. Safe Config Parsing
try:
    DEFAULT_MAX_ROUNDS = int(os.getenv("IMPROV_MAX_ROUNDS", "3"))
except (TypeError, ValueError):
    logger.warning("Invalid IMPROV_MAX_ROUNDS env var; defaulting to 3.")
    DEFAULT_MAX_ROUNDS = 3

MAX_ROUNDS_CAP = 8

# -------------------------
# Constants & Patterns
# -------------------------
SCENARIOS = [
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are an overenthusiastic TV infomercial host selling a product that clearly does not work as advertised.",
    "You are an astronaut who just discovered the ship's coffee machine has developed a personality.",
    "You are a nervous wedding officiant who keeps getting the couple's names mixed up in ridiculous ways.",
    "You are a ghost trying to give a performance review to a living employee.",
    "You are a medieval king reacting to a very modern delivery service showing up at court.",
    "You are a detective interrogating a suspect who only answers in awkward metaphors."
]

# Precompile end-scene regex
END_SCENE_PAT = re.compile(r"\b(?:" + "|".join([re.escape(x) for x in [
    "end scene", "and scene", "scene end", "stop scene", "that's the scene",
]]) + r")\b", flags=re.IGNORECASE)

# -------------------------
# State Management
# -------------------------
def _now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

@dataclass
class Userdata:
    player_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=_now_ts)
    improv_state: Dict = field(default_factory=lambda: {
        "current_round": 0,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "rounds": [],
        "phase": "idle", # "intro", "awaiting_improv", "reacting", "done"
        "used_indices": [],
        "current_scenario": None
    })
    history: List[Dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

# -------------------------
# Helpers
# -------------------------
def _sanitize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def _pick_scenario(userdata: Userdata) -> str:
    used = userdata.improv_state.get("used_indices", [])
    candidates = [i for i in range(len(SCENARIOS)) if i not in used]
    
    if not candidates:
        userdata.improv_state["used_indices"] = []
        candidates = list(range(len(SCENARIOS)))
    
    idx = secrets.choice(candidates)
    userdata.improv_state["used_indices"].append(idx)
    scenario = SCENARIOS[idx]
    userdata.improv_state["current_scenario"] = scenario
    return scenario

def _clean_performance_text(text: str) -> str:
    if not text:
        return ""
    cleaned = END_SCENE_PAT.sub("", text)
    cleaned = _sanitize_text(cleaned)
    MAX_PERF_CHARS = 2000
    if len(cleaned) > MAX_PERF_CHARS:
        cleaned = cleaned[:MAX_PERF_CHARS-3] + "..."
    return cleaned

def _generate_reaction(performance: str) -> str:
    # 5. Empty Input Guard
    if not performance or not performance.strip():
        return "Nice start — I didn't catch many words there. Try leaning into one clear choice next time."

    p_lower = performance.lower()
    
    # Simple Vibe Detection
    vibe = "neutral"
    if any(w in p_lower for w in ["haha", "lol", "funny", "joke", "laugh"]):
        vibe = "comedic"
    elif any(w in p_lower for w in ["sad", "cry", "sorry", "tragic", "tears"]):
        vibe = "dramatic"
    elif any(w in p_lower for w in ["wait", "um", "uh", "pause", "..."]):
        vibe = "hesitant"
    elif len(performance.split()) < 5:
        vibe = "short"

    # 7. Secure Randomness
    r = secrets.randbelow(100) / 100.0
    
    # 50% Supportive
    if r < 0.5:
        if vibe == "comedic":
            return "Ha! That was genuinely funny. Great timing on the punchline."
        elif vibe == "dramatic":
            return "Wow — you actually made that surprisingly emotional. Good commitment."
        elif vibe == "short":
            return "Short and punchy — works well!"
        else:
            return "I love the energy — clear choices and commitment."
            
    # 30% Neutral/Witty
    elif r < 0.8:
        if vibe == "hesitant":
            return "I could feel the gears turning there — interesting choices."
        elif vibe == "comedic":
            return "A bit absurd, and I think that helped this scene."
        else:
            return "Interesting take — I didn't expect that turn."
            
    # 20% Constructive Critique
    else:
        if vibe == "short":
            return "That felt a little rushed — let the moment breathe more next time."
        elif vibe == "hesitant":
            return "You seemed unsure in places — trust your first impulse."
        else:
            return "Good start; try raising the stakes more for clearer beats."

def _build_summary(userdata: Userdata) -> str:
    rounds = userdata.improv_state.get("rounds", [])
    if not rounds:
        return "No rounds were played. Maybe next time!"
        
    lines = [f"Thanks for playing, {userdata.player_name or 'Contestant'}! Quick recap:"]
    
    for r in rounds:
        perf = (r.get("performance") or "").strip()
        perf_snip = perf if len(perf) <= 80 else perf[:77] + "..."
        lines.append(f"Round {r['round']}: {r['scenario']} — You: '{perf_snip}' | Host: {r['reaction']}")
        
    long_scenes = sum(1 for r in rounds if len((r.get("performance") or "").split()) > 20)
    
    if long_scenes == len(rounds):
        profile = "verbose and detailed"
    elif long_scenes == 0:
        profile = "concise and punchy"
    else:
        profile = "balanced"
        
    lines.append(f"You performed as a {profile} improviser. Keep leaning into clear choices.")
    userdata.history.append({"time": _now_ts(), "action": "summarize_show"})
    
    return "\n".join(lines)

# -------------------------
# Agent Tools
# -------------------------

@function_tool
async def start_show(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Player name", default=None)] = None,
    max_rounds: Annotated[int, Field(description="Number of rounds (1-8)", default=3)] = 3,
) -> str:
    """Initialize the game state, introduce the host, and start Round 1."""
    userdata = ctx.userdata
    async with userdata.lock:
        if name:
            userdata.player_name = _sanitize_text(name)
        userdata.player_name = userdata.player_name or "Contestant"
        
        try:
            m_rounds = int(max_rounds)
        except Exception:
            m_rounds = DEFAULT_MAX_ROUNDS
        m_rounds = max(1, min(m_rounds, MAX_ROUNDS_CAP))
        
        userdata.improv_state.update({
            "max_rounds": m_rounds,
            "current_round": 1,
            "rounds": [],
            "phase": "awaiting_improv",
            "used_indices": []
        })
        
        scenario = _pick_scenario(userdata)
        userdata.history.append({"time": _now_ts(), "action": "start_show", "round": 1, "scenario": scenario})
        
        intro_text = (
            f"Welcome to Improv Battle! I'm your host. "
            f"{userdata.player_name}, we'll play {m_rounds} rounds. "
            "When you are done with a scene say 'End scene' or pause. "
            f"Round 1: {scenario}. Action!"
        )
        logger.info("Session %s started for %s (rounds=%d)", userdata.session_id, userdata.player_name, m_rounds)
        return intro_text

@function_tool
async def next_scenario(ctx: RunContext[Userdata]) -> str:
    """Moves to the next round if available, or ends the show."""
    userdata = ctx.userdata
    async with userdata.lock:
        state = userdata.improv_state
        if state["phase"] == "done":
            return "The show is already over. Say 'start show' to play again."
            
        cur = int(state.get("current_round", 0))
        maxr = int(state.get("max_rounds", DEFAULT_MAX_ROUNDS))
        
        if cur >= maxr:
            state["phase"] = "done"
            return _build_summary(userdata)
            
        next_round = cur + 1
        scenario = _pick_scenario(userdata)
        state["current_round"] = next_round
        state["phase"] = "awaiting_improv"
        
        userdata.history.append({"time": _now_ts(), "action": "next_scenario", "round": next_round, "scenario": scenario})
        logger.info("Session %s advancing to round %d", userdata.session_id, next_round)
        
        return f"Round {next_round}: {scenario}. Go!"

@function_tool
async def record_performance(
    ctx: RunContext[Userdata],
    performance: Annotated[str, Field(description="The user's acted out speech")],
) -> str:
    """
    Saves the user's performance, generates a host reaction, 
    and checks if the game should end or continue.
    """
    userdata = ctx.userdata
    async with userdata.lock:
        state = userdata.improv_state
        perf = _clean_performance_text(performance)
        round_num = int(state.get("current_round", 0))
        scenario = state.get("current_scenario") or "(unknown)"
        
        reaction = _generate_reaction(perf)
        
        state["rounds"].append({
            "round": round_num,
            "scenario": scenario,
            "performance": perf,
            "reaction": reaction,
            "ts": _now_ts()
        })
        
        state["phase"] = "reacting"
        userdata.history.append({"time": _now_ts(), "action": "record_performance", "round": round_num})
        logger.info("Session %s recorded performance for round %d (len=%d)", userdata.session_id, round_num, len(perf))
        
        if round_num >= int(state.get("max_rounds", DEFAULT_MAX_ROUNDS)):
            state["phase"] = "done"
            summary = _build_summary(userdata)
            return f"{reaction}\n\nThat was the final round! {summary}"
            
        return f"{reaction}\n\nWhen you're ready say 'Next' or 'Ready'."

@function_tool
async def summarize_show(ctx: RunContext[Userdata]) -> str:
    """Generates a final summary of the player's style."""
    userdata = ctx.userdata
    async with userdata.lock:
        return _build_summary(userdata)

@function_tool
async def stop_show(
    ctx: RunContext[Userdata],
    confirm: Annotated[bool, Field(description="Must be True to exit")] = False
) -> str:
    """Ends the game early."""
    userdata = ctx.userdata
    async with userdata.lock:
        if not confirm:
            return "Are you sure you want to end the show early? Say 'Yes' to confirm."
        
        userdata.improv_state["phase"] = "done"
        userdata.history.append({"time": _now_ts(), "action": "stop_show"})
        logger.info("Session %s stopped by user", userdata.session_id)
        return "Okay — show ended. Thanks for playing!"

@function_tool
async def get_game_state(ctx: RunContext[Userdata]) -> str:
    """Debug tool for the LLM to check current round/phase."""
    s = ctx.userdata.improv_state
    return f"Round: {s['current_round']}/{s['max_rounds']}, Phase: {s['phase']}, Scenario: {s.get('current_scenario')}"

# -------------------------
# Agent Definition
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        instructions = """
        You are the host of "Improv Battle", a fast-paced voice-only improv game show.
        
        YOUR ROLE:
        - High-energy, witty, slightly sarcastic but supportive TV host.
        - Your goal is to keep the show moving briskly.
        
        FLOW:
        1. **Intro**: Use `start_show` to welcome the player and give the first scenario.
        2. **Listening**: When the player acts, wait for them to finish. 
           - If they say "End Scene", call `record_performance`.
           - If they pause for a long time, call `record_performance`.
        3. **Reacting**: `record_performance` will return your reaction text. Read it with character.
        4. **Next**: Ask if they are ready, then call `next_scenario`.
        5. **End**: `summarize_show` is called automatically after the last round.
        
        CRITICAL RULES:
        - DO NOT make up your own scenarios. ALWAYS use the tools.
        - DO NOT interrupt the player while they are acting.
        - If the player says "Stop" or "Quit", use `stop_show`.
        """
        super().__init__(
            instructions=instructions,
            tools=[
                start_show, 
                next_scenario, 
                record_performance, 
                summarize_show, 
                stop_show,
                get_game_state
            ],
        )

# -------------------------
# Main Execution
# -------------------------
def prewarm(proc: JobProcess):
    # 2. Log exceptions in prewarm
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD successfully prewarmed")
    except Exception as e:
        logger.exception("VAD prewarm failed; continuing without preloaded VAD: %s", e)

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("🚀 STARTING IMPROV BATTLE HOST AGENT")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3")),
        llm=google.LLM(model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")),
        tts=murf.TTS(
            voice=os.getenv("MURF_VOICE", "en-US-marcus"),
            style=os.getenv("MURF_STYLE", "Conversational"),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # 3. Wrap session.start in try/except
    try:
        await session.start(
            agent=GameMasterAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
        )
    except Exception as e:
        logger.exception("Failed to start session: %s", e)
        raise

    await ctx.connect()

# 6. Basic shutdown cleanup logger
def _on_shutdown(*args):
    logger.info("Worker shutting down: cleaning up resources...")
    # Add any specific cleanup logic here if needed (e.g., closing DB connections)
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, _on_shutdown)
    signal.signal(signal.SIGTERM, _on_shutdown)
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
