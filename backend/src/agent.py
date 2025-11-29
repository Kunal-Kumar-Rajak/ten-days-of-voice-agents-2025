import logging
import uuid
import random
import copy
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
# Logging
# -------------------------
logger = logging.getLogger("voice_game_master")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")

# -------------------------
# Game World Template
# -------------------------
WORLD_TEMPLATE = {
    "intro": {
        "title": "A Shadow over Brinmere",
        "desc": (
            "You awake on the damp shore of Brinmere. A ruined watchtower smolders inland. "
            "Beside you in the sand is a locked wooden box."
        ),
        "choices": {
            "inspect_box": {
                "desc": "Inspect the wooden box.",
                "result_scene": "box",
            },
            "force_box": {
                "desc": "Smash the box open (Strength Check DC 12).",
                "type": "check",
                "stat": "str",
                "dc": 12,
                "result_success": "box_smashed",
                "result_fail": "box_hurt",
            },
            "approach_tower": {
                "desc": "Head to the watchtower.",
                "result_scene": "tower",
            },
        },
    },
    "box": {
        "title": "The Wooden Box",
        "desc": "It is delicate but locked. You hear a rattle inside.",
        "choices": {
            "leave_box": {"desc": "Leave it.", "result_scene": "intro"},
            "smash_it": {
                "desc": "Smash it against a rock (Strength DC 10).",
                "type": "check",
                "stat": "str",
                "dc": 10,
                "result_success": "box_smashed",
                "result_fail": "box_hurt"
            }
        }
    },
    "box_smashed": {
        "title": "Splinters",
        "desc": "The wood shatters! Inside you find a Rusty Dagger.",
        "choices": {
            "take_dagger": {
                "desc": "Take the dagger.",
                "result_scene": "intro",
                "effects": {"add_inventory": "rusty_dagger", "add_journal": "Found a rusty dagger."}
            }
        }
    },
    "box_hurt": {
        "title": "Ouch",
        "desc": "The box is tougher than it looks. You hurt your hand hitting it.",
        "choices": {
            "nurse_hand": {
                "desc": "Back away.",
                "result_scene": "intro",
                "effects": {"damage": 2}
            }
        }
    },
    "tower": {
        "title": "The Watchtower",
        "desc": "The tower looms above. An iron hatch is rusted shut at the base.",
        "choices": {
            "pry_hatch": {
                "desc": "Pry the hatch open with the dagger.",
                "req_item": "rusty_dagger",
                "result_scene": "cellar"
            },
            "kick_hatch": {
                "desc": "Kick the hatch (Strength DC 15).",
                "type": "check",
                "stat": "str",
                "dc": 15,
                "result_success": "cellar",
                "result_fail": "foot_injury"
            },
            "retreat": {
                "desc": "Go back to shore.",
                "result_scene": "intro"
            }
        },
    },
    "foot_injury": {
        "title": "Solid Iron",
        "desc": "You kick the iron hatch with all your might. It doesn't budge, but your ankle cracks.",
        "choices": {
            "limp_back": {
                "desc": "Limp back to start.",
                "result_scene": "intro",
                "effects": {"damage": 4}
            }
        }
    },
    "cellar": {
        "title": "The Cellar",
        "desc": "You are in. It's dark and smells of ozone. You find a potion.",
        "choices": {
            "drink_potion": {
                "desc": "Drink the glowing red liquid.", 
                "result_scene": "cellar", 
                "effects": {"heal": 5, "add_journal": "Drank a strange potion."}
            },
            "acrobatics": {
                "desc": "Do a backflip (Dexterity DC 12) just to show off.",
                "type": "check",
                "stat": "dex",
                "dc": 12,
                "result_success": "cellar",
                "result_fail": "cellar", # Flavour check
                "effects": {"add_journal": "Did a cool backflip."}
            },
            "restart": {"desc": "Restart adventure.", "result_scene": "intro", "effects": {"restart": True}}
        }
    },
    "game_over": {
        "title": "Death",
        "desc": "Your vision fades to black. You have perished.",
        "choices": {
            "restart": {"desc": "Resurrect (Restart)", "result_scene": "intro", "effects": {"restart": True}}
        }
    }
}

# -------------------------
# User Data
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    current_scene: str = "intro"
    hp: int = 10
    max_hp: int = 10
    
    # RPG Stats
    str_mod: int = 1  # Strength Modifier (+1 bonus)
    dex_mod: int = 0  # Dexterity Modifier
    
    # Dynamic World State 
    world_state: Dict = field(default_factory=dict)
    
    history: List[Dict] = field(default_factory=list)
    journal: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

# -------------------------
# Helpers
# -------------------------

def scene_text(scene_key: str, userdata: Userdata) -> str:
    """
    Builds scene description.
    IMPROVEMENT: Adds hints for hidden items instead of hard locking.
    """
    # Use the Dynamic World State, not the static template
    scene = userdata.world_state.get(scene_key)
    if not scene:
        return "Void. What do you do?"

    if userdata.hp <= 0 and scene_key != "game_over":
        userdata.current_scene = "game_over"
        return scene_text("game_over", userdata)

    desc = f"{scene['desc']}\n(HP: {userdata.hp}/{userdata.max_hp})\n\n"
    
    available_choices = []
    hidden_hints = []

    for cid, cmeta in scene.get("choices", {}).items():
        # Check requirements
        if "req_item" in cmeta and cmeta["req_item"] not in userdata.inventory:
            # Add a subtle hint instead of the choice
            hidden_hints.append(f"You see a path ({cid}) that requires a {cmeta['req_item']}.")
            continue
        
        available_choices.append(f"- {cmeta['desc']} (ID: {cid})")

    if hidden_hints:
        desc += "Hints: " + " ".join(hidden_hints) + "\n\n"
        
    desc += "Options:\n" + "\n".join(available_choices)
    desc += "\n\nWhat do you do?"
    return desc

def apply_effects(effects: dict, userdata: Userdata) -> str:
    msg_parts = []
    if not effects: return ""

    if "damage" in effects:
        dmg = int(effects["damage"])
        userdata.hp -= dmg
        msg_parts.append(f"Took {dmg} damage.")
    
    if "heal" in effects:
        heal = int(effects["heal"])
        userdata.hp = min(userdata.max_hp, userdata.hp + heal)
        msg_parts.append(f"Healed {heal} HP.")
    
    if "add_inventory" in effects:
        item = effects["add_inventory"]
        if item not in userdata.inventory:
            userdata.inventory.append(item)
            msg_parts.append(f"Obtained: {item}.")
    
    if "add_journal" in effects:
        if effects["add_journal"] not in userdata.journal:
            userdata.journal.append(effects["add_journal"])
    
    if "restart" in effects:
        userdata.hp = userdata.max_hp
        userdata.inventory = []
        userdata.journal = []
        userdata.world_state = copy.deepcopy(WORLD_TEMPLATE)
        userdata.current_scene = "intro"
        msg_parts.append("World reset.")

    userdata.hp = max(0, min(userdata.max_hp, userdata.hp))
    if userdata.hp <= 0:
        userdata.current_scene = "game_over"
        msg_parts.append("You have fallen.")

    return " ".join(msg_parts)

# -------------------------
# Tools
# -------------------------

@function_tool
async def start_adventure(
    ctx: RunContext[Userdata],
    player_name: Annotated[Optional[str], Field(description="Player name")] = None,
) -> str:
    userdata = ctx.userdata
    if player_name: userdata.player_name = player_name
    
    # Initialize World State dynamically
    userdata.world_state = copy.deepcopy(WORLD_TEMPLATE)
    userdata.current_scene = "intro"
    userdata.hp = 10
    userdata.inventory = []
    
    return f"Welcome {userdata.player_name or 'Adventurer'}.\n{scene_text('intro', userdata)}"

@function_tool
async def perform_action(
    ctx: RunContext[Userdata],
    choice_id: Annotated[str, Field(description="The ID of the choice to take.")],
) -> str:
    userdata = ctx.userdata
    current_key = userdata.current_scene
    scene = userdata.world_state.get(current_key) # Use dynamic state

    if not scene or choice_id not in scene.get("choices", {}):
        return "You cannot do that here. " + scene_text(current_key, userdata)

    choice = scene["choices"][choice_id]

    # Validate Requirements
    if "req_item" in choice and choice["req_item"] not in userdata.inventory:
        return f"Locked. You need: {choice['req_item']}."

    outcome_desc = ""
    next_scene_key = current_key

    # Handle Dice Rolls with Stats
    if choice.get("type") == "check":
        roll = random.randint(1, 20)
        stat = choice.get("stat", "")
        # Fetch modifier from Userdata (e.g., str_mod, dex_mod)
        modifier = getattr(userdata, f"{stat}_mod", 0) if stat else 0
        total = roll + modifier
        dc = int(choice.get("dc", 10))

        # Log details for history/debugging
        log_entry = {
            "time": datetime.utcnow().isoformat(),
            "action": choice_id,
            "roll": roll,
            "mod": modifier,
            "total": total,
            "dc": dc,
            "result": "success" if total >= dc else "fail"
        }
        userdata.history.append(log_entry)

        if total >= dc:
            outcome_desc = f"[Rolled {roll} + {modifier} ({stat.upper()}) = {total} vs DC {dc}] Success!"
            next_scene_key = choice.get("result_success", current_key)
        else:
            outcome_desc = f"[Rolled {roll} + {modifier} ({stat.upper()}) = {total} vs DC {dc}] Failed."
            next_scene_key = choice.get("result_fail", current_key)
    else:
        next_scene_key = choice.get("result_scene", current_key)

    userdata.current_scene = next_scene_key
    effect_msg = apply_effects(choice.get("effects", {}), userdata)

    if userdata.hp <= 0:
        return f"{outcome_desc} {effect_msg}\n\n{scene_text('game_over', userdata)}"

    return f"{outcome_desc} {effect_msg}\n\n{scene_text(next_scene_key, userdata)}"

@function_tool
async def show_journal(ctx: RunContext[Userdata]) -> str:
    u = ctx.userdata
    return f"HP: {u.hp}\nStats: STR+{u.str_mod}, DEX+{u.dex_mod}\nInv: {u.inventory}\nHistory: {len(u.history)} actions."

# -------------------------
# Agent
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a D&D Game Master. Use 'perform_action' to execute user choices. If they fail a check, narrate the failure dramatically.",
            tools=[start_adventure, perform_action, show_journal],
        )

# -------------------------
# Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    try: proc.userdata["vad"] = silero.VAD.load()
    except: pass

async def entrypoint(ctx: JobContext):
    logger.info("🚀 STARTING RPG MASTER")
    userdata = Userdata()
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-marcus", style="Conversational"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )
    await session.start(agent=GameMasterAgent(), room=ctx.room, room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()))
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
