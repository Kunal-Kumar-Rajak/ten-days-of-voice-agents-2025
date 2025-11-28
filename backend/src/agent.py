import json
import logging
import os
import sqlite3
import uuid
import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Annotated

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
logger = logging.getLogger("food_agent_sqlite")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Config
# -------------------------
DB_FILE = "order_db.sqlite"
SIM_STATUS_INTERVAL = int(os.getenv("SIM_STATUS_INTERVAL_S", "5"))
SAVE_ORDER_JSON = os.getenv("SAVE_ORDER_JSON", "1") != "0"  # default: save JSON

# -------------------------
# DB helpers & concurrency
# -------------------------
DB_WRITE_LOCK = threading.Lock()

def get_db_path() -> str:
    """Return absolute path for the DB file. If __file__ is not defined (interactive), fall back to cwd."""
    try:
        base = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        base = os.getcwd()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, DB_FILE)


def get_conn():
    """Return a new sqlite3 connection. Each thread/task should get its own connection."""
    path = get_db_path()
    conn = sqlite3.connect(path, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    # Optional pragmas for WAL and busy timeout to reduce locking issues
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout = 3000;")  # ms
    except Exception:
        pass
    return conn


def _save_order_json_file(order_obj: dict):
    try:
        if not SAVE_ORDER_JSON:
            return
        d = os.path.dirname(get_db_path())
        os.makedirs(d, exist_ok=True)
        filename = os.path.join(d, f"order_{order_obj['order_id']}.json")
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(order_obj, fh, default=str, indent=2)
        logger.info("Saved order JSON: %s", filename)
    except Exception:
        logger.exception("Failed to save order JSON")


def seed_database():
    """Create tables and seed the Indian catalog if empty. Also creates helpful indexes."""
    try:
        with DB_WRITE_LOCK:
            with get_conn() as conn:
                cur = conn.cursor()
                # Create catalog table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS catalog (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        category TEXT,
                        price REAL NOT NULL,
                        brand TEXT,
                        size TEXT,
                        units TEXT,
                        tags TEXT
                    )
                """)
                # Orders table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        total REAL,
                        customer_name TEXT,
                        address TEXT,
                        status TEXT DEFAULT 'received',
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
                # Order items
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS order_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT,
                        item_id TEXT,
                        name TEXT,
                        unit_price REAL,
                        quantity INTEGER,
                        notes TEXT,
                        FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE CASCADE
                    )
                """)
                # Indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(LOWER(customer_name));")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);")

                # Seed catalog if empty
                cur.execute("SELECT COUNT(1) FROM catalog")
                if cur.fetchone()[0] == 0:
                    catalog = [
                        ("milk-Sudha-1l", "Sudha Milk", "Dairy", 59.00, "Sudha", "1L", "pack", json.dumps(["dairy", "essential"])),
                        ("paneer-200g", "Sudha Paneer", "Dairy", 80.00, "Sudha", "200g", "pack", json.dumps(["dairy", "protein", "veg"])),
                        ("butter-100g", "Sudha Butter", "Dairy", 55.00, "Sudha", "100g", "pack", json.dumps(["dairy"])),
                        ("curd-400g", "Sudha Dairy Dahi", "Dairy", 45.00, "Sudha Dairy", "400g", "cup", json.dumps(["dairy"])),
                        ("atta-5kg", "Aashirvaad Whole Wheat Atta", "Staples", 250.00, "Aashirvaad", "5kg", "bag", json.dumps(["flour", "roti"])),
                        ("rice-basmati-1kg", "India Gate Basmati Rice", "Staples", 150.00, "India Gate", "1kg", "bag", json.dumps(["rice", "premium"])),
                        ("dal-toor-1kg", "Tata Sampann Toor Dal", "Staples", 175.00, "Tata", "1kg", "pack", json.dumps(["protein", "dal"])),
                        ("salt-1kg", "Tata Salt", "Staples", 25.00, "Tata", "1kg", "pack", json.dumps(["essential"])),
                        ("sugar-1kg", "Madhur Sugar", "Staples", 55.00, "Madhur", "1kg", "pack", json.dumps(["sweet"])),
                        ("maggi-masala", "Maggi Noodles", "Instant Food", 15.00, "Nestle", "70g", "pack", json.dumps(["snack", "noodles"])),
                        ("biscuits-marie", "Britannia Marie Gold", "Snacks", 35.00, "Britannia", "250g", "pack", json.dumps(["tea-time"])),
                        ("chips-lays", "Lays Magic Masala", "Snacks", 20.00, "Lays", "50g", "pack", json.dumps(["snack", "spicy"])),
                        ("tea-250g", "Red Label Tea", "Beverages", 150.00, "Brooke Bond", "250g", "pack", json.dumps(["chai", "tea"])),
                        ("potato-1kg", "Fresh Potatoes", "Vegetables", 50.00, "", "1kg", "kg", json.dumps(["veg"])),
                        ("onion-1kg", "Fresh Onions", "Vegetables", 35.00, "", "1kg", "kg", json.dumps(["veg"])),
                        ("tomato-1kg", "Fresh Tomatoes", "Vegetables", 30.00, "", "1kg", "kg", json.dumps(["veg"])),
                        ("ginger-100g", "Fresh Ginger", "Vegetables", 15.00, "", "100g", "g", json.dumps(["veg", "chai"])),
                    ]
                    cur.executemany("""
                        INSERT INTO catalog (id, name, category, price, brand, size, units, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, catalog)
                    conn.commit()
                    logger.info(f"✅ Seeded Indian catalog into {get_db_path()}")
    except Exception:
        logger.exception("Failed to seed database: %s", get_db_path())
        raise

# Seed DB on import/run
seed_database()

# -------------------------
# In-memory per-session cart
# -------------------------
@dataclass
class CartItem:
    item_id: str
    name: str
    unit_price: float
    quantity: int = 1
    notes: str = ""

@dataclass
class Userdata:
    cart: List[CartItem] = field(default_factory=list)
    customer_name: Optional[str] = None

# -------------------------
# DB Helpers (use context managers)
# -------------------------
def find_catalog_item_by_id_db(item_id: str) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM catalog WHERE LOWER(id) = LOWER(?) LIMIT 1", (item_id,))
        row = cur.fetchone()
    if not row:
        return None
    record = dict(row)
    try:
        record["tags"] = json.loads(record.get("tags") or "[]")
    except Exception:
        record["tags"] = []
    return record

def search_catalog_by_name_db(query: str) -> List[dict]:
    q = f"%{query.lower()}%"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM catalog
            WHERE LOWER(name) LIKE ? OR LOWER(tags) LIKE ?
            LIMIT 50
        """, (q, q))
        rows = cur.fetchall()
    results = []
    for r in rows:
        rec = dict(r)
        try:
            rec["tags"] = json.loads(rec.get("tags") or "[]")
        except Exception:
            rec["tags"] = []
        results.append(rec)
    return results

def insert_order_db(order_id: str, timestamp: str, total: float, customer_name: str, address: str, status: str, items: List[CartItem]):
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    order_obj = {
        "order_id": order_id,
        "timestamp": timestamp,
        "total": total,
        "customer_name": customer_name,
        "address": address,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "items": [ci.__dict__ for ci in items],
    }
    try:
        with DB_WRITE_LOCK:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO orders (order_id, timestamp, total, customer_name, address, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (order_id, timestamp, total, customer_name, address, status, now, now))
                for ci in items:
                    cur.execute("""
                        INSERT INTO order_items (order_id, item_id, name, unit_price, quantity, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (order_id, ci.item_id, ci.name, ci.unit_price, ci.quantity, ci.notes))
                conn.commit()
        # optional JSON export for debugging / audit
        _save_order_json_file(order_obj)
    except Exception:
        logger.exception("insert_order_db failed for %s", order_id)
        raise

def get_order_db(order_id: str) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM orders WHERE order_id = ? LIMIT 1", (order_id,))
        o = cur.fetchone()
        if not o:
            return None
        order = dict(o)
        cur.execute("SELECT * FROM order_items WHERE order_id = ?", (order_id,))
        items = [dict(r) for r in cur.fetchall()]
    order["items"] = items
    return order

def list_orders_db(limit: int = 10, customer_name: Optional[str] = None) -> List[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        if customer_name:
            cur.execute("SELECT * FROM orders WHERE LOWER(customer_name) = LOWER(?) ORDER BY created_at DESC LIMIT ?", (customer_name, limit))
        else:
            cur.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    return rows

def update_order_status_db(order_id: str, new_status: str) -> bool:
    try:
        with DB_WRITE_LOCK:
            with get_conn() as conn:
                cur = conn.cursor()
                now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                cur.execute("UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?", (new_status, now, order_id))
                changed = cur.rowcount
                conn.commit()
        return changed > 0
    except Exception:
        logger.exception("Failed to update order status %s -> %s", order_id, new_status)
        return False

# -------------------------
# Logic helpers
# -------------------------
RECIPE_MAP = {
    "chai": ["milk-Sudha-1l", "tea-250g", "sugar-1kg", "ginger-100g"],
    "paneer butter masala": ["paneer-200g", "butter-100g", "tomato-1kg"],
    "maggi": ["maggi-masala"],
    "dal chawal": ["dal-toor-1kg", "rice-basmati-1kg"],
}

import re
_NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

def _parse_servings_from_text(text: str) -> int:
    text = (text or "").lower()
    m = re.search(r"for\s+(\d+)\s*(?:people|person|servings)?", text)
    if m:
        try:
            return max(1, int(m.group(1)))
        except Exception:
            pass
    for word, num in _NUMBER_WORDS.items():
        if f"for {word}" in text:
            return num
    return 1

def _infer_items_from_tags(query: str, max_results: int = 6) -> List[str]:
    words = re.findall(r"\w+", (query or "").lower())
    found = []
    with get_conn() as conn:
        cur = conn.cursor()
        for w in words:
            if len(found) >= max_results:
                break
            q = f"%\"{w}\"%"
            cur.execute("SELECT * FROM catalog WHERE LOWER(tags) LIKE ? OR LOWER(name) LIKE ? LIMIT 10", (q, f"%{w}%"))
            rows = cur.fetchall()
            for r in rows:
                rid = r["id"]
                if rid not in found:
                    found.append(rid)
                    if len(found) >= max_results:
                        break
    return found

STATUS_FLOW = ["received", "confirmed", "shipped", "out_for_delivery", "delivered"]

async def simulate_delivery_flow(order_id: str):
    """
    Background task: automatically advances order status every SIM_STATUS_INTERVAL seconds.
    Flow: received -> confirmed -> shipped -> out_for_delivery -> delivered
    """
    logger.info(f"🔄 [Simulation] Started tracking simulation for {order_id}")
    await asyncio.sleep(SIM_STATUS_INTERVAL)

    for next_status in STATUS_FLOW[1:]:
        curr_order = get_order_db(order_id)
        if curr_order and curr_order.get("status") == "cancelled":
            logger.info(f"🛑 [Simulation] Order {order_id} was cancelled. Stopping simulation.")
            return

        success = update_order_status_db(order_id, next_status)
        if success:
            logger.info(f"🚚 [Simulation] Order {order_id} updated to '{next_status}'")
        else:
            logger.warning("⚠️ [Simulation] Failed to update order %s to %s", order_id, next_status)
        await asyncio.sleep(SIM_STATUS_INTERVAL)

    logger.info(f"✅ [Simulation] Order {order_id} simulation complete (Delivered).")

def cart_total(cart: List[CartItem]) -> float:
    return round(sum(ci.unit_price * ci.quantity for ci in cart), 2)

# -------------------------
# Robust background runner
# -------------------------
def run_coro_in_background(coro):
    """
    Run coroutine in a dedicated background thread with its own loop when necessary.
    If a running loop exists, schedule the coroutine on it and return the Task.
    If no loop exists (sync context), spawn a background thread and run the coroutine there.
    """
    try:
        loop = asyncio.get_running_loop()
        # schedule on existing loop
        return loop.create_task(coro)
    except RuntimeError:
        # no running loop in this thread -> create one in background thread
        def _runner(loop, coro):
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(coro)
            finally:
                loop.close()

        bg_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_runner, args=(bg_loop, coro), daemon=True)
        t.start()
        return t  # thread handle

# -------------------------
# Agent tools
# -------------------------
@function_tool
async def find_item(
    ctx: RunContext[Userdata],
    query: Annotated[str, Field(description="Name or partial name of item (e.g., 'milk', 'paneer')")],
) -> str:
    matches = search_catalog_by_name_db(query)
    if not matches:
        return f"No items found matching '{query}'. Try generic names like 'milk' or 'rice'."
    lines = []
    for it in matches[:10]:
        lines.append(f"- {it['name']} (id: {it['id']}) — ₹{it['price']:.2f} — {it.get('size','')}")
    return "Found:\n" + "\n".join(lines)

@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    item_id: Annotated[str, Field(description="Catalog item id")],
    quantity: Annotated[int, Field(description="Quantity", default=1)] = 1,
    notes: Annotated[str, Field(description="Optional notes")] = "",
) -> str:
    if quantity < 1:
        return "Quantity must be at least 1."
    item = find_catalog_item_by_id_db(item_id)
    if not item:
        return f"Item id '{item_id}' not found."

    for ci in ctx.userdata.cart:
        if ci.item_id.lower() == item_id.lower():
            ci.quantity += quantity
            if notes:
                ci.notes = notes
            total = cart_total(ctx.userdata.cart)
            return f"Updated '{ci.name}' quantity to {ci.quantity}. Cart total: \u20B9{total:.2f}"

    ci = CartItem(item_id=item["id"], name=item["name"], unit_price=float(item["price"]), quantity=quantity, notes=notes)
    ctx.userdata.cart.append(ci)
    total = cart_total(ctx.userdata.cart)
    return f"Added {quantity} x '{item['name']}' to cart. Cart total: \u20B9{total:.2f}"

@function_tool
async def remove_from_cart(
    ctx: RunContext[Userdata],
    item_id: Annotated[str, Field(description="Catalog item id to remove")],
) -> str:
    before = len(ctx.userdata.cart)
    ctx.userdata.cart = [ci for ci in ctx.userdata.cart if ci.item_id.lower() != item_id.lower()]
    after = len(ctx.userdata.cart)
    if before == after:
        return f"Item '{item_id}' was not in your cart."
    total = cart_total(ctx.userdata.cart)
    return f"Removed item '{item_id}' from cart. Cart total: \u20B9{total:.2f}"

@function_tool
async def update_cart_quantity(
    ctx: RunContext[Userdata],
    item_id: Annotated[str, Field(description="Catalog item id to update")],
    quantity: Annotated[int, Field(description="New quantity")],
) -> str:
    if quantity < 1:
        return await remove_from_cart(ctx, item_id)
    for ci in ctx.userdata.cart:
        if ci.item_id.lower() == item_id.lower():
            ci.quantity = quantity
            total = cart_total(ctx.userdata.cart)
            return f"Updated '{ci.name}' quantity to {ci.quantity}. Cart total: \u20B9{total:.2f}"
    return f"Item '{item_id}' not found in cart."

@function_tool
async def show_cart(ctx: RunContext[Userdata]) -> str:
    if not ctx.userdata.cart:
        return "Your cart is empty."
    lines = []
    for ci in ctx.userdata.cart:
        lines.append(f"- {ci.quantity} x {ci.name} @ \u20B9{ci.unit_price:.2f} each = \u20B9{round(ci.unit_price * ci.quantity,2):.2f}")
    total = cart_total(ctx.userdata.cart)
    return "Your cart:\n" + "\n".join(lines) + f"\nTotal: \u20B9{total:.2f}"

@function_tool
async def add_recipe(
    ctx: RunContext[Userdata],
    dish_name: Annotated[str, Field(description="Name of dish, e.g. 'chai', 'maggi', 'dal chawal'")],
) -> str:
    key = dish_name.strip().lower()
    if key not in RECIPE_MAP:
        return f"Sorry, I don't have a recipe for '{dish_name}'. Try 'chai', 'maggi' or 'paneer butter masala'."
    added = []
    for item_id in RECIPE_MAP[key]:
        item = find_catalog_item_by_id_db(item_id)
        if not item:
            continue
        found = False
        for ci in ctx.userdata.cart:
            if ci.item_id.lower() == item_id.lower():
                ci.quantity += 1
                found = True
                break
        if not found:
            ctx.userdata.cart.append(CartItem(item_id=item["id"], name=item["name"], unit_price=float(item["price"]), quantity=1))
        added.append(item["name"])

    total = cart_total(ctx.userdata.cart)
    return f"Added ingredients for '{dish_name}': {', '.join(added)}. Cart total: \u20B9{total:.2f}"

@function_tool
async def ingredients_for(
    ctx: RunContext[Userdata],
    request: Annotated[str, Field(description="Natural language request, e.g. 'ingredients for peanut butter sandwich for two'")],
) -> str:
    text = (request or "").strip()
    servings = _parse_servings_from_text(text)

    m = re.search(r"ingredients? for (.+)", text, re.I)
    if m:
        dish = m.group(1)
    else:
        m2 = re.search(r"(?:make|for making|get me what i need for|i need) (.+)", text, re.I)
        dish = m2.group(1) if m2 else text

    dish = re.sub(r"for\s+\w+(?: people| person| persons)?", "", dish, flags=re.I).strip()
    key = dish.lower()

    item_ids = []
    if key in RECIPE_MAP:
        item_ids = RECIPE_MAP[key]
    else:
        item_ids = _infer_items_from_tags(dish)

    if not item_ids:
        return f"Sorry, I couldn't determine ingredients for '{request}'. Try a simpler phrase like 'chai' or 'maggi'."

    added = []
    for iid in item_ids:
        item = find_catalog_item_by_id_db(iid)
        if not item:
            continue
        found = False
        for ci in ctx.userdata.cart:
            if ci.item_id.lower() == iid.lower():
                ci.quantity += servings
                found = True
                break
        if not found:
            ctx.userdata.cart.append(CartItem(item_id=item['id'], name=item['name'], unit_price=float(item['price']), quantity=servings))
        added.append(item['name'])

    total = cart_total(ctx.userdata.cart)
    return f"I've added {', '.join(added)} to your cart for '{dish}'. (Servings: {servings}). Cart total: ₹{total:.2f}"

@function_tool
async def place_order(
    ctx: RunContext[Userdata],
    customer_name: Annotated[str, Field(description="Customer name")],
    address: Annotated[str, Field(description="Delivery address")],
) -> str:
    if not ctx.userdata.cart:
        return "Your cart is empty."

    order_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    total = cart_total(ctx.userdata.cart)

    # Persist to DB (and JSON)
    try:
        insert_order_db(order_id=order_id, timestamp=now, total=total, customer_name=customer_name, address=address, status="received", items=ctx.userdata.cart)
    except Exception:
        return "Failed to place order due to a server error. Please try again."

    # Clear Cart
    ctx.userdata.cart = []
    ctx.userdata.customer_name = customer_name

    # Trigger Background Simulation (Received -> Delivered)
    run_coro_in_background(simulate_delivery_flow(order_id))

    return f"Order placed successfully! Order ID: {order_id}. Total: \u20B9{total:.2f}. Express tracking is enabled and the status will update automatically."

@function_tool
async def cancel_order(
    ctx: RunContext[Userdata],
    order_id: Annotated[str, Field(description="Order ID to cancel")],
) -> str:
    o = get_order_db(order_id)
    if not o:
        return f"No order found with id {order_id}."

    status = o.get("status", "")
    if status == "delivered":
        return f"Order {order_id} has already been delivered and cannot be cancelled."

    if status == "cancelled":
        return f"Order {order_id} is already cancelled."

    updated = update_order_status_db(order_id, "cancelled")
    if updated:
        return f"Order {order_id} has been cancelled successfully."
    else:
        return f"Failed to cancel order {order_id}. Please try again."

@function_tool
async def get_order_status(
    ctx: RunContext[Userdata],
    order_id: Annotated[str, Field(description="Order ID to check")],
) -> str:
    o = get_order_db(order_id)
    if not o:
        return f"No order found with id {order_id}."
    return f"Order {order_id} status: {o.get('status', 'unknown')}. Updated at: {o.get('updated_at')}"

@function_tool
async def order_history(
    ctx: RunContext[Userdata],
    customer_name: Annotated[Optional[str], Field(description="Optional customer name to filter", default=None)] = None,
) -> str:
    rows = list_orders_db(limit=5, customer_name=customer_name)
    if not rows:
        return "No orders found."
    lines = []
    for o in rows:
        lines.append(f"- {o['order_id']} | \u20B9{o['total']:.2f} | Status: {o.get('status')}")
    prefix = "Recent Orders"
    if customer_name:
        prefix += f" for {customer_name}"
    return prefix + ":\n" + "\n".join(lines)

# -------------------------
# Agent Definition
# -------------------------
class FoodAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are 'Badal', a helpful assistant for 'Alpha Store', an Indian grocery store.
            Currency is Indian Rupees (₹).

            Capabilities:
            1. Catalog: Search for Indian items (Sudha milk, Tata salt, Maggi, Basmati rice).
            2. Cart: Add/Remove items, Show cart.
            3. Recipes: Add ingredients for dishes like Chai, Maggi, Paneer Butter Masala.
            4. Orders: Place orders.
            5. Cancellation: You can CANCEL an order if the user asks, provided it's not delivered yet.

            When placing an order, mention that express tracking is enabled.
            If user asks "Where is my order?", check status.
            The status advances automatically (simulated) so encourage them to check back in a few seconds.
            """,
            tools=[find_item, add_to_cart, remove_from_cart, update_cart_quantity, show_cart, add_recipe, place_order, cancel_order, get_order_status, order_history],
        )

# -------------------------
# Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    # load VAD model and stash on process userdata
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🇮🇳" * 12)
    logger.info("🚀 STARTING ALPHA SHOP (Indian Context + Auto-Tracking)")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    await session.start(
        agent=FoodAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
