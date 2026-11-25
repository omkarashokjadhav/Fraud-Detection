"""
WARDEN backend — Fraud Analyst Desk API
=========================================
Run:
    pip install -r requirements.txt
    python app.py

By default this runs with a built-in heuristic scorer so it works out of the
box with no model file. To wire up your real notebook artifacts, set these
environment variables before starting:

    export WARDEN_MODEL_PATH=/path/to/xgb_model.pkl
    export WARDEN_DATA_PATH=/path/to/X_test.csv          # optional, for real ROC/PR curves
    export WARDEN_LABELS_PATH=/path/to/y_test.csv         # optional, pairs with WARDEN_DATA_PATH

Default login (seeded on first run):
    admin@warden.local / Warden123!   (role: admin)
    analyst@warden.local / Warden123! (role: analyst)
"""

import os
import io
import csv
import json
import math
import random
import sqlite3
import datetime
from functools import wraps

from flask import Flask, request, jsonify, g, Response, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt

try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

try:
    import joblib
    import shap
    import numpy as np
    import pandas as pd
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "warden.db")
SECRET_KEY = os.environ.get("WARDEN_SECRET", "dev-secret-change-me")
TOKEN_TTL_HOURS = 8
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

MODEL_PATH = os.environ.get("WARDEN_MODEL_PATH", "")
DATA_PATH = os.environ.get("WARDEN_DATA_PATH", "")
LABELS_PATH = os.environ.get("WARDEN_LABELS_PATH", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

EXPECTED_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "errorBalanceOrig",
    "errorBalanceDest", "origAccountEmptied",
]
FEATURE_LABELS = {
    "step": "Hour of simulation",
    "amount": "Transaction amount",
    "oldbalanceOrg": "Sender balance before",
    "newbalanceOrig": "Sender balance after",
    "oldbalanceDest": "Recipient balance before",
    "newbalanceDest": "Recipient balance after",
    "errorBalanceOrig": "Sender balance mismatch",
    "errorBalanceDest": "Recipient balance mismatch",
    "origAccountEmptied": "Sender account emptied",
}

CITIES = [
    ("New York, US", 40.71, -74.00), ("London, UK", 51.51, -0.13),
    ("Lagos, NG", 6.52, 3.38), ("Mumbai, IN", 19.08, 72.88),
    ("Sao Paulo, BR", -23.55, -46.63), ("Singapore, SG", 1.35, 103.82),
    ("Moscow, RU", 55.76, 37.62), ("Toronto, CA", 43.65, -79.38),
    ("Dubai, AE", 25.20, 55.27), ("Sydney, AU", -33.87, 151.21),
]
DEVICES = ["iOS · Safari", "Android · Chrome", "Windows · Edge", "macOS · Chrome", "Linux · Firefox"]

app = Flask(__name__)
CORS(app)
from flask import send_from_directory
import os

@app.route("/")
@app.route("/dashboard")
def serve_dashboard():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

socketio = SocketIO(app, cors_allowed_origins="*") if SOCKETIO_AVAILABLE else None

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'analyst',
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until TEXT,
        created_at TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS login_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        success INTEGER NOT NULL,
        ip TEXT,
        ts TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actor_email TEXT NOT NULL,
        action TEXT NOT NULL,
        target TEXT,
        detail TEXT,
        ts TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        payload TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        risk REAL NOT NULL,
        ts TEXT NOT NULL
    )""")
    conn.commit()

    c.execute("SELECT COUNT(*) AS n FROM users")
    if c.fetchone()["n"] == 0:
        now = datetime.datetime.utcnow().isoformat()
        for email, role in [("admin@warden.local", "admin"), ("analyst@warden.local", "analyst")]:
            c.execute(
                "INSERT INTO users (email, password_hash, role, created_at) VALUES (?,?,?,?)",
                (email, generate_password_hash(""), role, now),
            )
        conn.commit()
    conn.close()


def log_audit(actor, action, target="", detail=""):
    conn = get_db()
    conn.execute(
        "INSERT INTO audit_log (actor_email, action, target, detail, ts) VALUES (?,?,?,?,?)",
        (actor, action, target, detail, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def create_token(user_row):
    payload = {
        "sub": user_row["email"],
        "role": user_row["role"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def token_required(admin_only=False):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return jsonify({"error": "Missing or malformed Authorization header."}), 401
            token = auth.split(" ", 1)[1]
            try:
                decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            except jwt.ExpiredSignatureError:
                return jsonify({"error": "Session expired. Sign in again."}), 401
            except jwt.InvalidTokenError:
                return jsonify({"error": "Invalid session token."}), 401
            if admin_only and decoded.get("role") != "admin":
                return jsonify({"error": "Admin access required."}), 403
            g.user_email = decoded["sub"]
            g.user_role = decoded["role"]
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Scoring — real model if available, deterministic heuristic fallback
# ---------------------------------------------------------------------------
class ModelWrapper:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.loaded = False
        if ML_LIBS_AVAILABLE and MODEL_PATH and os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.explainer = shap.TreeExplainer(self.model)
                self.loaded = True
            except Exception as e:
                print(f"[WARDEN] Could not load model at {MODEL_PATH}: {e}")

    def score(self, features: dict):
        if self.loaded:
            row = pd.DataFrame([{k: features.get(k, 0) for k in EXPECTED_COLUMNS}])
            proba = float(self.model.predict_proba(row)[0][1])
            raw_shap = self.explainer.shap_values(row)
            vals = raw_shap[1][0] if isinstance(raw_shap, list) else raw_shap[0]
            contrib = {k: float(v) for k, v in zip(EXPECTED_COLUMNS, vals)}
            return proba, contrib
        return heuristic_score(features)


def heuristic_score(f: dict):
    """Fallback scorer mirroring the notebook's engineered-feature logic,
    used when no real model file is configured."""
    contrib = {}
    contrib["origAccountEmptied"] = 0.27 + random.random() * 0.08 if f.get("origAccountEmptied") else -0.03 - random.random() * 0.03
    contrib["amount"] = (0.16 + random.random() * 0.08) if f.get("amount", 0) > 500000 else (
        -0.06 - random.random() * 0.04 if f.get("amount", 0) < 40000 else 0.04 + random.random() * 0.04)
    contrib["errorBalanceOrig"] = 0.12 + random.random() * 0.06 if abs(f.get("errorBalanceOrig", 0)) > 1 else -0.02 - random.random() * 0.02
    contrib["errorBalanceDest"] = 0.10 + random.random() * 0.06 if abs(f.get("errorBalanceDest", 0)) > 1 else -0.02 - random.random() * 0.02
    contrib["oldbalanceOrg"] = 0.03 + random.random() * 0.03 if f.get("oldbalanceOrg", 0) < 20000 else -0.08 - random.random() * 0.05
    contrib["newbalanceOrig"] = -0.02 - random.random() * 0.02
    contrib["oldbalanceDest"] = 0.09 + random.random() * 0.05 if f.get("oldbalanceDest", 0) == 0 else -0.03 - random.random() * 0.02
    contrib["newbalanceDest"] = 0.02 + random.random() * 0.02
    contrib["step"] = -0.04 - random.random() * 0.04
    base = 0.08
    proba = max(0.02, min(0.98, base + sum(contrib.values())))
    return proba, contrib


model_wrapper = ModelWrapper()

def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("The 'requests' package isn't installed. Run: pip install requests")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY isn't set on the server. See README for how to set it.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    resp = requests.post(
        url,
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.4},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip() or "No response generated."
    except (KeyError, IndexError):
        raise RuntimeError("Gemini returned no usable content — it may have been blocked by safety filters.")

# Reported results from the notebook's model comparison phase.
MODEL_COMPARISON = [
    {"model": "XGBoost", "f1": 0.989, "precision": 0.986, "recall": 0.992, "auc": 0.998, "chosen": True},
    {"model": "Random Forest", "f1": 0.985, "precision": 0.981, "recall": 0.989, "auc": 0.996, "chosen": False},
    {"model": "Logistic Regression", "f1": 0.125, "precision": 0.081, "recall": 0.274, "auc": 0.812, "chosen": False},
]


# ---------------------------------------------------------------------------
# Synthetic transaction generator (used for /simulate and to seed the feed)
# ---------------------------------------------------------------------------
def generate_transaction_payload():
    tx_type = random.choice(["CASH_OUT", "CASH_OUT", "TRANSFER"])
    amount = round(5000 + random.random() * random.random() * 950000, 2)
    emptied = random.random() < 0.42
    old_org = amount + round(random.random() * 500) if emptied else round(amount * (1 + random.random() * 3))
    new_orig = max(0, old_org - amount)
    dest_new = random.random() < 0.55
    old_dest = 0 if dest_new else round(random.random() * 40000)
    new_dest = old_dest + amount
    err_orig = new_orig - (old_org - amount)
    err_dest = new_dest - (old_dest + amount)
    city, lat, lng = random.choice(CITIES)

    features = {
        "step": random.randint(1, 24),
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorBalanceOrig": err_orig,
        "errorBalanceDest": err_dest,
        "origAccountEmptied": 1 if new_orig == 0 and old_org > 0 else 0,
    }
    proba, contrib = model_wrapper.score(features)
    tx_id = "TX-" + str(random.randint(100000, 999999))
    return {
        "id": tx_id,
        "type": tx_type,
        "userId": "U-" + str(random.randint(1000, 9999)),
        "card": "•••• " + str(random.randint(1000, 9999)),
        "device": random.choice(DEVICES),
        "location": {"city": city, "lat": lat, "lng": lng},
        "features": features,
        "risk": round(proba * 100, 1),
        "contrib": contrib,
        "status": "pending",
        "ts": datetime.datetime.utcnow().isoformat(),
    }


def save_transaction(tx):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO transactions (id, payload, status, risk, ts) VALUES (?,?,?,?,?)",
        (tx["id"], json.dumps(tx), tx["status"], tx["risk"], tx["ts"]),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or "@" not in email:
        return jsonify({"error": "Enter a valid work email."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if existing:
        conn.close()
        return jsonify({"error": "An account with that email already exists."}), 409

    conn.execute(
        "INSERT INTO users (email, password_hash, role, created_at) VALUES (?,?,?,?)",
        (email, generate_password_hash(password), "analyst", datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    log_audit(email, "register", detail="New analyst account created")
    return jsonify({"message": "Account created. You can now sign in."}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    if user and user["locked_until"]:
        locked_until = datetime.datetime.fromisoformat(user["locked_until"])
        if datetime.datetime.utcnow() < locked_until:
            conn.close()
            return jsonify({"error": f"Account locked until {locked_until.isoformat()}Z after repeated failed attempts."}), 423

    ip = request.remote_addr or "unknown"
    if not user or not check_password_hash(user["password_hash"], password):
        if user:
            attempts = user["failed_attempts"] + 1
            locked_until = None
            if attempts >= MAX_LOGIN_ATTEMPTS:
                locked_until = (datetime.datetime.utcnow() + datetime.timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
            conn.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE email=?", (attempts, locked_until, email))
            conn.commit()
        conn.execute("INSERT INTO login_history (user_email, success, ip, ts) VALUES (?,?,?,?)",
                     (email, 0, ip, datetime.datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({"error": "Invalid email or password."}), 401

    conn.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE email=?", (email,))
    conn.execute("INSERT INTO login_history (user_email, success, ip, ts) VALUES (?,?,?,?)",
                 (email, 1, ip, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    token = create_token(user)
    role = user["role"]
    conn.close()
    log_audit(email, "login", detail=f"Signed in from {ip}")
    return jsonify({"token": token, "email": email, "role": role})


@app.route("/api/auth/me")
@token_required()
def me():
    return jsonify({"email": g.user_email, "role": g.user_role})


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------
@app.route("/api/transactions", methods=["GET"])
@token_required()
def list_transactions():
    conn = get_db()
    rows = conn.execute("SELECT payload FROM transactions ORDER BY ts DESC LIMIT 500").fetchall()
    conn.close()
    txs = [json.loads(r["payload"]) for r in rows]

    q = (request.args.get("q") or "").lower()
    status = request.args.get("status")
    min_risk = request.args.get("min_risk", type=float)
    if q:
        txs = [t for t in txs if q in t["id"].lower() or q in t.get("userId", "").lower()
               or q in t.get("card", "").lower() or q in str(t["features"]["amount"])]
    if status and status != "all":
        txs = [t for t in txs if t["status"] == status]
    if min_risk is not None:
        txs = [t for t in txs if t["risk"] >= min_risk]
    return jsonify(txs)


@app.route("/api/transactions/simulate", methods=["POST"])
@token_required()
def simulate_transaction():
    tx = generate_transaction_payload()
    save_transaction(tx)
    if socketio:
        socketio.emit("new_transaction", tx)
    return jsonify(tx), 201


@app.route("/api/transactions/<tx_id>/decision", methods=["POST"])
@token_required()
def decide_transaction(tx_id):
    data = request.get_json(force=True, silent=True) or {}
    decision = data.get("decision")
    if decision not in ("approved", "rejected"):
        return jsonify({"error": "decision must be 'approved' or 'rejected'."}), 400

    conn = get_db()
    row = conn.execute("SELECT payload FROM transactions WHERE id=?", (tx_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Transaction not found."}), 404
    tx = json.loads(row["payload"])
    tx["status"] = decision
    conn.execute("UPDATE transactions SET payload=?, status=? WHERE id=?", (json.dumps(tx), decision, tx_id))
    conn.commit()
    conn.close()
    log_audit(g.user_email, f"transaction_{decision}", target=tx_id)
    if socketio:
        socketio.emit("transaction_updated", tx)
    return jsonify(tx)


@app.route("/api/transactions/<tx_id>/explain")
@token_required()
def explain_transaction(tx_id):
    conn = get_db()
    row = conn.execute("SELECT payload FROM transactions WHERE id=?", (tx_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Transaction not found."}), 404
    tx = json.loads(row["payload"])
    ranked = sorted(
        [{"feature": k, "label": FEATURE_LABELS.get(k, k), "contribution": v} for k, v in tx["contrib"].items()],
        key=lambda x: abs(x["contribution"]), reverse=True,
    )
    return jsonify({"risk": tx["risk"], "top_features": ranked})


def _transaction_context(tx):
    ranked = sorted(
        [{"label": FEATURE_LABELS.get(k, k), "contribution": round(v, 2)} for k, v in tx["contrib"].items()],
        key=lambda x: abs(x["contribution"]), reverse=True,
    )
    return json.dumps({
        "id": tx["id"], "type": tx["type"], "amount": tx["features"]["amount"],
        "senderBalanceBefore": tx["features"]["oldbalanceOrg"], "senderBalanceAfter": tx["features"]["newbalanceOrig"],
        "recipientBalanceBefore": tx["features"]["oldbalanceDest"], "recipientBalanceAfter": tx["features"]["newbalanceDest"],
        "senderAccountEmptied": bool(tx["features"]["origAccountEmptied"]),
        "riskScore": tx["risk"], "location": tx.get("location", {}).get("city"), "device": tx.get("device"),
        "topShapFeatures": ranked,
    }, indent=2)


@app.route("/api/transactions/<tx_id>/narrative", methods=["POST"])
@token_required()
def narrative_route(tx_id):
    conn = get_db()
    row = conn.execute("SELECT payload FROM transactions WHERE id=?", (tx_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Transaction not found."}), 404
    tx = json.loads(row["payload"])
    system = (
        "You are WARDEN, an AI assistant embedded in a bank's fraud operations dashboard. "
        "Given one flagged transaction and its SHAP feature attributions, write a short plain-English "
        "narrative (3-5 sentences) for a human fraud analyst: what happened, why the model flagged it, "
        "and a clear suggested next step. Be direct and concrete. Do not invent facts not present in the data."
    )
    try:
        text = call_llm(system, "Transaction and SHAP data:\n" + _transaction_context(tx))
        return jsonify({"narrative": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/transactions/<tx_id>/chat", methods=["POST"])
@token_required()
def chat_route(tx_id):
    conn = get_db()
    row = conn.execute("SELECT payload FROM transactions WHERE id=?", (tx_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Transaction not found."}), 404
    tx = json.loads(row["payload"])

    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    history = data.get("history", [])[-6:]
    if not question:
        return jsonify({"error": "question is required."}), 400

    system = (
        "You are WARDEN, a fraud analyst's AI assistant. Answer the analyst's question about the specific "
        "flagged transaction below, grounded strictly in its data and SHAP attributions. Be concise (2-4 "
        "sentences), concrete, and give a direct recommendation when asked for one."
    )
    convo = "\n".join(f"{'Analyst' if m.get('role')=='user' else 'WARDEN'}: {m.get('text','')}" for m in history)
    user_prompt = (
        "Transaction and SHAP data:\n" + _transaction_context(tx)
        + "\n\nConversation so far:\n" + convo
        + "\n\nAnalyst's latest question: " + question
    )
    try:
        text = call_llm(system, user_prompt)
        return jsonify({"answer": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.route("/api/metrics/model-comparison")
@token_required()
def model_comparison():
    return jsonify(MODEL_COMPARISON)


@app.route("/api/metrics/confusion-matrix")
@token_required()
def confusion_matrix_route():
    if ML_LIBS_AVAILABLE and model_wrapper.loaded and DATA_PATH and LABELS_PATH and os.path.exists(DATA_PATH):
        X = pd.read_csv(DATA_PATH)[EXPECTED_COLUMNS]
        y = pd.read_csv(LABELS_PATH).iloc[:, 0]
        preds = model_wrapper.model.predict(X)
        tp = int(((preds == 1) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
    else:
        tp, tn, fp, fn = 812, 552940, 41, 71
    return jsonify({"tp": tp, "tn": tn, "fp": fp, "fn": fn})


@app.route("/api/metrics/curve")
@token_required()
def curve_route():
    kind = request.args.get("kind", "roc")
    if ML_LIBS_AVAILABLE and model_wrapper.loaded and DATA_PATH and LABELS_PATH and os.path.exists(DATA_PATH):
        from sklearn.metrics import roc_curve, precision_recall_curve
        X = pd.read_csv(DATA_PATH)[EXPECTED_COLUMNS]
        y = pd.read_csv(LABELS_PATH).iloc[:, 0]
        proba = model_wrapper.model.predict_proba(X)[:, 1]
        if kind == "pr":
            precision, recall, _ = precision_recall_curve(y, proba)
            points = [{"x": float(r), "y": float(p)} for r, p in zip(recall[::5], precision[::5])]
        else:
            fpr, tpr, _ = roc_curve(y, proba)
            points = [{"x": float(f), "y": float(t)} for f, t in zip(fpr[::5], tpr[::5])]
        return jsonify({"points": points, "synthetic": False})

    n = 40
    points = []
    for i in range(n + 1):
        x = i / n
        y = 1 - (1 - x) ** 6 if kind == "roc" else max(0.85, 1 - x ** 4 * 0.3)
        points.append({"x": round(x, 3), "y": round(min(1, y), 4)})
    return jsonify({"points": points, "synthetic": True})


@app.route("/api/metrics/summary")
@token_required()
def metrics_summary():
    conn = get_db()
    rows = conn.execute("SELECT payload FROM transactions").fetchall()
    conn.close()
    txs = [json.loads(r["payload"]) for r in rows]
    total = len(txs)
    flagged = len([t for t in txs if t["risk"] >= 65])
    avg_risk = round(sum(t["risk"] for t in txs) / total, 1) if total else 0
    return jsonify({
        "total_transactions": total,
        "flagged_today": flagged,
        "avg_risk": avg_risk,
        "model_loaded": model_wrapper.loaded,
    })


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@app.route("/api/export/transactions.csv")
@token_required()
def export_csv():
    conn = get_db()
    rows = conn.execute("SELECT payload FROM transactions ORDER BY ts DESC").fetchall()
    conn.close()
    txs = [json.loads(r["payload"]) for r in rows]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "type", "amount", "risk", "status", "userId", "card", "device", "city", "timestamp"])
    for t in txs:
        writer.writerow([t["id"], t["type"], t["features"]["amount"], t["risk"], t["status"],
                          t.get("userId", ""), t.get("card", ""), t.get("device", ""),
                          t.get("location", {}).get("city", ""), t["ts"]])
    log_audit(g.user_email, "export_csv", detail=f"{len(txs)} transactions")
    return Response(buf.getvalue(), mimetype="text/csv",
                     headers={"Content-Disposition": "attachment; filename=warden_transactions.csv"})


@app.route("/api/export/report.pdf")
@token_required()
def export_pdf():
    if not PDF_AVAILABLE:
        return jsonify({"error": "PDF export requires the 'fpdf2' package. Run: pip install fpdf2"}), 501

    conn = get_db()
    rows = conn.execute("SELECT payload FROM transactions ORDER BY risk DESC LIMIT 25").fetchall()
    conn.close()
    txs = [json.loads(r["payload"]) for r in rows]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "WARDEN - Fraud Risk Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated {datetime.datetime.utcnow().isoformat()}Z by {g.user_email}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(30, 8, "ID", border=1); pdf.cell(25, 8, "Type", border=1)
    pdf.cell(30, 8, "Amount", border=1); pdf.cell(20, 8, "Risk", border=1)
    pdf.cell(30, 8, "Status", border=1); pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for t in txs:
        pdf.cell(30, 7, t["id"], border=1)
        pdf.cell(25, 7, t["type"], border=1)
        pdf.cell(30, 7, f'{t["features"]["amount"]:.0f}', border=1)
        pdf.cell(20, 7, f'{t["risk"]:.0f}', border=1)
        pdf.cell(30, 7, t["status"], border=1)
        pdf.ln()

    out = io.BytesIO(pdf.output(dest="S").encode("latin-1"))
    log_audit(g.user_email, "export_pdf", detail=f"{len(txs)} transactions")
    return send_file(out, mimetype="application/pdf", as_attachment=True, download_name="warden_report.pdf")


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@app.route("/api/admin/users")
@token_required(admin_only=True)
def admin_users():
    conn = get_db()
    rows = conn.execute("SELECT email, role, created_at, failed_attempts, locked_until FROM users").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/audit-log")
@token_required(admin_only=True)
def admin_audit_log():
    conn = get_db()
    rows = conn.execute("SELECT * FROM audit_log ORDER BY ts DESC LIMIT 200").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/login-history")
@token_required(admin_only=True)
def admin_login_history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM login_history ORDER BY ts DESC LIMIT 200").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_wrapper.loaded, "socketio": SOCKETIO_AVAILABLE})


if __name__ == "__main__":
    init_db()
    _conn = get_db()
    _count = _conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()["n"]
    _conn.close()
    if _count == 0:
        for _ in range(15):
            save_transaction(generate_transaction_payload())
    if socketio:
        socketio.run(app, host="0.0.0.0", port=5000, debug=True)
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)