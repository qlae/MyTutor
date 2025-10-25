from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from q_learning import QLearningAgent
from pydantic import BaseModel, ValidationError
import json
import os
import random

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
CORS(app)

ACTIONS = ['easy', 'medium', 'hard']
agent = QLearningAgent(ACTIONS)
agent.load_q_table()

def load_seed(path: str = "questions_seed.json"):
    with open(path, "r") as f:
        return json.load(f)

SEED = load_seed()

def subjects():
    return list(SEED.keys())

def get_diagnostic(subject: str):
    return SEED[subject]["diagnostic"]

def slice_bank(subject: str, n_each: int = 2):
    bank = SEED[subject]["bank"][:]
    random.shuffle(bank)
    chunk = max(1, len(bank) // 3)
    easy_pool = bank[:chunk]
    med_pool = bank[chunk:2*chunk]
    hard_pool = bank[2*chunk:]

    def pick(pool):
        src = pool if len(pool) >= n_each else bank
        return random.sample(src, k=min(n_each, len(src)))

    return {
        "easy": pick(easy_pool),
        "medium": pick(med_pool or bank),
        "hard": pick(hard_pool or bank)
    }

def init_session():
    session.setdefault("state", "start")
    session.setdefault("subject", None)
    session.setdefault("difficulty", None)
    session.setdefault("last_quiz", [])
    session.modified = True

class StartPayload(BaseModel):
    subject: str

class QuizAnswers(BaseModel):
    answers: dict

@app.route("/")
def index():
    init_session()
    return render_template("index.html")

@app.get("/subjects")
def list_subjects():
    return jsonify({"subjects": subjects()})

@app.post("/start")
def start():
    init_session()
    try:
        data = StartPayload(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    if data.subject not in SEED:
        return jsonify({"error": "Unknown subject"}), 400

    session["subject"] = data.subject
    session["state"] = "diagnostic"
    session["difficulty"] = None
    session["last_quiz"] = []
    session.modified = True

    return jsonify({"diagnostic": get_diagnostic(data.subject)})

@app.post("/submit_diagnostic")
def submit_diagnostic():
    init_session()
    subject = session.get("subject")
    if not subject:
        return jsonify({"error": "Start a session first"}), 400

    rl_state = f"{subject}::start"
    difficulty = agent.choose_action(rl_state)

    pools = slice_bank(subject)
    quiz = pools[difficulty]

    session["state"] = "quiz"
    session["difficulty"] = difficulty
    session["last_quiz"] = quiz
    session.modified = True

    return jsonify({"difficulty": difficulty, "questions": quiz})

@app.post("/submit_quiz")
def submit_quiz():
    init_session()
    subject = session.get("subject")
    difficulty = session.get("difficulty")
    quiz = session.get("last_quiz", [])

    if not subject or not difficulty or not quiz:
        return jsonify({"error": "No active quiz. Start again."}), 400

    try:
        payload = QuizAnswers(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    correct = 0
    hints = []
    for i, q in enumerate(quiz):
        ua = payload.answers.get(f"q{i}")
        if ua == q.get("answer"):
            correct += 1
        else:
            if q.get("hint"):
                hints.append(q["hint"])

    total = len(quiz) if quiz else 1
    acc = correct / total
    if acc == 1:
        reward = 2.0
        next_diff = "hard" if difficulty != "hard" else "hard"
    elif acc >= 0.75:
        reward = 1.0
        next_diff = "hard" if difficulty == "medium" else "medium"
    elif acc >= 0.5:
        reward = 0.0
        next_diff = difficulty
    else:
        reward = -1.0
        next_diff = "easy"

    rl_state = f"{subject}::{difficulty}"
    next_state = f"{subject}::{next_diff}"
    agent.update_q_value(rl_state, difficulty, reward, next_state)
    agent.decay_exploration()
    agent.save_q_table()

    pools = slice_bank(subject)
    next_quiz = pools[next_diff]
    session["difficulty"] = next_diff
    session["last_quiz"] = next_quiz
    session.modified = True

    return jsonify({
        "score": correct,
        "total": total,
        "accuracy": round(acc, 2),
        "reward": reward,
        "next": next_diff,
        "hints": hints
    })

@app.get("/quiz")
def next_quiz():
    init_session()
    subject = session.get("subject")
    difficulty = session.get("difficulty")
    quiz = session.get("last_quiz", [])

    if not subject or not difficulty:
        return jsonify({"error": "Start first."}), 400

    if not quiz:
        quiz = slice_bank(subject)[difficulty]
        session["last_quiz"] = quiz
        session.modified = True

    return jsonify({"difficulty": difficulty, "questions": quiz})

@app.get("/docs")
def docs():
    return jsonify({
        "endpoints": {
            "GET /subjects": "list available subjects",
            "POST /start": {"body": {"subject": "Math"}, "returns": {"diagnostic": []}},
            "POST /submit_diagnostic": "returns quiz based on RL action",
            "POST /submit_quiz": {"body": {"answers": {"q0": "4"}}, "returns": "score, reward, next"},
            "GET /quiz": "get next quiz (same session)"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
