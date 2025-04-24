from flask import Flask, request, jsonify, render_template
from q_learning import QLearningAgent
import json

app = Flask(__name__, template_folder='.')

actions = ['easy', 'medium', 'hard']
agent = QLearningAgent(actions)
agent.load_q_table()

user_state = {
    "subject": None,
    "state": "start",
    "difficulty": None,
    "last_quiz": []
}

questions = {
    "Math": {
        "diagnostic": [
            {"question": "What is 3 + 3?", "answer": "6"},
            {"question": "What is 10 - 4?", "answer": "6"}
        ],
        "easy": [
            {"question": "2 + 2 = ?", "options": ["3", "4", "5"], "answer": "4", "hint": "Think of how many fingers you have on one hand."},
            {"question": "3 + 1 = ?", "options": ["3", "4", "5"], "answer": "4", "hint": "Add one to three."}
        ],
        "medium": [
            {"question": "12 / 3 = ?", "options": ["3", "4", "5"], "answer": "4", "hint": "How many times does 3 fit in 12?"},
            {"question": "2 x 6 = ?", "options": ["10", "12", "14"], "answer": "12", "hint": "Multiplication of 2 times 6."}
        ],
        "hard": [
            {"question": "√81 = ?", "options": ["8", "9", "10"], "answer": "9", "hint": "What number times itself is 81?"},
            {"question": "30% of 100 = ?", "options": ["20", "30", "40"], "answer": "30", "hint": "Find 1/10, then multiply by 3."}
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    data = request.json
    subject = data['subject']
    user_state['subject'] = subject
    return jsonify(questions[subject]['diagnostic'])

@app.route('/submit_diagnostic', methods=['POST'])
def submit_diagnostic():
    # Fixed logic — start at easy
    difficulty = 'easy'
    subject = user_state['subject']
    user_state['state'] = 'quiz'
    user_state['difficulty'] = difficulty
    return jsonify({
        "difficulty": difficulty,
        "questions": questions[subject][difficulty]
    })

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.json
    subject = user_state['subject']
    difficulty = user_state['difficulty']
    quiz = questions[subject][difficulty]

    correct = 0
    hints = []
    for i, q in enumerate(quiz):
        user_answer = data.get(f"q{i}")
        if user_answer == q['answer']:
            correct += 1
        else:
            hints.append(q['hint'])

    # Determine reward + next difficulty
    if correct == 2:
        reward = 2
        next_difficulty = 'hard'
    elif correct == 1:
        reward = 0
        next_difficulty = difficulty  # stay
    else:
        reward = -2
        next_difficulty = 'easy'

    # Update RL agent
    agent.update_q_value("quiz", difficulty, reward, "quiz")
    agent.decay_exploration()
    agent.save_q_table()

    # Save for retry
    user_state['difficulty'] = next_difficulty
    user_state['last_quiz'] = questions[subject][next_difficulty]

    return jsonify({
        "score": correct,
        "reward": reward,
        "next": next_difficulty,
        "hints": hints
    })

@app.route('/quiz', methods=['GET'])
def next_quiz():
    subject = user_state['subject']
    difficulty = user_state['difficulty']
    quiz = questions[subject][difficulty]
    return jsonify({
        "difficulty": difficulty,
        "questions": quiz
    })
if __name__ == '__main__':
    app.run(debug=True)
