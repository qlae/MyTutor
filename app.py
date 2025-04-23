from flask import Flask, request, jsonify, render_template
from q_learning import QLearningAgent
import json, random

app = Flask(__name__, template_folder='.')

actions = ['easy', 'medium', 'hard']
agent = QLearningAgent(actions)
agent.load_q_table()

user_state = {
    "subject": None,
    "state": "start",
    "difficulty": None
}

questions = {
    "Math": {
        "diagnostic": [
            {"question": "What is 4 + 4?", "answer": "8"},
            {"question": "What is 9 - 3?", "answer": "6"}
        ],
        "easy": [
            {"question": "2 + 2 = ?", "options": ["3", "4", "5"], "answer": "4"},
            {"question": "3 + 1 = ?", "options": ["3", "4", "5"], "answer": "4"}
        ],
        "medium": [
            {"question": "12 / 3 = ?", "options": ["3", "4", "5"], "answer": "4"},
            {"question": "2 x 6 = ?", "options": ["10", "12", "14"], "answer": "12"}
        ],
        "hard": [
            {"question": "√81 = ?", "options": ["8", "9", "10"], "answer": "9"},
            {"question": "30% of 100 = ?", "options": ["20", "30", "40"], "answer": "30"}
        ]
    },
    "Science": {
        "diagnostic": [
            {"question": "What gas do humans breathe?", "answer": "Oxygen"},
            {"question": "What planet do we live on?", "answer": "Earth"}
        ],
        "easy": [
            {"question": "Water is made of?", "options": ["O2", "H2O", "CO2"], "answer": "H2O"},
            {"question": "Sun rises in the?", "options": ["East", "West", "North"], "answer": "East"}
        ],
        "medium": [
            {"question": "Gas plants release?", "options": ["CO2", "Oxygen", "Nitrogen"], "answer": "Oxygen"},
            {"question": "What does H stand for in H2O?", "options": ["Hydrogen", "Helium", "Heat"], "answer": "Hydrogen"}
        ],
        "hard": [
            {"question": "Organelle that makes energy?", "options": ["Nucleus", "Mitochondria", "Ribosome"], "answer": "Mitochondria"},
            {"question": "What causes tides?", "options": ["Sun", "Wind", "Moon"], "answer": "Moon"}
        ]
    },
    "English": {
        "diagnostic": [
            {"question": "Past tense of run?", "answer": "ran"},
            {"question": "Plural of child?", "answer": "children"}
        ],
        "easy": [
            {"question": "Synonym for happy?", "options": ["Sad", "Joyful", "Angry"], "answer": "Joyful"},
            {"question": "Which is a noun?", "options": ["Run", "Fast", "Apple"], "answer": "Apple"}
        ],
        "medium": [
            {"question": "Antonym of hot?", "options": ["Cold", "Warm", "Burning"], "answer": "Cold"},
            {"question": "Which is a verb?", "options": ["Jump", "Quick", "Smart"], "answer": "Jump"}
        ],
        "hard": [
            {"question": "What is a conjunction?", "options": ["And", "Run", "Quick"], "answer": "And"},
            {"question": "Which is an adjective?", "options": ["Soft", "Run", "Dance"], "answer": "Soft"}
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
    subject = user_state['subject']
    user_state['state'] = 'quiz'
    difficulty = agent.choose_action('quiz')
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

    correct = sum(1 for i, q in enumerate(quiz) if data.get(f"q{i}") == q['answer'])
    reward = correct - (2 - correct)

    agent.update_q_value("quiz", difficulty, reward, "quiz")
    agent.decay_exploration()
    agent.save_q_table()

    return jsonify({
        "score": correct,
        "reward": reward,
        "next": agent.choose_action("quiz")
    })

if __name__ == '__main__':
    app.run(debug=True)
