from flask import Flask, jsonify, request, render_template
from q_learning import QLearningAgent
import os

app = Flask(__name__, template_folder='.')

# Initialize the Q-Learning Agent
actions = ['easy', 'medium', 'hard']
agent = QLearningAgent(actions=actions)
agent.load_q_table()

user_state = {
    "current_state": "start",
    "score": 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    state = user_state["current_state"]
    action = data.get('difficulty')
    reward = data.get('reward')
    next_state = data.get('next_state')

    agent.update_q_value(state, action, reward, next_state)
    agent.decay_exploration()
    agent.save_q_table()

    user_state["current_state"] = next_state
    user_state["score"] += reward

    return jsonify({"message": "Submitted successfully", "score": user_state["score"]})

@app.route('/recommend', methods=['GET'])
def recommend():
    state = user_state["current_state"]
    recommended_action = agent.choose_action(state)
    return jsonify({"recommended_difficulty": recommended_action})

if __name__ == '__main__':
    app.run(debug=True)
