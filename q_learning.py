import json
import random

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.95):
        self.actions = actions
        self.q_table = {}  # {(state, action): value}
        self.lr = learning_rate
        self.df = discount_factor
        self.er = exploration_rate
        self.decay = exploration_decay

    def get_q(self, s, a):
        return self.q_table.get((s, a), 0.0)

    def choose_action(self, state):
        if random.random() < self.er:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.get_q(state, a))

    def update_q_value(self, s, a, r, next_s):
        best_next = max([self.get_q(next_s, a2) for a2 in self.actions])
        current_q = self.get_q(s, a)
        self.q_table[(s, a)] = current_q + self.lr * (r + self.df * best_next - current_q)

    def decay_exploration(self):
        self.er = max(0.01, self.er * self.decay)

    def save_q_table(self, file='q_table.json'):
        # Convert tuple keys to string keys so we can save to JSON
        serializable_q = {f"{k[0]}::{k[1]}": v for k, v in self.q_table.items()}
        with open(file, 'w') as f:
            json.dump(serializable_q, f)

    def load_q_table(self, file='q_table.json'):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Convert string keys back to tuple keys
                self.q_table = {tuple(k.split("::")): v for k, v in data.items()}
        except FileNotFoundError:
            self.q_table = {}
