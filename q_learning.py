import numpy as np
import random
import json

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.actions = actions  # Possible actions: ['easy', 'medium', 'hard']
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Stores Q-values in memory

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max(self.actions, key=lambda a: self.get_q_value(next_state, a))
        target = reward + self.discount_factor * self.get_q_value(next_state, best_next_action)
        old_value = self.get_q_value(state, action)
        self.q_table[(state, action)] = old_value + self.learning_rate * (target - old_value)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)
        else:
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            return max(q_values, key=q_values.get)

    def decay_exploration(self):
        self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)

    def save_q_table(self, file_name='q_table.json'):
        with open(file_name, 'w') as f:
            json.dump(self.q_table, f)

    def load_q_table(self, file_name='q_table.json'):
        try:
            with open(file_name, 'r') as f:
                self.q_table = json.load(f)
        except FileNotFoundError:
            self.q_table = {}
