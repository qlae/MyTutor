import json
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple


@dataclass
class QLearningAgent:
    actions: Iterable[str]
    learning_rate: float = 0.15
    discount_factor: float = 0.9
    exploration_rate: float = 0.6
    exploration_decay: float = 0.97
    min_exploration: float = 0.05
    q_table: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def _key(self, state: str, action: str) -> Tuple[str, str]:
        return (state, action)

    def get_q(self, state: str, action: str) -> float:
        return self.q_table.get(self._key(state, action), 0.0)

    def choose_action(self, state: str) -> str:
        if random.random() < self.exploration_rate:
            return random.choice(list(self.actions))
        return max(self.actions, key=lambda a: self.get_q(state, a))

    def update_q_value(self, state: str, action: str, reward: float, next_state: str) -> None:
        best_next = max((self.get_q(next_state, a2) for a2 in self.actions), default=0.0)
        k = self._key(state, action)
        current_q = self.q_table.get(k, 0.0)
        target = reward + self.discount_factor * best_next
        self.q_table[k] = current_q + self.learning_rate * (target - current_q)

    def decay_exploration(self) -> None:
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

    def save_q_table(self, file: str = 'q_table.json') -> None:
        serializable_q = {f"{s}::{a}": v for (s, a), v in self.q_table.items()}
        with open(file, 'w') as f:
            json.dump(serializable_q, f, indent=2)

    def load_q_table(self, file: str = 'q_table.json') -> None:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            self.q_table = {}
            for k, v in data.items():
                s, a = k.split("::", 1)
                self.q_table[(s, a)] = float(v)
        except FileNotFoundError:
            self.q_table = {}
