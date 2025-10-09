from dataclasses import dataclass

import numpy as np

from gymnasium_env import Coordinate


@dataclass(frozen=True)
class QState:
    coord: Coordinate
    milestones: int
    game_state: str


class QAgent:
    def __init__(
        self, lr: float, gamma: float, num_actions: int, init_q: float = 0.1
    ):
        self.q_table: dict[QState, np.ndarray] = {}
        self.lr = lr
        self.gamma = gamma
        self.num_actions = num_actions
        self.init_q = init_q

    def get_q_values(self, state: QState) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.ones(self.num_actions) * self.init_q

        return self.q_table[state]

    def get_action(self, state: QState, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, state: QState, action: int, reward: float, next_state: QState, done: bool):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        target = reward + (0 if done else self.gamma * np.max(next_q_values))
        q_values[action] += self.lr * (target - q_values[action])

    
