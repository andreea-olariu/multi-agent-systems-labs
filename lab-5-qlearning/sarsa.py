import numpy as np
from collections import defaultdict

from utils import epsilon_greedy


class SARSAAgent:
 
    def __init__(self,  n_actions: int,  lr: float, discount_factor: float, epsilon: float):
        self.n_actions       = n_actions
        self.lr              = lr
        self.discount_factor = discount_factor
        self.epsilon         = epsilon
        
        self._next_action: int | None = None
 
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
 
    def select_action(self, state: int) -> int:
        if self._next_action is not None:
            action = self._next_action
            self._next_action = None
            return action
        
        return epsilon_greedy(self.q_table, state, self.n_actions, self.epsilon)
 
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        
        if done:
            td_target = reward
            self._next_action = None

        else:
            next_action = epsilon_greedy(
                self.q_table, next_state, self.n_actions, self.epsilon
            )
            self._next_action = next_action
 
            td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
 
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
 
        return td_error
 
    def reset_episode(self):
        self._next_action = None
 
    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q_table[state]))