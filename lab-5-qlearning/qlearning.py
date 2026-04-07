from collections import defaultdict
import numpy as np

from utils import epsilon_greedy

class QLearningAgent:
    def __init__(self, n_actions: int, lr: float, discount_factor: float, epsilon: float):
        self.n_actions       = n_actions
        self.lr              = lr    
        self.discount_factor = discount_factor    
        self.epsilon         = epsilon  
 
        # start Q table with all zeros
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state: int) -> int:
        return epsilon_greedy(self.q_table, state, self.n_actions, self.epsilon)
 
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        future_value = 0.0 if done else np.max(self.q_table[next_state])
 
        td_target = reward + self.discount_factor * future_value
 
        td_error = td_target - self.q_table[state][action]
 
        self.q_table[state][action] += self.lr * td_error
 
        return td_error
 
    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q_table[state]))
