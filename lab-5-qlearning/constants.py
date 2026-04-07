
from enum import Enum

class Constants:
    DISCOUNT_FACTOR = [0.5, 0.9]
    EPSILON = [0.1, 0.5, 0.8]
    LEARNING_RATE = [0.1, 0.5, 0.9]
    EVAL_EVERY = 100
    NUM_EPISODES = 10_000
    EVAL_EPISODES = 50


class AgentType(Enum):
    Q_LEARNING = "Q-Learning"
    SARSA = "SARSA"