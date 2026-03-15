from abc import ABC
import gym

class Game(ABC):
    def init_env(self, game: str, *args, **kwargs):
        print(f"Initializing environment for game: {game}")
        self.env = gym.make(game, *args, **kwargs)


    def get_states(self):
        return self.env.observation_space
    

    def get_actions(self):
        return self.env.action_space
    
    def get_probability(self, state, action):
        return self.env.P[state][action]
    


class FrozenGame(Game):
    def __init__(self):
        self.init_env("FrozenLake-v1", map_name="8x8", is_slippery=True)



class TaxiGame(Game):
    def __init__(self):
        self.init_env("Taxi-v3")
