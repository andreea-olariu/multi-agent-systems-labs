import gymnasium
import numpy as np

class Trainer:
    def __init__(self, env_name: str, agent,
                 eval_every: int = 50,
                 num_episodes: int = 2000, 
                 eval_episodes: int = 50):
        
        self.env_name      = env_name
        self.agent         = agent
        self.eval_every    = eval_every
        self.num_episodes  = num_episodes
        self.eval_episodes = eval_episodes
 
        self.train_rewards: list[float] = []   
        self.eval_rewards:  list[float] = []   
        self.eval_steps:    list[int]   = []   
 
    def run_episode(self, env: gymnasium.Env, training: bool) -> float:
        state, _ = env.reset()
 
        if hasattr(self.agent, 'reset_episode') and training:
            self.agent.reset_episode()
 
        total_reward = 0.0
        done = False
 
        while not done:
            if training:
                action = self.agent.select_action(state)
            else:
                action = self.agent.greedy_action(state)
 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
 
            if training:
                self.agent.update(state, action, reward, next_state, done)
 
            total_reward += reward
            state = next_state
 
        return total_reward
 
    def evaluate(self, env: gymnasium.Env) -> float:
        rewards = [
            self.run_episode(env, training=False)
            for _ in range(self.eval_episodes)
        ]
        return float(np.mean(rewards))
 
    def train(self) -> dict:
        env_kwargs = {}
        if self.env_name == "FrozenLake-v1":
            env_kwargs["is_slippery"] = False
 
        train_env = gymnasium.make(self.env_name, **env_kwargs)
        eval_env  = gymnasium.make(self.env_name, **env_kwargs)
 
        for episode in range(1, self.num_episodes + 1):
            reward = self.run_episode(train_env, training=True)
            self.train_rewards.append(reward)
 
            if episode % self.eval_every == 0:
                avg_eval = self.evaluate(eval_env)
                self.eval_rewards.append(avg_eval)
                self.eval_steps.append(episode)
 
                print(f"  Episode {episode:5d}/{self.num_episodes}"
                      f" | train_reward={reward:6.1f}"
                      f" | eval_avg={avg_eval:6.3f}")
 
        train_env.close()
        eval_env.close()
 
        return {
            "train_rewards": self.train_rewards,
            "eval_rewards":  self.eval_rewards,
            "eval_steps":    self.eval_steps,
        }