
import gymnasium
import sys

from constants import Constants, AgentType
from trainer import Trainer
from qlearning import QLearningAgent
from sarsa import SARSAAgent
from utils import plot_comparison


sys.stdout = open('output.txt', 'w')

def run_experiment(env_name: str, agent_type: AgentType,
                   discount_factor: float, epsilon: float, learning_rate: float) -> dict:
    print(f"\n  [{agent_type.value}] disc_factor={discount_factor}, epsilon={epsilon}, lr={learning_rate}")
 
    env = gymnasium.make(env_name)
 
    if agent_type == AgentType.Q_LEARNING:
        agent = QLearningAgent(
            n_actions=env.action_space.n,
            lr=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )
    else:
        agent = SARSAAgent(
            n_actions=env.action_space.n,
            lr=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )
 
    env.close()
 
    trainer = Trainer(
        env_name=env_name,
        agent=agent,
        eval_every=Constants.EVAL_EVERY,
        num_episodes=Constants.NUM_EPISODES,
        eval_episodes=Constants.EVAL_EPISODES,
    )
 
    return trainer.train()

def collect_all_results(env_name: str) -> dict:
    all_results = {}
 
    total = (len(Constants.DISCOUNT_FACTOR) *
             len(Constants.EPSILON) *
             len(Constants.LEARNING_RATE) *
             2) 
    done = 0
 
    for discount_factor in Constants.DISCOUNT_FACTOR:
        for epsilon in Constants.EPSILON:
            for learning_rate in Constants.LEARNING_RATE:
                for agent_type in [AgentType.Q_LEARNING, AgentType.SARSA]:

                    done += 1

                    print(f"\n[{done}/{total}] {env_name} | {agent_type.value}"
                          f" | discount_factor={discount_factor}, epsilon={epsilon}, lr={learning_rate}")
 
                    key = (agent_type, discount_factor, epsilon, learning_rate)
                    all_results[key] = run_experiment(
                        env_name, agent_type,
                        discount_factor, epsilon, learning_rate,
                    )
 
    return all_results


def plot_vary_alpha(env_name: str, all_results: dict,
                   fixed_gamma: float, fixed_epsilon: float) -> None:
    results = {}
 
    for lr in Constants.LEARNING_RATE:
        for agent_type in [AgentType.Q_LEARNING, AgentType.SARSA]:
            key   = (agent_type, fixed_gamma, fixed_epsilon, lr)
            label = f"{agent_type.value} lr={lr}"
            results[label] = all_results[key]
 
    title = (f"{env_name} | Varying lr"
             f" (disc_factor={fixed_gamma}, epsilon={fixed_epsilon} fixed)")
    save  = (f"results/{env_name}_vary_alpha"
             f"_gamma{fixed_gamma}_epsilon{fixed_epsilon}.png")
 
    plot_comparison(results, title=title, save_path=save)
 
 
def plot_vary_epsilon(env_name: str, all_results: dict,
                      fixed_gamma: float, fixed_alpha: float) -> None:
    results = {}
 
    for epsilon in Constants.EPSILON:
        for agent_type in [AgentType.Q_LEARNING, AgentType.SARSA]:
            key   = (agent_type, fixed_gamma, epsilon, fixed_alpha)
            label = f"{agent_type.value} epsilon={epsilon}"
            results[label] = all_results[key]
 
    title = (f"{env_name} | Varying epsilon"
             f" (gamma={fixed_gamma}, lr={fixed_alpha} fixed)")
    save  = (f"results/{env_name}_vary_epsilon"
             f"_gamma{fixed_gamma}_alpha{fixed_alpha}.png")
 
    plot_comparison(results, title=title, save_path=save)
 
 
def plot_vary_gamma(env_name: str, all_results: dict,
                    fixed_epsilon: float, fixed_alpha: float) -> None:
    results = {}
 
    for gamma in Constants.DISCOUNT_FACTOR:
        for agent_type in [AgentType.Q_LEARNING, AgentType.SARSA]:

            key   = (agent_type, gamma, fixed_epsilon, fixed_alpha)
            label = f"{agent_type.value} gamma={gamma}"
            results[label] = all_results[key]
 
    title = (f"{env_name} | Varying gamma"
             f" (epsilon={fixed_epsilon}, lr={fixed_alpha} fixed)")
    
    save  = (f"results/{env_name}_vary_gamma"
             f"_epsilon{fixed_epsilon}_alpha{fixed_alpha}.png")
 
    plot_comparison(results, title=title, save_path=save)
 
def plot_all(env_name: str, all_results: dict) -> None:
 
    for gamma in Constants.DISCOUNT_FACTOR:
        for epsilon in Constants.EPSILON:
            plot_vary_alpha(env_name, all_results,
                            fixed_gamma=gamma, fixed_epsilon=epsilon)
 
    for gamma in Constants.DISCOUNT_FACTOR:
        for alpha in Constants.LEARNING_RATE:
            plot_vary_epsilon(env_name, all_results,
                              fixed_gamma=gamma, fixed_alpha=alpha)
 
    for epsilon in Constants.EPSILON:
        for alpha in Constants.LEARNING_RATE:
            plot_vary_gamma(env_name, all_results,
                            fixed_epsilon=epsilon, fixed_alpha=alpha)

def main():
    envs = ['Taxi-v3', 'FrozenLake-v1']

    for env_name in envs:
        print("-" * 50 + f"\nEnvironment: {env_name}\n" + "-" * 50)
        all_results = collect_all_results(env_name)
        plot_all(env_name, all_results)


if __name__ == "__main__":
    main()