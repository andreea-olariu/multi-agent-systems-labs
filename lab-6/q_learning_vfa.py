import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import json
import tqdm
import random
import argparse
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CosineActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cos(x)

class Estimator(object):
    def __init__(self, state_dim = 4, action_dim = 2, hidden_dim = 100, lr = 0.0001, activation = 'cos'):

        if activation == 'cos':
            activation = CosineActivation()
        elif activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()

        self.criterion = torch.nn.MSELoss()
        self.model = None
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, action_dim)
        )

        self.model = self.model.to(DEVICE)
       
        self._initialize_weights_and_bias(state_dim, hidden_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    def _initialize_weights_and_bias(self, state_dim = 4, hidden_dim = 100):
        # Initialize the weights and biases of the first layer
        # the first weight is a state_dim x hidden_dim matrix. 
        # Initialize each row with a normal distribution with mean 0 and standard deviation sqrt((i+1) * 0.5), where i is the row index.
        # the bias is uniformly distributed between 0 and 2 pi

        for i in range(state_dim):
            torch.nn.init.normal_(self.model[0].weight[i], mean = 0, std = np.sqrt((i+1) * 0.5))
        torch.nn.init.uniform_(self.model[0].bias, a = 0, b = 2 * np.pi)
        
    def update(self, state, y):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
        y_pred = self.model(state_tensor)
        loss = self.criterion(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, state):
            with torch.no_grad():
                if self.model is None:
                    return torch.zeros((2,), device=DEVICE)
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
                return self.model(state_tensor)


def select_action(state, epsilon, model):

    randomness = random.random()
    if randomness < epsilon:
        action = env.action_space.sample()

    else:
        q_values = model.predict(state)
        action = torch.argmax(q_values).item()
    
    return action
            


def q_learning(
    env,
    model,
    episodes,
    decay = False,
    gamma = 0.9,
    epsilon = 0.1,
    eps_decay = 0.9,
):

    total_reward = []
    total_loss = []

    for episode in tqdm.tqdm(range(episodes)):
        state, _ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)

        done = False
        episode_reward = 0

        while not done:
            action = select_action(state, epsilon, model)
            
            step_res = env.step(action)
            next_state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
            episode_reward += reward

            state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=DEVICE)

            # TODO 3: Implement the Q-learning update rule, using the model.predict and model.update functions
            # predict the q values for the current state
            
            q_values = model.predict(state)

            # If the episode is done, the q value for the action taken should be the reward
            if done:
                q_values[action] = reward
                loss = model.update(state, q_values)

                total_loss.append(loss)
                break
            
            # otherwise, predict the q values for the next state and use them as the TD target for the update
            q_values_next = model.predict(next_state)
            td_target = reward + gamma * torch.max(q_values_next).item()
            q_values[action] = td_target
            loss = model.update(state, q_values)

            total_loss.append(loss)

            state = next_state

        # Update epsilon
        if decay:
            epsilon = max(epsilon * eps_decay, 0.001)

        total_reward.append(episode_reward)

    return total_reward, total_loss


def moving_average_with_variance(data, window_size=50):
    """Calculate moving average and variance over the given window size."""
    if len(data) < window_size:
        return [], [], []
    
    indices = np.arange(window_size - 1, len(data))
    means = []
    upper_bounds = []
    lower_bounds = []
    
    for i in range(window_size - 1, len(data)):
        window = data[i - window_size + 1 : i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        means.append(mean_val)
        upper_bounds.append(mean_val + std_val)
        lower_bounds.append(mean_val - std_val)
    
    return indices, means, [lower_bounds, upper_bounds]

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gamma", type=float, default=1, help="Discount factor for Q-learning")
    arg_parser.add_argument("--activation", type=str, default='sigmoid', help="Activation function for the neural network (cos, sigmoid, tanh)")
    arg_parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for the neural network")
    arg_parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy action selection")
    arg_parser.add_argument("--decay", action='store_true', help="Whether to decay epsilon over time", default=False)

    env = gym.make("CartPole-v1", max_episode_steps=100)
    env.reset(seed=42)

    # parameters = arg_parser.parse_args()
    # gamma = parameters.gamma
    # activation = parameters.activation
    # lr = parameters.lr
    # epsilon = parameters.epsilon
    # decay = parameters.decay

    for lr in [0.0001, 0.0005, 0.001]:
        for activation in ['cos', 'sigmoid', 'tanh']:
            for decay in [False, True]:
                for epsilon in [0.1, 0.0]:
                    for gamma in [1, 0.99]:

                        spec = {
                            'episodes': 1000,
                            'gamma': gamma,
                            
                            # to experiment with
                            'activation': activation,
                            'lr': lr,
                            'epsilon': epsilon,
                            'decay': decay
                        }

                        
                        spec_str = '_'.join([f'{k}={v}' for k, v in spec.items()])
                        if os.path.exists(f'dumps/experiment_{spec_str}.json'):
                            print(f"Experiment with spec {spec_str} already exists, skipping...")
                            continue

                        print(f"Running experiment with spec: {spec_str}")

                        estimator = Estimator(
                            state_dim = env.observation_space.shape[0],
                            action_dim = env.action_space.n,
                            hidden_dim = 100,
                            lr = spec['lr']
                        )
                        
                        reward, total_loss = q_learning(
                            env,
                            estimator,
                            spec['episodes'],
                            gamma = spec['gamma'],
                            epsilon = spec['epsilon'],
                            decay = spec['decay']
                        )

                        # dump the spec dict into a key value string

                        os.makedirs("dumps", exist_ok=True)

                        with open(f'dumps/experiment_{spec_str}.json', 'wt') as f:
                            json.dump({
                                'reward': reward,
                                'total_loss': total_loss,
                                'spec': spec
                                }, f)

                        # close the environment
                        env.close()

                        # plot the results, showing loss and reward on a plot with two subplots and displaying a moving average over 50 episodes
                        # Create the figure and subplots with shared x-axis
                        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

                        # print(f"Reward length: {len(reward)}")
                        # print(f"Loss length: {len(total_loss)}")

                        # # Process and plot reward data
                        # x_reward, reward_mean, reward_var = moving_average_with_variance(np.array(reward), window_size=50)
                        # ax1.plot(x_reward, reward_mean, color='blue', label='Moving Avg (500 episodes)')
                        # ax1.fill_between(x_reward, reward_var[0], reward_var[1], alpha=0.3, color='blue')
                        # ax1.set_ylabel('Total Reward')
                        # ax1.set_title('Moving Average of Total Reward with Variance')
                        # reward_tick = (max(reward) - min(reward)) / 40
                        # ax1.yaxis.set_major_locator(plt.MultipleLocator(reward_tick))
                        # ax1.legend()
                        # ax1.grid(True, alpha=0.3)

                        # # Process and plot loss data
                        # x_loss, loss_mean, loss_var = moving_average_with_variance(np.array(total_loss), window_size=500)
                        # ax2.plot(x_loss, loss_mean, color='red', label='Moving Avg (50 episodes)')
                        # ax2.fill_between(x_loss, loss_var[0], loss_var[1], alpha=0.3, color='red')
                        # loss_tick = (max(total_loss) - min(total_loss)) / 40
                        # ax2.set_xlabel('Episodes')
                        # ax2.set_ylabel('Total Loss')
                        # ax2.set_title('Moving Average of Total Loss with Variance')
                        # ax2.yaxis.set_major_locator(plt.MultipleLocator(loss_tick))
                        # ax2.legend()
                        # ax2.grid(True, alpha=0.3)

                        # plt.tight_layout()
                        # # plt.show()