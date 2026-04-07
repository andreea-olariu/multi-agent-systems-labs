import random
import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy(q_table: dict, state: int, n_actions: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return int(np.argmax(q_table[state]))
    
def smooth(rewards: list[float], window: int = 50) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(rewards, kernel, mode="valid")

def plot_comparison(results: dict, title: str, save_path: str | None = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
 
    colors = plt.cm.tab10.colors
 
    for idx, (label, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
 
        smoothed = smooth(data["train_rewards"], window=50)
        ax1.plot(smoothed, label=label, color=color, linewidth=1.5)
 
        ax2.plot(data["eval_steps"], data["eval_rewards"],
                 label=label, color=color, linewidth=1.5, marker="o", markersize=3)
 
    ax1.set_title("Training reward (smoothed, window=50)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)
 
    ax2.set_title("Evaluation average reward (greedy policy)")
    ax2.set_xlabel("Training episode")
    ax2.set_ylabel("Avg reward over 50 eval episodes")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")
 
    # plt.show()


