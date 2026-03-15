import argparse
import numpy as np
import matplotlib.pyplot as plt

from games import Game, FrozenGame, TaxiGame
from vi_algorithms import ValueIterationSolver, PolicyIterationSolver, GaussSeidelValueIterationSolver, PrioritiesedSweepingValueIterationSolver


def run_vi(game: Game | None):
    vi_solver = ValueIterationSolver(game)
    vi_solver.compute()
    v_star = vi_solver.V.copy()

    vi_solver = GaussSeidelValueIterationSolver(game)
    vi_solver.compute(v_star)

    vi_solver = PrioritiesedSweepingValueIterationSolver(game)
    vi_solver.compute(v_star)


def run_pi(game: Game | None):
    vi_solver = ValueIterationSolver(game)
    vi_solver.compute()
    v_star = vi_solver.V.copy()
    
    results = []
    for trial in range(5):
        pi_solver = PolicyIterationSolver(game)
        num_iters = pi_solver.compute(v_star)
        results.append(num_iters)
        print(f"Trial {trial + 1}: Policy Iteration converged in {num_iters} iterations.")
    
    avg_iters = np.mean(results)
    print(f"\nAverage number of iterations for {game.__class__.__name__}: {avg_iters}")
    return avg_iters


def plot_convergence_graph(game):
    solver = ValueIterationSolver(game)
    solver.compute()

    v_star = solver.V.copy() 

    solvers = {
        "Value Iteration": ValueIterationSolver(game),
        "Gauss-Seidel": GaussSeidelValueIterationSolver(game),
        "Prioritized Sweeping": PrioritiesedSweepingValueIterationSolver(game),
        # "Policy Iteration": PolicyIterationSolver(game)
    }
    
    plt.figure(figsize=(10, 6))

    for name, solver in solvers.items():
        print(f"Running {name}...")

        history = solver.compute(v_star, track_history=True) 
        plt.plot(history, label=name)

    plt.yscale('log')
    plt.xlabel('Number of State Updates (Iterations)')
    plt.ylabel('||V - V*||2 (L2 Norm)')
    plt.title(f'Convergence Speed Analysis: {game.__class__.__name__}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Value Iteration on a specified game.")
    parser.add_argument("--game", type=str, choices=["frozen", "taxi"], default="frozen",
                        help="The game to run Value Iteration on. Choices are 'frozen' and 'taxi'. Default is 'frozen'.")
    
    args = parser.parse_args()
    
    game = None
    if args.game == "frozen":
        game = FrozenGame()
    elif args.game == "taxi":
        game = TaxiGame()

    run_vi(game)
    run_pi(game)
    plot_convergence_graph(game)