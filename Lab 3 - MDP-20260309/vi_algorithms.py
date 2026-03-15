import numpy as np
import heapq
from abc import ABC, abstractmethod
from constants import MAX_ITERATIONS, EPSILON, GAMMA

class VISolver(ABC):
    def __init__(self, game, max_iteratios=MAX_ITERATIONS, epsilon=EPSILON, gamma=GAMMA):
        self.game = game
        self.max_iterations = max_iteratios
        self.epsilon = epsilon
        self.gamma = gamma

        self.states = self.game.get_states()
        self.actions = self.game.get_actions()
        self.V = np.zeros(self.states.n)
        print(f"Initialized {self.__class__.__name__} with {self.states.n} states and {self.actions.n} actions.")

    @abstractmethod
    def compute(self):
        pass

    def _has_converged_to_vstar(self, v_star):
        return np.linalg.norm(self.V - v_star, ord=2) < self.epsilon

    def _get_max_reward(self, s, v_source=None):
        if v_source is None:
            v_source = self.V
            
        action_values = []
        for a in range(self.actions.n):
            transitions = self.game.get_probability(s, a)

            immediate_reward = transitions[0][2]
            sum_future_value = 0
            for prob, next_state, reward, done in transitions:
                sum_future_value += prob * (0 if done else v_source[next_state])
            
            action_values.append(immediate_reward + self.gamma * sum_future_value)
        
        return max(action_values)

class ValueIterationSolver(VISolver):
    def __init__(self, game, max_iteratios=MAX_ITERATIONS, epsilon=EPSILON, gamma=GAMMA):
        super().__init__(game, max_iteratios, epsilon, gamma)

    def compute(self, v_star=None, track_history=False):
        iterations = 0
        history = None

        if v_star is None:
            v_star = np.zeros_like(self.V)

        if track_history:
            history = [np.linalg.norm(self.V - v_star, ord=2)]

        while iterations < self.max_iterations:
            delta = 0
            V_k = self.V.copy()

            for s in range(self.states.n):
                if iterations >= self.max_iterations:
                    break

                self.V[s] = self._get_max_reward(s, v_source=V_k)
                iterations += 1
                delta = max(delta, abs(self.V[s] - V_k[s]))

                if track_history:
                    history.append(np.linalg.norm(self.V - v_star, ord=2))

            converged = (
                self._has_converged_to_vstar(v_star)
                if v_star is not None
                else delta < self.epsilon
            )

            if converged:
                print(f"Value Iteration converged after {iterations} state updates.")
                return history if track_history else iterations
            
        return history if track_history else iterations

class GaussSeidelValueIterationSolver(VISolver):
    def __init__(self, game, max_iteratios=MAX_ITERATIONS, epsilon=EPSILON, gamma=GAMMA):
        super().__init__(game, max_iteratios, epsilon, gamma)

    def compute(self, v_star, track_history=False):
        iterations = 0
        history = None

        if track_history:
            history = [np.linalg.norm(self.V - v_star, ord=2)] 

        while iterations < self.max_iterations:
            delta = 0

            for s in range(self.states.n):
                if iterations >= self.max_iterations:
                    break

                old_v = self.V[s]
                self.V[s] = self._get_max_reward(s)
                iterations += 1
                delta = max(delta, abs(self.V[s] - old_v))

                if track_history:
                    history.append(np.linalg.norm(self.V - v_star, ord=2))

            converged = (
                self._has_converged_to_vstar(v_star)
                if v_star is not None
                else delta < self.epsilon
            )

            if converged:
                print(f"Gauss-Seidel converged after {iterations} state updates.")
                return history if track_history else iterations
            
        print(f"Gauss-Seidel completed {iterations} state updates.")
        return history if track_history else iterations

class PrioritiesedSweepingValueIterationSolver(VISolver):
    def __init__(self, game, max_iteratios=MAX_ITERATIONS, epsilon=EPSILON, gamma=GAMMA):
        super().__init__(game, max_iteratios, epsilon, gamma)
        self.predecessors = {s: set() for s in range(self.states.n)}
        for s in range(self.states.n):
            for a in range(self.actions.n):
                for prob, next_s, _, _ in self.game.get_probability(s, a):
                    if prob > 0: self.predecessors[next_s].add(s)

    def push_to_queue(self, priority_queue, state_in_queue, s):
        bellman_error = abs(self._get_max_reward(s) - self.V[s])
        
        if bellman_error > self.epsilon and s not in state_in_queue:
            state_in_queue.add(s)
            heapq.heappush(priority_queue, (-bellman_error, s))

    def compute(self, v_star, track_history=False):
        priority_queue = []
        state_in_queue = set()

        if v_star is None:
            v_star = np.zeros_like(self.V)

        if track_history:
            history = [np.linalg.norm(self.V - v_star, ord=2)]

        for s in range(self.states.n): 
            self.push_to_queue(priority_queue, state_in_queue, s)
        
        iterations = 0
        while priority_queue and iterations < self.max_iterations:
            neg_err, s = heapq.heappop(priority_queue)
            state_in_queue.discard(s)

            self.V[s] = self._get_max_reward(s)
            iterations += 1

            if track_history:
                history.append(np.linalg.norm(self.V - v_star, ord=2))

            for p in self.predecessors[s]: 
                self.push_to_queue(priority_queue, state_in_queue, p)

            if v_star is not None and self._has_converged_to_vstar(v_star):
                print(f"Prioritized Sweeping converged after {iterations} state updates.")
                return history if track_history else iterations

        print(f"Prioritized Sweeping completed {iterations} state updates.")
        return history if track_history else iterations

class PolicyIterationSolver(VISolver):
    def __init__(self, game, max_iterations=MAX_ITERATIONS, epsilon=EPSILON, gamma=GAMMA):
        super().__init__(game, max_iterations, epsilon, gamma)
        
        
    def compute(self, v_star, track_history=False):
        total_updates = 0
        self.V = np.zeros(self.states.n)
        self.policy = np.random.randint(0, self.actions.n, self.states.n)

        history = []

        if track_history:
            history.append(np.linalg.norm(self.V - v_star, ord=2))

        self.v_star = v_star.copy()

        for _ in range(self.max_iterations):
            updates, eval_history = self._policy_evaluation(v_star, track_history)
            total_updates += updates

            if track_history:
                history.extend(eval_history)

            if self._has_converged_to_vstar(v_star):
                print(f"Policy Iteration converged after {total_updates} state updates.")
                return history if track_history else total_updates
            
            self._policy_improvement() 
            
        return history if track_history else total_updates

    def _policy_evaluation(self, v_star, track_history):
        updates = 0
        eval_history = []

        while True:
            delta = 0
            V_old = self.V.copy()

            for s in range(self.states.n):
                a = self.policy[s]
                transitions = self.game.get_probability(s, a)

                immediate_reward = transitions[0][2]

                sum_future_value = 0
                for prob, next_state, reward, done in transitions:
                    sum_future_value += prob * (0 if done else V_old[next_state])
                
                sum_future_value *= self.gamma

                self.V[s] = immediate_reward + sum_future_value
                updates += 1

                if track_history:
                    eval_history.append(np.linalg.norm(self.V - v_star, ord=2))

                delta = max(delta, abs(V_old[s] - self.V[s]))

            if self._has_converged_to_vstar(v_star) or delta < self.epsilon:
                break

        return updates, eval_history

    def _policy_improvement(self):
        policy_stable = True

        for s in range(self.states.n):
            old_action = self.policy[s]

            action_values = []

            for a in range(self.actions.n):
                transitions = self.game.get_probability(s, a)

                immediate_reward = transitions[0][2]
                sum_future_value = 0

                for prob, next_state, reward, done in transitions:
                    sum_future_value += prob * (0 if done else self.V[next_state])

                sum_future_value *= self.gamma

                action_values.append(immediate_reward + sum_future_value)
            
            new_action = np.argmax(action_values)
            self.policy[s] = new_action
            
            if old_action != new_action:
                policy_stable = False
                
        return policy_stable