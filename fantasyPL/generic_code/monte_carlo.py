from collections import defaultdict
import math

from fantasyPL.generic_code.binary_tree import Node
from fantasyPL.generic_code.reinforment_learning import PolicyOrValueIteration, INITIAL_STATE


class MCTS(PolicyOrValueIteration):
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, problem_obj, exploration_weight=1.0):
        super().__init__(problem_obj)
        self.Q = defaultdict(float)  # total reward of each node
        self.N = defaultdict(float)  # total visit count for each node
        self.children = dict()  # children of each node: key is explored node, value is set of children
        self.exploration_weight = exploration_weight
    def algorithm(self,gamma,epsilon):
        ''''''
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def run_mcts(self, node, num_rollout):
        "Run on iteration of select -> expand -> simulation(rollout) -> backup"
        path = self.select(node)
        leaf = path[-1]
        # self.expand(leaf)
        reward = 0
        for i in range(num_rollout):
            reward += self.simulate(leaf)
        reward = reward /num_rollout
        self.backup(path, reward)

    def select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded

        state = node.node_id
        allowed_actions = self.get_allowed_actions(state)
        children = []
        for action in allowed_actions:
            next_state = self.problem_obj.transition_model(state, action, start_state=INITIAL_STATE == state)
            rnd, selection = next_state
            value = self.problem_obj.reward_function(next_state)
            children.append(Node(node_id=next_state, value=value, terminal=self.problem_obj.rounds.stop == rnd))
        node.children = children
        self.children[node] = children

    def simulate(self, node):
        "Run a random simulation from node as starting point"
        score = 0
        while True:
            score+=node.value
            if node.is_terminal():
                return score
            self.expand( node)
            node = node.find_random_child()

    def backup(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        # a node is fully expanded if and only if all children are explored
        is_all_children_expanded = all(n in self.children for n in self.children[node])
        if not is_all_children_expanded:
            raise ValueError("Can only select fom fully expanded node")

        log_N_parent = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_parent / self.N[n]
            )

        return max(self.children[node], key=uct)
