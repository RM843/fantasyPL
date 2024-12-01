import random
from collections import defaultdict
import math
from typing import List, Any, Dict

from fantasyPL.generic_code.binary_tree import Node
from reinforcement_learning.reinforment_learning import PolicyOrValueIteration, INITIAL_STATE
from helper_methods import timing_decorator


class NodeDict:
    def __init__(self, data: Dict):
        self._data = data

    def __setitem__(self, key, value):
        if isinstance(key, Node):
            key = key.node_id
        self._data[key] = value

    def __contains__(self, key):
        return key.node_id in self._data

    def __getattr__(self, item: str) -> Any:
        """
        Custom attribute access method.

        Args:
            item (str): The attribute to access.

        Returns:
            Any: The value corresponding to the attribute.

        Raises:
            AttributeError: If the attribute is not found in the dictionary.
        """
        if item in self._data:
            return self._data[item]
        raise AttributeError(f"'DictPlus' object has no attribute '{item}'")

    def __iter__(self):
        """
        Iterator method.

        Returns:
            Iterator: An iterator over the dictionary keys.
        """
        return iter(self._data)

    def __getitem__(self, key):
        """
        Method to get an item by key.

        Args:
            key: The key of the item to retrieve.

        Returns:
            Any: The value corresponding to the key.

        Raises:
            KeyError: If the key is not found in the dictionary.
        """
        if isinstance(key, Node):
            key = key.node_id
        return self._data[key]

    def keys(self):
        """
        Method to return the keys of the dictionary.

        Returns:
            KeysView: The keys of the dictionary.
        """
        return self._data.keys()

    def items(self):
        """
        Method to return the items of the dictionary.

        Returns:
            ItemsView: The items of the dictionary.
        """
        return self._data.items()

    def values(self):
        """
        Method to return the values of the dictionary.

        Returns:
            ValuesView: The values of the dictionary.
        """
        return self._data.values()

    def get(self, key, default=None):
        if isinstance(key, Node):
            key = key.node_id

        return self._data.get(key, default)


class MCTS(PolicyOrValueIteration):
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, problem_obj, exploration_weight=1.0):
        super().__init__(problem_obj)
        self.Q = NodeDict(defaultdict(float))  # total reward of each node
        self.N = NodeDict(defaultdict(float))  # total visit count for each node
        self.explored = NodeDict(dict())  # children of each node: key is explored node, value is set of children
        self.exploration_weight = exploration_weight

    def algorithm(self, gamma, epsilon):
        ''''''

    def get_strat(self) -> List:

        self.strat = []  # Initialize the strategy list
        state = self.initial_state
        while True:
            max_action_value = float("-inf")
            max_action = None

            # not ready to derive strat yet
            if self.explored.get(state, []) == []:
                return
            action_values = [self.Q[x] / self.N[x] for x in self.explored[state] if self.N[x] != 0]
            for i, action_value in enumerate(action_values):
                if action_value > max_action_value:
                    max_action_value = action_value
                    max_action = i
            new_state = self.explored[state][max_action]
            action = self.get_action_that_creates_state(old_state=state,
                                                        new_state=new_state,
                                                        start_state= self.initial_state == state)
            self.strat.append({"state": state.node_id, "action": action, "value": new_state.value})
            state = new_state
            if self.is_final(state.node_id):
                break

        return self.strat

    def get_action_that_creates_state(self, old_state, new_state, start_state):
        for act in self.get_allowed_actions(old_state.node_id):
            if new_state.node_id == self.problem_obj.transition_model(state=old_state.node_id,
                                                                      action=act,
                                                                      start_state=start_state):
                return act
        raise Exception("No action found")

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

    def run(self, iterations, initial_node, num_rollout):
        self.initial_state = initial_node
        self.get_all_children(initial_node)
        # node = initial_node
        for _ in range(iterations):
            path = self.select(initial_node)
            leaf = path[-1]
            if not leaf.terminal:

                leaf = self.expand(leaf)
                path = path+[leaf]
            else:
                pass
            av_return = self.simulate(path, rollouts=num_rollout)
            self.backup(path, av_return)


    def select(self, node):
        "Find an unexplored descendant of `node`"
        path = []
        while True:
            path.append(node)
            is_leaf = node not in self.explored
            if is_leaf or node.terminal:
                # node is either unexplored or terminal
                return path
            unexplored_child_nodes = self.get_unexplored_children(node)
            # explore nodes in order of value
            unexplored_child_nodes = set(sorted(unexplored_child_nodes, key=lambda x: x.value))
            if unexplored_child_nodes != set():
                n = unexplored_child_nodes.pop()
                path.append(n)
                return path
            node = self.select_child_node(node,exploration_rate=self.exploration_weight)  # descend a layer deeper

            # node = self._uct_select(node)  # descend a layer deeper
    def get_unexplored_children(self, node):
        self.get_all_children(node)
        explored_ids = set([x for x in self.explored])
        children_ids = set([x.node_id for x in node.children])
        unexplored_ids = children_ids - explored_ids
        return [[y for y in node.children if x == y.node_id][0] for x in unexplored_ids]

    def expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.explored or node.terminal:
            return node  # already expanded
        self.get_all_children(node)
        assert not (node.children == [] and not node.terminal)
        self.explored[node] = node.children
        return node.find_random_child()

    def get_all_children(self, node):
        if node.children != [] or node.terminal:
            return
        state = node.node_id
        allowed_actions = self.get_allowed_actions(state)
        children = []
        for action in allowed_actions:
            next_state = self.problem_obj.transition_model(state, action, start_state=INITIAL_STATE == state)
            rnd, selection = next_state
            value = self.problem_obj.reward_function(next_state)
            assert value != 0
            children.append(Node(node_id=next_state, value=value, terminal=self.problem_obj.rounds.stop == rnd))
        node.children = children

    def simulate(self, path, rollouts):
        "Run a random simulation from node as starting point"
        average_score = 0
        score_so_far = 0
        for node in path:
            score_so_far += node.value
        if path[-1].terminal:
            return score_so_far

        self.get_all_children( path[-1])
        node = path[-1].find_random_child()

        for i in range(rollouts):
            score = 0
            while True:

                score += node.value
                # count += 1
                if node.is_terminal():
                    average_score +=( score +score_so_far)
                    break
                # # need to do this each time as too expensive to get upfront for all nodes
                self.get_all_children(node)

                # if node in self.explored:
                #     if all(n in self.explored for n in self.explored[node]):
                #         node = self.select_child_node(node, exploration_rate=self.exploration_weight)
                #         continue
                node = node.find_random_child()

        return average_score / rollouts

    def backup(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            # new_value =  (reward + ( self.Q[node]* self.N[node]))/( self.N[node]+1)
            # print(abs(new_value-self.Q[node]))
            self.Q[node] += reward
            self.N[node] += 1
    # @timing_decorator
    def select_child_node(self, node, exploration_rate):
        is_all_children_expanded = all(n in self.explored for n in self.explored[node])
        assert all(n in self.N for n in self.explored[node])
        if not is_all_children_expanded:
            raise ValueError("Can only select fom fully expanded node")

        if random.random() < exploration_rate:
            return random.choice(self.explored[node])

        else:
            def state_val(n):
                "Upper confidence bound for trees"
                return self.Q[n] / self.N[n]
            # child_values = [self.Q[x] for x in self.explored[node]]
            # max_index = child_values.index(max(child_values))
            return max(self.explored[node], key=state_val)

    @timing_decorator
    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        # a node is fully expanded if and only if all children are explored
        is_all_children_expanded = all(n in self.explored for n in self.explored[node])
        if not is_all_children_expanded:
            raise ValueError("Can only select fom fully expanded node")
        assert all(n in self.N for n in self.explored[node])

        log_N_parent = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_parent / self.N[n]
            )

        return max(self.explored[node], key=uct)

    def get_state_values(self):
        return {x: self.Q[x] / self.N[x] for x in self.N if self.N[x] != 0}
