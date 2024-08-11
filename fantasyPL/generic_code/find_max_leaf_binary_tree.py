from fantasyPL.generic_code.binary_tree import make_binary_tree, Node
from  fantasyPL.generic_code.monte_carlo import MCTS
import argparse

from fantasyPL.generic_code.reinforment_learning import INITIAL_STATE


def mcts_playout(initial_selection, num_iter, num_rollout, exploration_weight,problem_obj):
    # root, leaf_nodes_dict = make_binary_tree(depth=depth)
    # leaf_nodes_dict_sorted = sorted(leaf_nodes_dict.items(), key=lambda x: x[1], reverse=True)
    # print("Expected (max) leaf node: {}, value: {}".format(leaf_nodes_dict_sorted[0][0],
    #                                                        leaf_nodes_dict_sorted[0][1]))
    # print("Expected (min) leaf node: {}, value: {}".format(leaf_nodes_dict_sorted[-1][0],
    #                                                        leaf_nodes_dict_sorted[-1][1]))
    #

    # value = problem_obj.reward_function((problem_obj.rounds.start,initial_selection))
    root_node = Node(node_id=INITIAL_STATE,value=0,terminal=False)

    mcts = MCTS(exploration_weight=exploration_weight,problem_obj=problem_obj)
    mcts.get_selections_superset()
    while True:
        # we run MCTS simulation for many times
        for _ in range(num_iter):
            mcts.run_mcts(root_node, num_rollout=num_rollout)
        # we choose the best greedy action based on simulation results
        # root_node = mcts.choose(root_node)
        # we repeat until root is terminal
        # if root_node.is_terminal():
        print("Found optimal (max) leaf node: {}, value: {}".format(root_node, root_node.value))
        #     return root_node.value


if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='MCTS main runner')
    parser.add_argument("--num_iter", type=int, default=50,
                        help="number of MCTS iterations starting from a specific root node")
    parser.add_argument("--num_rollout", type=int, default=1, help="number of rollout simulations in a MCTS iteration")
    parser.add_argument("--depth", type=int, default=12, help="number of depth of the binary tree")
    parser.add_argument("--exploration_weight", type=float, default=51, help="exploration weight, c number in UCT")
    args = parser.parse_args()
    mcts_playout(args.depth, args.num_iter, args.num_rollout, args.exploration_weight)