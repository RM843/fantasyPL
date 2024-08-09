from optimal_policy import Example


class FantasyPL(Example):
    def is_legal_state(self, state):
        rnd, selection = state
        conds = [len([x for x in selection if x in ['A', 'E']]) < 2,
                 len(selection) == len(set(selection))]
        return all(conds)