from tests.test_generic_code import SelectorGameMini


class FantasyPL(SelectorGameMini):
    def is_legal_selection(self, selection):
        # rnd, selection = state
        conds = [len([x for x in selection if x in ['A', 'E']]) < 2,
                 len(selection) == len(set(selection))]
        return all(conds)
