MAJORITY = "majority"
AVERAGE  = "average"
WEIGHTED = "weighted"
VOTING   = (MAJORITY, AVERAGE, WEIGHTED)


class Ensemble:
    def __init__(self, experiments: list, voting: str, experiments_dir = ):
        assert voting in VOTING, "Ensemble.__init__: valid voting schemes include {}, {} and {}".format(*VOTING)
        self.voting = voting
        
    def load_experiments(self):
