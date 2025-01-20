class PerceptionModel():

    def digest(self, observations):
        # returns dict with current prior/posterior
        pass

    def lookahead(self, actions, reconstruct=False):
        # returns dict with imagined actions/observations
        pass

    def reconstruct(self, state):
        pass

    def reset(self, state=None):
        pass

    