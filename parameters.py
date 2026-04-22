
class Parameters:
    def __init__(self):
        self.gamma_values = [0.51, 0.50, 0.43, 0.45] # in paper: sect. 3.1
        self.delta = 0.05 # in paper: sect. 3.1
        self.eps = 1e-8
        self.histogram_bins = 256