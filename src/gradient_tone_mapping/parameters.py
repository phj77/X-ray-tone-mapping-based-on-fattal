
class Parameters:
    def __init__(self):
        self.gamma_values = [0.5, 0.6, 0.7, 0.8] # in paper: sect. 3.1
        self.delta = 0.05 # in paper: sect. 3.1
        self.eps = 1e-8
        self.neighbor_size = 3 # the neighborhood pixel number, in paper: setc. 2.2.2.1
        self.dt = 0.25 # in paper: sect. 2.2.2.2 - time step for gradient descent, in the paper set default 0.25
        self.fc = 0.5 # in paper: sect. 2.2.3 - default value for fc, but we will determine fc from histogram of f
        self.max_iterations = 1000 # maximum number of iterations for gradient descent