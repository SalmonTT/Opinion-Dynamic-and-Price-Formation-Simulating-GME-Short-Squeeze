import numpy as np


def DoSimulation():

    # --- Initialization --- #
    n = 100 # number of agents
    t = 10 # number of rounds
    r = 0.02  # risk-free interest rate
    a = 0.1  # constant absolute risk aversion coefficient (CARA)
    var = 0.1  # variance of stock in risk premium
    alpha = np.random.uniform(0, 1, n)  # alpha [0,1] is the update propensity parameter.
    eps_BC = np.random.uniform(0, 0.4, n) # epsilon for BC model
    eps_PA = np.full(n, 0.1)
    X = np.random.uniform(1, 10, n) # X is X(t=0) which is the expected price for t=1 (next period)
    A = np.identity(n) # initialize A(t=0) as an identity matrix
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones

    
