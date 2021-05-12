import numpy as np

"""
Simulation Model 1
This is the simulation described in paper with the equilibrium assumption
"""

def PricingEq(X, r):
    # eq.4 of the paper, however uses x(t) instead of x(t+1)
    discounted_ave_opinion = X.mean() * (1/(1+r))
    P = discounted_ave_opinion
    return P

def BoundedConfidence(X, n, eps):
    # C is the update matrix
    C = np.zeros((n,n))
    for i in range(n):
        # count is the number of agents that are similar to agent i
        count = 0
        # for every agent, compare his/her opinion with others
        for j in range(n):
            if abs((X[i] - X[j])/X[i]) <= eps[i]:
                count += 1
                C[i][j] = 1 # mark C[i][j] with 1 to show that they are similar
        # print(count)
        C[i] = np.where(C[i] == 1, 1/count, C[i]) # formula after eq.6

    return C

def DoSimulation():

    # --- Initialization --- #
    n = 100  # number of agents
    t = 50 # number of rounds
    r = 0.0007700  # risk-free interest rate
    # a = 1  # constant absolute risk aversion coefficient (CARA)
    # var = 0.445093979  # variance of stock in risk premium
    # z_s = 0  # supply per agent (assumed to be constant here under equilibrium condition)
    alpha = np.random.uniform(0, 0.2, n)  # alpha [0,1] is the update propensity parameter.
    # alpha = np.full(n, 0.2)
    # print(alpha)
    eps_BC = np.random.uniform(0, 0.05, n) # epsilon for BC model
    # eps_BC = np.full(n, 0.2)
    # print(eps_BC)
    # eps_PA = np.random.uniform(0, 0.05, n)
    X_BC = np.random.normal(19.95, 3, n) # X is X(t=0) which is the expected price for t=1 (next period)
    # print(X_BC)
    # X_BC = np.random.uniform(1, 10, n)
    A_BC = np.identity(n) # initialize A(t=0) as an identity matrix
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones

    price_list_BC = []
    std_BC = [X_BC.std()]

    # --- simulation --- #
    for rd in range(t):

        # print("round", rd)
        # X_prev = X_BC
        # --- obtain P(t), the equilibrium price on risky asset at time t ---
        P = PricingEq(X_BC, r)
        price_list_BC.append(P)
        # --- obtain C(t) ---
        C = BoundedConfidence(X_BC, n, eps_BC)
        # for j in range(n):
        #     print(sum(C[j]))
        # C = PriceAdaptive(X_prev, P, n, eps_PA)
        # --- obtain A(t) ---
        for i in range(n):
            for j in range(n):
                A_BC[i][j] = alpha[i]*C[i][j] + (1-alpha[i])*A_BC[i][j]
        # --- update X(round) -> X(round+1) ---
        X_BC = np.matmul(A_BC, X_BC)
        std_BC.append(X_BC.std())

        # print("Round ", round)
        # print("P: ", P)
        # print("C: ")
        # print(C)
        # print("A: ")
        # print(A_BC)
        # print("X: ")
        # print(X)

    # print('price list BC')
    print(price_list_BC)
    # print('std BC')
    print(std_BC)
    print("next round")

for i in range(5):
    DoSimulation()






