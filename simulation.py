import numpy as np

def PricingEq(X, r, a, var, z_s, n):
    # eq.4 of the paper, however uses x(t) instead of x(t+1)
    discounted_ave_opinion = X.mean() * (1/(1+r))
    risk_premium = a*var*z_s
    P = discounted_ave_opinion - risk_premium
    return P

def BoundedConfidence(X, n, eps):
    # C is the update matrix
    C = np.zeros((n,n))
    for i in range(n):
        # count is the number of agents that are similar to agent i
        # since {i} is included in I(i, x(t-1), p(t-1)), count is at least 1
        count = 0
        # for every agent, compare his/her opinion with others
        for j in range(n):
            if abs(X[i] - X[j]) <= eps[i]:
                count += 1
                C[i][j] = 1 # mark C[i][j] with 1 to show that they are similar
        tmp = np.full(n, 1 / count)
        C[i] = np.where(C[i] == 1, tmp, C[i]) # formula after eq.6

    return C

def DoSimulation():

    # --- Initialization --- #
    n = 100 # number of agents
    t = 10 # number of rounds
    r = 0.02  # risk-free interest rate
    a = 0.1  # constant absolute risk aversion coefficient (CARA)
    var = 0.1  # variance of stock in risk premium
    z_s = 100  # supply per agent (assumed to be constant here under equilibrium condition)
    alpha = np.full(n, .25) # alpha [0,1] is the update propensity parameter. Current it is set to be same across agents
    eps = np.random.uniform(0, 0.4, n) # epsilon for BC model
    print(eps)
    X = np.random.uniform(1, 10, n) # X is X(t=0) which is the expected price for t=1 (next period)
    A = np.identity(n) # alternatively, generate a network structure as initial A(t=0)
    # --- simulation --- #
    for round in range(t):

        # obtain P(t)
        P = PricingEq(X, r, a, var, z_s, n)
        # obtain C(t)
        C = BoundedConfidence(X, n, eps)
        # obtain A(t)
        A = alpha*C + (1-alpha)*A
        # update X
        X = np.matmul(A, X)

        print("Round ", round)
        print("P: ", P)
        print("A: ")
        print(A)
        print("C: ")
        print(C)
        print("X: ")
        print(X)


DoSimulation()






