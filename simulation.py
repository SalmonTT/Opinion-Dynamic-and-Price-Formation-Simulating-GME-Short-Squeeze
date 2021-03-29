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
        count = 0
        # for every agent, compare his/her opinion with others
        for j in range(n):
            if abs(X[i] - X[j]) <= eps[i]:
                count += 1
                C[i][j] = 1 # mark C[i][j] with 1 to show that they are similar
        C[i] = np.where(C[i] == 1, 1/count, C[i]) # formula after eq.6

    return C

def PriceAdaptive(X, P, n, eps):
    # C is the update matrix
    C = np.zeros((n, n))
    for i in range(n):
        # count is the number of agents whose opinions are closest to the actual price at time t
        count = 0
        for j in range(n):
            if abs((P - X[j])/P) <= eps[i]:
                count += 1
                C[i][j] = 1  # mark C[i][j] with 1 to show that they are similar
        C[i] = np.where(C[i] == 1, 1/count, C[i]) # formula after eq.6
    return C


def DoSimulation():

    # --- Initialization --- #
    n = 100 # number of agents
    t = 10 # number of rounds
    r = 0.02  # risk-free interest rate
    a = 0.1  # constant absolute risk aversion coefficient (CARA)
    var = 0.1  # variance of stock in risk premium
    z_s = 100  # supply per agent (assumed to be constant here under equilibrium condition)
    alpha = np.random.uniform(0, 1, n)  # alpha [0,1] is the update propensity parameter.
    eps_BC = np.random.uniform(0, 0.4, n) # epsilon for BC model
    eps_PA = np.full(n, 0.1)
    X = np.random.uniform(1, 10, n) # X is X(t=0) which is the expected price for t=1 (next period)
    A = np.identity(n) # initialize A(t=0) as an identity matrix
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones

    price_list = []

    # --- simulation --- #
    for round in range(t):

        X_prev = X
        # --- obtain P(t), the equilibrium price on risky asset at time t ---
        P = PricingEq(X, r, a, var, z_s, n)
        price_list.append(P)
        # --- obtain C(t) ---
        # C = BoundedConfidence(X, n, eps_BC)
        C = PriceAdaptive(X_prev, P, n, eps_PA)
        # --- obtain A(t) ---
        A = alpha*C + (1-alpha)*A
        # --- update X(round) -> X(round+1) ---
        X = np.matmul(A, X)

        print("Round ", round)
        # print("P: ", P)
        # print("C: ")
        # print(C)
        # print("A: ")
        # print(A)
        print("X: ")
        print(X)
    print(price_list)

DoSimulation()






