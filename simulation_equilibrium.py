import numpy as np

"""
This is the simulation described in paper with the equilibrium assumption
"""

def PricingEq(X, r, a, var, z_s, n):
    # eq.4 of the paper, however uses x(t) instead of x(t+1)
    discounted_ave_opinion = X.mean() * (1/(1+r))
    # print("discounted ave opinion")
    # print(discounted_ave_opinion)
    risk_premium = a*var*z_s
    # print("risk premium")
    # print(risk_premium)
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
            if abs((X[i] - X[j])/X[i]) <= eps[i]:
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
        if count != 0 :
            C[i] = np.where(C[i] == 1, 1/count, C[i]) # formula after eq.6
    return C


def DoSimulation():

    # --- Initialization --- #
    n = 10  # number of agents
    t = 20  # number of rounds
    r = 0.0007700  # risk-free interest rate
    a = 0.1  # constant absolute risk aversion coefficient (CARA)
    var = 0.000173627  # variance of stock in risk premium
    z_s = 100  # supply per agent (assumed to be constant here under equilibrium condition)
    alpha = np.random.uniform(0, 1, n)  # alpha [0,1] is the update propensity parameter.
    eps_BC = np.random.uniform(0, 0.1, n) # epsilon for BC model
    eps_PA = np.random.uniform(0, 0.1, n)
    X_BC = np.random.normal(394.730011, 100, n) # X is X(t=0) which is the expected price for t=1 (next period)
    # X_BC = np.random.uniform(1, 10, n)
    X_PA = X_BC
    print(X_PA)
    A_BC = np.identity(n) # initialize A(t=0) as an identity matrix
    A_PA = A_BC
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones

    price_list_BC = []
    std_BC = [X_BC.std()]
    price_list_PA = []
    std_PA = [X_PA.std()]

    # --- simulation --- #
    for round in range(t):

        X_prev = X_BC
        # --- obtain P(t), the equilibrium price on risky asset at time t ---
        P = PricingEq(X_BC, r, a, var, z_s, n)
        price_list_BC.append(P)
        # --- obtain C(t) ---
        C = BoundedConfidence(X_BC, n, eps_BC)
        # C = PriceAdaptive(X_prev, P, n, eps_PA)
        # --- obtain A(t) ---
        A_BC = alpha*C + (1-alpha)*A_BC
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


    for round in range(t):

        X_prev = X_PA
        # --- obtain P(t), the equilibrium price on risky asset at time t ---
        P = PricingEq(X_PA, r, a, var, z_s, n)
        price_list_PA.append(P)
        # --- obtain C(t) ---
        # C = BoundedConfidence(X, n, eps_BC)
        C = PriceAdaptive(X_prev, P, n, eps_PA)
        # --- obtain A(t) ---
        A_PA = alpha*C + (1-alpha)*A_PA
        # --- update X(round) -> X(round+1) ---
        X_PA = np.matmul(A_PA, X_PA)
        std_PA.append(X_PA.std())

        # print("Round ", round)
        # print("P: ", P)
        # print("C: ")
        # print(C)
        # print("A: ")
        # print(A_PA)
        # print("X: ")
        # print(X_PA)
    print('price list BC')
    print(price_list_BC)
    print('price list PA')
    print(price_list_PA)
    print('std BC')
    print(std_BC)
    print('std PA')
    print(std_PA)

DoSimulation()






