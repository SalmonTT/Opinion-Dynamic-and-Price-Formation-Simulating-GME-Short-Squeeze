import numpy as np
import decimal
import random

def getCurrentZ(a, var, r, p, X):
    '''
    given the current price (p(t)) and current opinion (x(t), note x(t) is expected price for
    period t+1), we we obtain z(t) which is the optimal # of shares held for current period t.
    '''
    # note that Z must be a discrete number
    Z = (1/(a*var)) * (X - (1+r)*p) # equation 2
    Z = Z.astype(int)
    return Z

def getAction(Z_now, Z_prev, actions, n):
    '''
    Compare the optimal # of shares held by agents during this period and # shares held during last period
    and update the actions for current period
    '''
    for i in range(n):
        if(Z_now[i] > Z_prev[i]):
            actions[i] = 1 # Buy
        if (Z_now[i] < Z_prev[i]):
            actions[i] = -1  # Sell
        else:
            actions[i] = 0 # Hold
    return actions

def getOrderPrice(action, beta, p, X, orderPrice, n, r):
    '''
    return the prices of orders for each agent at current period t
    '''
    for i in range(n):
        if(action[i] == 1): # buy order
            # returns a float with two decimal places that is greater than or equal to (1+r)*p(t)-beta and less than x(t)
            orderPrice[i] = decimal.Decimal(random.randrange(int(((1+r)*p - beta[i]) * 100), int(X[i] * 100))) / 100
        elif(action[i] == -1): # sell order
            # returns a float with two decimal places that is greater than or equal to x(t) and less than (1+r)*p(t)+beta
            orderPrice[i] = decimal.Decimal(random.randrange(int(X[i] * 100), int(((1+r)*p + beta[i]) * 100))) / 100
        else: # hold
            return orderPrice
    return orderPrice

def updatePrice(actions, orderPrices, n):
    '''
    Order matching:
        - ignore the traditional order matching mechanism and fulfill everyone's orders
        - After orders are executed, we update price p(t) using the Bid-Ask prices.
    Returns the new price for next period
    '''
    bids = [] # an array of all bidding prices
    asks = [] # an array of al asking prices
    for i in range(n):
        if(actions[i] == 1):
            bids.append(orderPrices[i])
        elif(actions[i] == -1):
            asks.append(orderPrices[i])
    ave_bids = sum(bids)
    ave_asks = sum(asks)
    return (ave_asks+ave_bids)/2

def updateXwithBC(X, n, eps):
    '''
       Get C the update matrix to update X using BC
    '''
    # C is the update matrix
    C = np.zeros((n, n))
    for i in range(n):
        # count is the number of agents that are similar to agent i
        count = 0
        # for every agent, compare his/her opinion with others
        for j in range(n):
            if abs(X[i] - X[j]) <= eps[i]:
                count += 1
                C[i][j] = 1  # mark C[i][j] with 1 to show that they are similar
        C[i] = np.where(C[i] == 1, 1 / count, C[i])  # formula after eq.6

    return C

def updateXwithPA(X, P, n, eps):
    '''
       Get C the update matrix to update X using PA
    '''
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

    beta = np.random.uniform(0, 10, n) # An array, risk preference when it comes to placing orders
    actions = np.zeros(n) # current actions for each agent (discrete values of 1, -1 and 0 - Buy Sell Hold)
    orderPrice = np.zeros(n) # prices of current order for each agent
    Z = np.random.randint(100,1000,n) # each agent holds between 10 to 1000 shares
    p = 100 # initialize p(t=0) to be 100


    var = 0.1  # variance of stock in risk premium
    alpha = np.random.uniform(0, 1, n)  # alpha [0,1] is the update propensity parameter.
    eps_BC = np.random.uniform(0, 0.4, n) # epsilon for BC model
    eps_PA = np.full(n, 0.1)
    X = np.random.uniform(1, 10, n) # X is X(t=0) which is the expected price for t=1 (next period)
    A = np.identity(n) # initialize A(t=0) as an identity matrix
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones

    # --- Simulation --- #

    for round in range(t):
        # Portfolio holding dynamics
        Z_prev = Z
        Z = getCurrentZ(a, var, r, p, X)

        # Price dynamics
        actions = getAction(Z, Z_prev, actions, n)
        orderPrice = getOrderPrice(actions, beta, p, X, orderPrice, n, r)
        p = updatePrice(actions, orderPrice, n)

        # Expected price dynamics
        C = updateXwithBC(X, n, eps_BC)
        # C = updateXwithPA(X_prev, P, n, eps_PA)
        A = alpha*C + (1-alpha)*A
        X = np.matmul(A, X)


        # console.log
        print("Round ", round)
        print("Price: ", p)


DoSimulation()

