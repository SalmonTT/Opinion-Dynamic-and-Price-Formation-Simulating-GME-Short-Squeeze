import numpy as np
import random

def getDeltaZ(a, var, r, p, X):
    '''
    given the current price (p(t)) and current opinion (x(t), note x(t) is expected price for
    period t+1), we we obtain z(t) which is the optimal # of shares held for current period t.
    '''
    print("price is", p)
    print("this is X")
    print(X)
    # note that Z must be a discrete number
    print(1/(a*var))
    Z = (1/(a*var)) * (X - (1+r)*p) # equation 2
    Z = Z.astype(int)
    return Z

def updateCurrentZ(Z_current, Z_delta, n):
    for i in range(n):
        if (Z_current[i] + Z_delta[i]) < 0:
            Z_current[i] = 0
        else:
            Z_current[i] += Z_delta[i]
    return Z_current

def getAction(Z_delta, Z_current, actions, n):
    '''
    Compare the optimal # of shares held by agents during this period and # shares held during last period
    and update the actions for current period
    '''
    for i in range(n):
        if(Z_delta[i] > 0):
            actions[i] = 1 # Buy
        elif (Z_delta[i] < 0):
            if (Z_current[i] > 0): # only if agent currently owns share
                actions[i] = -1  # Sell
            else: actions[i] = 0  # agent does not own share, do nothing
        else:
            actions[i] = 0 # Hold
        print("Z_delta: ", Z_delta[i])
        print("action: ",actions[i])
    return actions

def getOrderPrice(action, beta, p, X, orderPrice, n, r):
    '''
    return the prices of orders for each agent at current period t
    '''
    for i in range(n):
        if(action[i] == 1): # buy order
            # returns a float with two decimal places that is greater than or equal to (1+r)*p(t)-beta and less than x(t)
            orderPrice[i] = random.uniform(((1+r)*p - beta[i]), X[i])
        elif(action[i] == -1): # sell order
            # returns a float with two decimal places that is greater than or equal to x(t) and less than (1+r)*p(t)+beta
            # orderPrice[i] = decimal.Decimal(random.randrange(int(X[i] * 100), int(((1+r)*p + beta[i]) * 100))) / 100
            orderPrice[i] = random.uniform(X[i], (1 + r) * p - beta[i])
        else: # hold

            return orderPrice
    return orderPrice

def updatePrice(Z_delta, actions, orderPrices, n):
    '''
    Order matching:
        - ignore the traditional order matching mechanism and fulfill everyone's orders
        - After orders are executed, we update price p(t) using the Bid-Ask prices.
    Returns the new price for next period
    '''
    bids = [] # an array of all bidding prices
    asks = [] # an array of all asking prices
    order_sum = sum(map(abs, Z_delta)) # sum of the absolute value of orders
    for i in range(n):
        if(actions[i] == 1):
            bids.append(orderPrices[i]*(abs(Z_delta[i]))/order_sum)
        elif(actions[i] == -1):
            asks.append(orderPrices[i]*(abs(Z_delta[i]))/order_sum)

    print(asks)
    print(bids)
    if(len(bids)!=0):
        ave_bids = sum(bids)/len(bids)
    else: ave_bids = 0
    ave_asks = sum(asks)/len(asks)
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
    n = 10  # number of agents
    t = 2 # number of rounds
    r = 0.0007700  # risk-free interest rate
    a = 1  # constant absolute risk aversion coefficient (CARA)
    beta = np.random.uniform(0, 10, n) # An array, risk preference when it comes to placing order
    price = 394 # initialize p(t=0) to be price
    var = 0.000173627  # variance of stock in risk premium
    alpha = np.random.uniform(0, 1, n)  # alpha [0,1] is the update propensity parameter.
    eps_BC = np.random.uniform(0, 0.1, n) # epsilon for BC model
    X = np.random.normal(394.730011, 10, n)  # X is X(t=0) which is the expected price for t=1 (next period)
    A = np.identity(n) # initialize A(t=0) as an identity matrix
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones


    actions = np.zeros(n)  # current actions for each agent (discrete values of 1, -1 and 0 - Buy Sell Hold)
    orderPrice = np.zeros(n)  # prices of current order for each agent
    Z_current = np.random.randint(100, 1000, n)  # each agent holds between 10 to 1000 shares
    print("initial Z_current:")
    print(Z_current)
    price_list_BC = []
    std_BC = [X.std()]


    # --- Simulation --- #

    for round in range(t):
        # Portfolio holding dynamics
        Z_delta = getDeltaZ(a, var, r, price, X)
        print("this is Z_delta:")
        print(Z_delta)
        Z_current = updateCurrentZ(Z_current,Z_delta,n)
        print("this is Z_current after change:")
        print(Z_current)
        # Price dynamics
        actions = getAction(Z_delta, Z_current, actions, n)
        print("actions are:", actions)

        orderPrice = getOrderPrice(actions, beta, price, X, orderPrice, n, r)
        price = updatePrice(Z_delta, actions, orderPrice, n)
        print("Order prices: ", orderPrice)
        # Expected price dynamics
        C = updateXwithBC(X, n, eps_BC)
        # C = updateXwithPA(X_prev, P, n, eps_PA)
        A = alpha*C + (1-alpha)*A
        X = np.matmul(A, X)

        print("price: ", price)
        price_list_BC.append(price)
        std_BC.append(X.std())


    # print(price_list_BC)
    # print(std_BC)
DoSimulation()

