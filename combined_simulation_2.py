import numpy as np
import random

'''
Difference to version 1:
- Add max Z_current for irrational agents (as we assume these irrational agents are retail investors with 
shallow pockets

Things to be explored:
- Different levels of max_current
- How quickly is max_current reached for all irrational agents
- Price dynamics after setting max_current
'''

def getDeltaZ(a, var, r, p, X):
    '''
    given the current price (p(t)) and current opinion (x(t), note x(t) is expected price for
    period t+1), we we obtain z(t) which is the optimal # of
    shares held for current period t.
    '''
    # print("price is", p)
    # print("this is X")
    # print(X)
    # note that Z must be a discrete number
    deltaZ = (1 / (a * var)) * (X - (1 + r) * p)  # equation 2
    deltaZ = deltaZ.astype(int)
    # Constraint on deltaZ: potentially place a upper limit on deltaZ (wealth constraint)

    return deltaZ

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
    trading_volume = 0
    for i in range(n):
        if (Z_delta[i] > 0):
            actions[i] = 1  # Buy
            trading_volume = trading_volume + Z_delta[i]
        elif (Z_delta[i] < 0):
            if (Z_current[i] > 0):  # only if agent currently owns share
                if (abs(Z_delta[i]) > Z_current[i]):  # if agent owns less that Z_delta
                    Z_delta[i] = -Z_current[i]
                actions[i] = -1  # Sell
                trading_volume = trading_volume + abs(Z_delta[i])
            else:
                actions[i] = 0  # agent does not own share, do nothing
                Z_delta[i] = 0
        else:
            actions[i] = 0  # Hold

    return actions, trading_volume

def getOrderPrice(action, beta, p, X, orderPrice, n, r):
    '''
    return the prices of orders for each agent at current period t
    '''
    for i in range(n):
        if (action[i] == 1):  # buy order
            # returns a float with two decimal places that is greater than or equal to (1+r)*p(t)-beta and less than x(t)
            orderPrice[i] = random.uniform(((1 + r) * p * (1 - beta[i])), X[i])
        elif (action[i] == -1):  # sell order
            # returns a float with two decimal places that is greater than or equal to x(t) and less than (1+r)*p(t)+beta
            # orderPrice[i] = decimal.Decimal(random.randrange(int(X[i] * 100), int(((1+r)*p + beta[i]) * 100))) / 100
            orderPrice[i] = random.uniform(X[i], (1 + r) * p *(1 + beta[i]))
        elif (action[i] == 0):
            orderPrice[i] = 0

    return orderPrice

def updatePrice(Z_delta_rational, Z_delta_irrational, actions_rational, orderPricesRational, orderPricesIrrational, rd):
    '''
    Order matching:
        - ignore the traditional order matching mechanism and fulfill everyone's orders
        - After orders are executed, we update price p(t) using the Bid-Ask prices.
    Returns the new price for next period
    '''
    actions_irrational = [1]*len(orderPricesIrrational) # list of 1s same length as orderPricesIrrational
    Z_delta_rational = Z_delta_rational.tolist()
    Z_delta = Z_delta_rational + Z_delta_irrational
    actions_rational = actions_rational.tolist()
    actions = actions_rational + actions_irrational

    orderPrices = orderPricesRational.tolist()
    orderPrices.extend(orderPricesIrrational)
    # if rd > 25:
    #     print("order prices in ", rd)
    #     print(orderPrices)
    bids = []  # an array of all bidding prices
    asks = []  # an array of all asking prices
    order_sum = sum(map(abs, Z_delta))  # sum of the absolute value of orders
    for i in range(len(actions)):
        if (actions[i] == 1):
            bids.append(orderPrices[i] * (abs(Z_delta[i])) / order_sum)
        elif (actions[i] == -1):
            asks.append(orderPrices[i] * (abs(Z_delta[i])) / order_sum)

    # print("asking prices: ", asks)
    # print("bidding prices:", bids)
    return sum(bids) + sum(asks)

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
            if abs(X[i] - X[j]) / abs(X[i]) <= eps[i]:
                count += 1
                C[i][j] = 1  # mark C[i][j] with 1 to show that they are similar
        C[i] = np.where(C[i] == 1, 1 / count, C[i])  # formula after eq.6

    return C


##### Irrational agents dynamics functions #####

def getDeltaZSimple(delta_Z, Z_current, max_current, max_delta):
    '''
    simply return delta_Z which is a list of random number in range [0, max_delta]
    '''
    for i in range(len(delta_Z)):
        delta_Z[i] = int(random.uniform(0, max_delta))
        if((delta_Z[i]+Z_current[i]) > max_current):
            delta_Z[i] = max_current - Z_current[i]
    return delta_Z

def getOrderPriceSimple(Z_delta, r, p, orderPconstant):
    '''
    orderPconstant is a list of constants where each constant > 0 (to be initialized in the beginning)
    '''
    orderPrice = np.zeros(len(Z_delta))
    for i in range(len(Z_delta)):
        orderPrice[i] = random.uniform((1+r)*p, (1+r)*p*(1+orderPconstant))
    return orderPrice

def DoSimulation2(n, t, r, a, beta, price, var, alpha, eps_BC, X, A, actions, order_price_rational, Z_current_rational, max_current_Z, max_Z_delta, orderPconstant, add_agents_sequence):
    '''Rational Network Initialization'''
    # n = 100  # number of agents
    # t = 50  # number of rounds
    # r = 0.0008588  # risk-free interest rate
    # a = 1  # constant absolute risk aversion coefficient (CARA)
    # beta = np.random.uniform(0, 0.1, n)  # An array, risk preference when it comes to placing order
    # price = 19.95  # initialize p(t=0) to be price
    # var = 1.152757  # variance of stock in risk premium
    # alpha = np.random.uniform(0, 0.8, n)  # alpha [0,1] is the update propensity parameter.
    # eps_BC = np.random.uniform(0, 0.1, n)  # epsilon for BC model
    # X = np.random.normal(19.95, 3, n)  # X is X(t=0) which is the expected price for t=1 (next period)
    # A = np.identity(n)  # initialize A(t=0) as an identity matrix
    # actions = np.zeros(n)  # current actions for each agent (discrete values of 1, -1 and 0 - Buy Sell Hold)
    # order_price_rational = np.zeros(n)  # prices of current order for each agent
    # Z_current_rational = np.random.randint(100, 500, n)

    # print("initial Z_current for rational agents:")
    # print(Z_current_rational)
    price_list_Rational = [price]
    X_opinion = [X.mean()]
    X_std_Rational = [X.std()]
    trading_volume_rational = [0]
    trading_volume_irrational = [0]
    total_volume_rational = [sum(Z_current_rational)]
    total_volume_irrational = [0]

    '''Irrational Network Initialization'''
    # max_current_Z = 500
    Z_current_irrational = [] # length of this list indicates the number of total irrational agents in the network
    Z_delta_irrational = []
    # max_Z_delta = 100
    # orderPconstant = 0.1
    # GME_volume_array = [14927612
    #                 ,7060665
    #                 ,144501736
    #                 ,93717410
    #                 ,46866358
    #                 ,74721924
    #                 ,33471789
    #                 ,57079754
    #                 ,197157946
    #                 ,177874000
    #                 ,178587974
    #                 ,93396666
    #                 ,58815805
    #                 ,50566055
    #                 ,37382152
    #                 ,78183071
    #                 ,42698511
    #                 ,62427275
    #                 ,81345013]
    # # number of agents added at each round proportional to the volume traded that day (Jan 11 - Feb 5)
    # add_agents_sequence = [x/10000000 for x in GME_volume_array]
    # add_agents_sequence = [int(x) for x in add_agents_sequence]
    # print(add_agents_sequence)

    # --- Simulation --- #

    for round in range(t):
        # print("---------------- ROUND ", round, " ----------------")
        # adding irrational agents
        if(round < len(add_agents_sequence)):
            # initialized the current # of stocks held to be 0
            for agent in range(add_agents_sequence[round]):
                Z_current_irrational.append(0)
                Z_delta_irrational.append(0)
            # print("added ", add_agents_sequence[round], "agents")


        # Irrational agents Z_delta dynamics
        Z_delta_irrational = getDeltaZSimple(Z_delta_irrational, Z_current_irrational, max_current_Z, max_Z_delta)
        trading_volume_irrational.append(sum(Z_delta_irrational))
        # print("this is the Z_delta for irrational agents")
        # print(Z_delta_irrational)
        zipped_lists = zip(Z_current_irrational, Z_delta_irrational)
        Z_current_irrational = [x + y for (x, y) in zipped_lists]
        total_volume_irrational.append(sum(Z_current_irrational))
        # print("this is the Z_current of irrational agents: ")
        # print(Z_current_irrational)
        # print("this is the Z_current of rational agents: ")
        # print(Z_current_rational)

        # Rational agents Z_delta dynamics
        Z_delta_rational = getDeltaZ(a, var, r, price, X)
        # print("this is Z_delta for rational agents")
        # print(Z_delta_rational)
        actions, trading_volume = getAction(Z_delta_rational, Z_current_rational, actions, n)
        trading_volume_rational.append(trading_volume)
        Z_current_rational = updateCurrentZ(Z_current_rational, Z_delta_rational, n)
        total_volume_rational.append(sum(Z_current_rational))


        # Price dynamics (takes into consideration both rational and irrational agents orders
        order_price_rational = getOrderPrice(actions, beta, price, X, order_price_rational, n, r)
        # print("this is orderprice rational:")
        # print(order_price_rational)
        order_price_irrational = getOrderPriceSimple(Z_delta_irrational, r, price, orderPconstant)
        # print("this is orderprice irrational:")
        # print(order_price_irrational)
        price = updatePrice(Z_delta_rational, Z_delta_irrational, actions, order_price_rational, order_price_irrational, round)


        # Rational agents Expected price dynamics
        C = updateXwithBC(X, n, eps_BC)
        # C = updateXwithPA(X_prev, P, n, eps_PA)
        for i in range(n):
            for j in range(n):
                # tmp[i][j] = alpha[i]*C[i][j]
                A[i][j] = alpha[i] * C[i][j] + (1 - alpha[i]) * A[i][j]
        X = np.matmul(A, X)

        price_list_Rational.append(price)
        X_opinion.append(X.mean())
        X_std_Rational.append(X.std())

        # print("this is Z_delta:")
        # print(Z_delta)
        # print("actions are:")
        # print(actions)
        # print("this is Z_current after change:")
        # print(Z_current)

        # print("price for next period: ", price)
    print("price_list_Rational + X_opinion")
    print(price_list_Rational)
    print(X_opinion)
    print("X_std_Rational")
    print(X_std_Rational)
    print("trading_volume_rational + trading_volume_irrational")
    print(trading_volume_rational)
    print(trading_volume_irrational)
    print("total_volume_rational + total_volume_irrational")
    print(total_volume_rational)
    print(total_volume_irrational)

