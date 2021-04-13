import numpy as np
import random


def getDeltaZcontrol(a, var, r, p, X):
    '''
    given the current price (p(t)) and current opinion (x(t), note x(t) is expected price for
    period t+1), we we obtain z(t) which is the optimal # of shares held for current period t.
    '''
    # print("price is", p)
    # print("this is X")
    # print(X)
    # note that Z must be a discrete number
    deltaZ = (1 / (a * var)) * (X - (1 + r) * p)  # equation 2
    deltaZ = [int(i) for i in deltaZ]
    # Constraint on deltaZ: potentially place a upper limit on deltaZ (wealth constraint)

    return deltaZ


def updateCurrentZcontrol(Z_current, Z_delta, n):
    for i in range(n):
        if (Z_current[i] + Z_delta[i]) < 0:
            Z_current[i] = 0
        else:
            Z_current[i] += Z_delta[i]
    return Z_current


def getActioncontrol(Z_delta, Z_current, actions, n):
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
                Z_delta[i] = 0  # reset Z_delta
        else:
            actions[i] = 0  # Hold
    return actions, trading_volume


def getOrderPricecontrol(action, beta, p, X, orderPrice, n, r):
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
            orderPrice[i] = random.uniform(X[i], (1 + r) * p * (1 + beta[i]))
        else:
            orderPrice[i] = 0

    return orderPrice


def updatePricecontrol(Z_delta, actions, orderPrices, n):
    '''
    Order matching:
        - ignore the traditional order matching mechanism and fulfill everyone's orders
        - After orders are executed, we update price p(t) using the Bid-Ask prices.
    Returns the new price for next period
    '''
    bids = []  # an array of all bidding prices
    asks = []  # an array of all asking prices
    # filter out Z_delta where action[i] == 0
    # Z_delta = Z_delta
    order_sum = sum(map(abs, Z_delta))  # sum of the absolute value of orders
    for i in range(n):
        if (actions[i] == 1):
            bids.append(orderPrices[i] * (abs(Z_delta[i])) / order_sum)
        elif (actions[i] == -1):
            asks.append(orderPrices[i] * (abs(Z_delta[i])) / order_sum)

    # print("asking prices: ", asks)
    # print("bidding prices:", bids)
    return sum(bids) + sum(asks)


def updateXwithBCcontrol(X, n, eps):
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
            if abs((X[i] - X[j]) / X[i]) <= eps[i]:
                count += 1
                C[i][j] = 1  # mark C[i][j] with 1 to show that they are similar
        C[i] = np.where(C[i] == 1, 1 / count, C[i])  # formula after eq.6
        # print("agent", i)
        # print(C[i])
    return C


def DoSimulationControl(n, t, r, a, beta, price, var, alpha, eps_BC, X, A, Z_current, add_agents_sequence):
    # --- Initialization --- #
    # n = 100  # number of agents
    # t = 200  # number of rounds
    # r = 0.0008588  # risk-free interest rate
    # a = 1  # constant absolute risk aversion coefficient (CARA)
    # beta = np.random.uniform(0, 10, n) # An array, risk preference when it comes to placing order
    # beta = np.random.uniform(0, 0.1, n)
    # price = 19.95 # initialize p(t=0) to be price
    # var = 0.443208  # variance of stock in risk premium
    # var = 1.152757
    # alpha = np.random.uniform(0, 0.8, n)  # alpha [0,1] is the update propensity parameter.
    # eps_BC = np.random.uniform(0, 0.1, n)  # epsilon for BC model
    # X = np.random.normal(19.95, 3, n)  # X is X(t=0) which is the expected price for t=1 (next period)
    # A = np.identity(n) # initialize A(t=0) as an identity matrix
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones

    actions = np.zeros(n)  # current actions for each agent (discrete values of 1, -1 and 0 - Buy Sell Hold)
    orderPrice = np.zeros(n)  # prices of current order for each agent
    # Z_current = np.random.randint(100, 500, n)  # each agent holds between 10 to 1000 shares
    Z_current = Z_current.tolist()
    Z_delta = []
    # print("sum of Z_current:", sum(Z_current))
    # print("initial Z_current:")
    # print(Z_current)
    price_list_BC = [price]
    X_opinion = [X.mean()]
    std_BC = [X.std()]
    trading_volume_rational = [0]
    # trading_volume_irrational = [0]
    total_volume_rational = [sum(Z_current)]
    # total_volume_irrational = [0]

    # add_agents_sequence = [1, 0, 14, 9, 4, 7, 3, 5, 19, 17, 17, 9, 5, 5, 3, 7, 4, 6, 8]

    # --- Simulation --- #

    for rd in range(t):
        # print("---------------- ROUND ", rd, " ----------------")
        # adding irrational agents
        if (rd < len(add_agents_sequence)):
            # initialized the current # of stocks held to be 0
            for agent in range(add_agents_sequence[rd]):
                Z_current.append(0)
                Z_delta.append(0)
                X = np.concatenate((X, np.random.normal(price, 3, 1)))
                alpha = np.concatenate((alpha, np.random.uniform(0, 0.8, 1)))
                eps_BC = np.concatenate((eps_BC, np.random.uniform(0, 0.1, 1)))
                beta = np.concatenate((beta, np.random.uniform(0, 0.1, 1)))
                actions = np.concatenate((actions, np.zeros(1)))
                orderPrice = np.concatenate((orderPrice, np.zeros(1)))
                # print(X)
            # print("added ", add_agents_sequence[rd], "agents")

        n = len(X)

        # Portfolio holding dynamics
        Z_delta = getDeltaZcontrol(a, var, r, price, X)
        # print("Z_delta for ", len(Z_delta), " agents")
        # print(Z_delta)
        actions, trading_volume = getActioncontrol(Z_delta, Z_current, actions, n)
        trading_volume_rational.append(trading_volume)
        # print("actions size:", len(actions))
        # print(actions)
        Z_current = updateCurrentZcontrol(Z_current, Z_delta, n)
        total_volume_rational.append(sum(Z_current))
        # print("Z_current for ", len(Z_current), " agents")
        # print(Z_current)
        # Price dynamics
        orderPrice = getOrderPricecontrol(actions, beta, price, X, orderPrice, n, r)
        # print("orderprice size:", len(orderPrice))
        # print(orderPrice)
        price = updatePricecontrol(Z_delta, actions, orderPrice, n)

        # Expected price dynamics
        C = updateXwithBCcontrol(X, n, eps_BC)
        if rd < len(add_agents_sequence):
            A = np.pad(A, ((0, add_agents_sequence[rd]), (0, add_agents_sequence[rd])), mode='constant',
                       constant_values=0)
            for i in range(add_agents_sequence[rd]):
                A[-(i + 1)][-(i + 1)] = 1
            # print("A updated in shape")
            # print("A.shape:", A.shape)
            # np.savetxt("A{0}.csv".format(rd), A, delimiter=",")
            # for i in range(len(X)):
            #     print(sum(A[i]))
        # print("lenX:", len(X))
        # print("lenalpha:", len(alpha))
        for i in range(n):
            for j in range(n):
                # tmp[i][j] = alpha[i]*C[i][j]
                A[i][j] = alpha[i] * C[i][j] + (1 - alpha[i]) * A[i][j]
            # print(sum(A[i]))
        X = np.matmul(A, X)

        price_list_BC.append(price)
        X_opinion.append(X.mean())
        std_BC.append(X.std())
        # print("price for next period: ", price)

        # if (rd %10==0) :
        #     print("---------------- ROUND ", rd, " ----------------")
        #     print("this is Z_delta:")
        #     print(Z_delta)
        #     print("actions are:")
        #     print(actions)
        #     print("this is Z_current after change:")
        #     print(Z_current)
        #     print("Order prices: ", orderPrice)
        #     print("price for next period: ", price)

    # print("sum of Z_current:", sum(Z_current))
    print("price_list_BC + X_opinion")
    print(price_list_BC)
    print(X_opinion)
    print("std_BC")
    print(std_BC)
    print("trading_volume")
    print(trading_volume_rational)
    print("total_volume")
    print(total_volume_rational)

