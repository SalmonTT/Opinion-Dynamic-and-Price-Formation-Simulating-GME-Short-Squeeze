import numpy as np
import decimal
import random

def getCurrentZ(a, var, r, p, X):
    '''
    given the current price (p(t)) and current opinion (x(t), note x(t) is expected price for
    period t+1), we we obtain z(t) which is the optimal # of shares held for current period t.
    '''
    # note that Z must be a discrete number
    Z = int((1/(a*var)) * (X - (1+r)*p)) # equation 2
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

def getOrderPrice(action, beta, p, X, orderPrice, n):
    '''
    return the prices of orders for each agent at current period t
    '''
    for i in range(n):
        if(action[i] == 1): # buy order
            # returns a float with two decimal places that is greater than or equal to p(t)-beta and less than x(t)
            orderPrice[i] = decimal.Decimal(random.randrange((p - beta[i]) * 100, X[i] * 100)) / 100
        elif(action[i] == -1): # sell order
            # returns a float with two decimal places that is greater than or equal to x(t) and less than p(t)+beta
            orderPrice[i] = decimal.Decimal(random.randrange(X[i] * 100, (p + beta[i]) * 100)) / 100
        else: # hold
            return orderPrice
    return orderPrice

def updatePrice(actions, orderPrices, p, Zt, Z_prev, Z_now):
    '''
    Order matching:
        - ignore the traditional order matching mechanism and fulfill everyone's orders given Zt constraint
    Zt constraint:
        - if sum of 'orders' is zero (meaning no change in total number of shares held by all agents), execute
        all orders.
        - randomly select an agent's order and tries to execute it:
            - Execute the very first order
            - For all other orders：
                - if
    '''
    orders = Z_now - Z_prev


    return



def DoSimulation():

    # --- Initialization --- #
    n = 100 # number of agents
    t = 10 # number of rounds
    r = 0.02  # risk-free interest rate
    a = 0.1  # constant absolute risk aversion coefficient (CARA)

    beta = [] # An array, risk preference when it comes to placing orders
    actions = np.zeros(n) # current actions for each agent (discrete values of 1, -1 and 0 - Buy Sell Hold)
    orderPrice = np.zeros(n) # prices of current order for each agent
    Z = [] # need to find a way to initialize Z


    var = 0.1  # variance of stock in risk premium
    alpha = np.random.uniform(0, 1, n)  # alpha [0,1] is the update propensity parameter.
    eps_BC = np.random.uniform(0, 0.4, n) # epsilon for BC model
    eps_PA = np.full(n, 0.1)
    X = np.random.uniform(1, 10, n) # X is X(t=0) which is the expected price for t=1 (next period)
    A = np.identity(n) # initialize A(t=0) as an identity matrix
    # A = np.ones(n)  # initialize A(t=0) as an matrix full of ones

    
