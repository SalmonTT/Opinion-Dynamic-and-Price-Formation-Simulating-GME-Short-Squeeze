from combined_simulation_1 import DoSimulation1
from combined_simulation_2 import DoSimulation2
from control_simulation import DoSimulationControl
import numpy as np
import random

def initialization():
    n = 100  # number of agents
    t = 30  # number of rounds
    r = 0.0008588  # risk-free interest rate
    a = 1  # constant absolute risk aversion coefficient (CARA)
    beta = np.random.uniform(0, 0.1, n)  # An array, risk preference when it comes to placing order
    price = 19.95  # initialize p(t=0) to be price
    var = 1.152757  # variance of stock in risk premium
    alpha = np.random.uniform(0, 0.8, n)  # alpha [0,1] is the update propensity parameter.
    eps_BC = np.random.uniform(0, 0.1, n)  # epsilon for BC model
    X = np.random.normal(19.95, 3, n)  # X is X(t=0) which is the expected price for t=1 (next period)
    A = np.identity(n)  # initialize A(t=0) as an identity matrix
    actions = np.zeros(n)  # current actions for each agent (discrete values of 1, -1 and 0 - Buy Sell Hold)
    order_price_rational = np.zeros(n)  # prices of current order for each agent
    Z_current_rational = np.random.randint(100, 500, n)  # each agent holds between 10 to 1000 shares

    print("initial Z_current for rational agents:")
    print(Z_current_rational)
    # price_list_Rational = []
    # X_std_Rational = [X.std()]

    '''Irrational Network Initialization'''
    max_current_Z = 500
    # Z_current_irrational = []  # length of this list indicates the number of total irrational agents in the network
    # Z_delta_irrational = []
    max_Z_delta = 30
    orderPconstant = 0.3
    GME_volume_array = [14927612
        , 7060665
        , 144501736
        , 93717410
        , 46866358
        , 74721924
        , 33471789
        , 57079754
        , 197157946
        , 177874000
        , 178587974
        , 93396666
        , 58815805
        , 50566055
        , 37382152
        , 78183071
        , 42698511
        , 62427275
        , 81345013]
    # number of agents added at each round proportional to the volume traded that day (Jan 11 - Feb 5)
    add_agents_sequence = [x / 10000000 for x in GME_volume_array]
    add_agents_sequence = [int(x) for x in add_agents_sequence]
    print("add_agents_sequence: ", add_agents_sequence)

    print("------------simulation1---------------")
    DoSimulation1(n, t, r, a, beta, price, var, alpha, eps_BC, X, A, actions, order_price_rational, Z_current_rational, max_Z_delta, orderPconstant, add_agents_sequence)
    print("------------simulation2---------------")
    DoSimulation2(n, t, r, a, beta, price, var, alpha, eps_BC, X, A, actions, order_price_rational, Z_current_rational, max_current_Z, max_Z_delta, orderPconstant, add_agents_sequence)
    print("------------simulation control---------------")
    DoSimulationControl(n, t, r, a, beta, price, var, alpha, eps_BC, X, A, Z_current_rational, add_agents_sequence)

initialization()
