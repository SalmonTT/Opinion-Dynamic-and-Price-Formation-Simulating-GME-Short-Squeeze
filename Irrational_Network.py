import numpy as np
import random

def getDeltaZSimple(delta_Z, max_delta, n):
    '''
    simply return delta_Z which is a list of random number in range [0, max_delta]
    '''
    for i in range(n):
        delta_Z[i] = random.uniform(0, max_delta)

    return delta_Z

def getDeltaZvoter():
    '''
    binary opinion: buy (1), hold (0)
    if buy, delta_Z is a constant
    '''
    delta_Z = []
    return delta_Z

def getOrderPriceSimple(orderPrice, n, r, p, orderPconstant):
    '''
    orderPconstant is a list of constants where each constant > 0 (to be initialized in the beginning)
    '''
    for i in range(n):
        orderPrice[i] = ((1+r)*p) + orderPconstant[i]
    return orderPrice

def getOrderPriceComplex(orderPrice, n, r, X, p):
    for i in range(n):
        orderPrice[i] = random.uniform((1 + r) * p , X[i])
    return orderPrice

