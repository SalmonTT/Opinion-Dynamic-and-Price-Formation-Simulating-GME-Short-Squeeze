import numpy as np
import matplotlib.pyplot as plt

# test different setting of mean and std
def testLognormalDistribution():
    mu, sigma = 3., .5
    s = np.random.lognormal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
    x = np.linspace(min(bins), max(bins), 10000)
    pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
           / (x * sigma * np.sqrt(2 * np.pi)))
    plt.plot(x, pdf, linewidth=2, color='r')
    plt.axis('tight')
    plt.show()

def PriceAdaptive(X, epsi):
    """
    :param X: the opinion vector X(t)
    :param epsi: threshold
    :return: the adaptive update matrix C(t)
    """
    # C is the adaptive update matrix
    C = np.zeros((100, 100))
    # iterate through every agent
    for i in range(100):
        # count is the number of agents that are similar to agent i
        # since {i} is included in I(i, x(t-1), p(t-1)), count is at least 1
        count = 0
        # for every agent, compare his/her opinion with others
        for j in range(100):
            if abs(X[i]-X[j]) <= epsi:
                count += 1
                C[i][j] = 100 # what does this mean?
        tmp = np.full(100, 1/count)
        C[i] = np.where(C[i] == 100, tmp, C[i])
    return C

def simulate(X, A, alpha, P, n):
    '''
    :param X: X(t=1)
    :param A: A(t=0)
    :param alpha:
    :param P: P(t=0)
    :param n: number of rounds
    :return: return P(t=n)
    '''
    for i in range(n):
        C = PriceAdaptive(X, 0.01)
        # print("before A: ")
        # print(A)
        A = alpha*C + (1-alpha)*A
        # print("After A: ")
        # print(np.all(A))
        # print(np.all(C))
        X = np.matmul(A,X)
        # print("-------------------------------------------")
        # print(X)
        P = np.append(P, X.mean()-a*var*z_s)
        # print("This is round ", i, "X is: ")
        # print(X)
    return P


""" Initialization """
r = 0.02 # risk-free interest rate
a = 1 # constant absolute risk aversion coefficient (CARA)
var = 0.5 # variance of stock in risk premium (assumed constant, may need to read Hommes & Wagener)


# Draw random samples from normal distribution
X0 = np.random.uniform(0, 1, 100)
A0 = PriceAdaptive(X0, 0.01)
# print("initialized A: ")
# print(np.all(A0))
# alpha is an ndarray filled with 0.5
alpha = np.full(100, .25)
# Update the opinion matrix to get X(t=1)
X1 = np.matmul(A0, X0)
# initialize P(t=0)
P0 = 0.5
# P0 = np.full(1, X1.mean()-a*var*z_s)
# initialize Z(t=0) Question: do we discretize this?
Z0 = (1/(a*var))*(X1-(1+r)*P0)
print(Z0)
z_s = Z0.mean()
print(z_s)
"""
after initialization we now have the following:
A(t=0)
X(t=0), X(t=1)
P(t=0) 
"""

""" Simulation """
P = simulate(X1, A0, alpha, P0, 10)
print(P)