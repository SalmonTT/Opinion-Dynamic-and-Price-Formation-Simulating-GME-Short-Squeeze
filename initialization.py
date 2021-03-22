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

def boundedConfidence(X, epsi):
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
        count = 1
        # for every agent, compare his/her opinion with others
        for j in range(100):
            if abs(X[i]-X[j]) <= epsi:
                count += 1
                C[i][j] = 100 # what does this mean?
        tmp = np.full(100, 1/count)
        C[i] = np.where(C[i] == 100, tmp, C[i])
    return C

def simulate(G, A, alpha, P, n):
    for i in range(n):
        C = boundedConfidence(G, 5)
        # print(C)
        A = alpha*C + (1-alpha)*A
        print(A)
        G = np.matmul(G, A)
        print("-------------------------------------------")
        # print(G)
        P = np.append(P, G.mean()-a*var*z_s)
    return P

mu, sigma = 30, 15 # mean and standard deviation for normal distribution
r = 0.5 # risk-free interest rate
a = 1 # constant absolute risk aversion coefficient (CARA)
var = 0.5 # variance of stock in risk premium (assumed constant, may need to read Hommes & Wagener)
z_s = 10 # supply per agent (Why not use equation 3 for this?)

# Draw random samples from normal distribution
# here X is X(t=0)
X = np.random.normal(mu, sigma, 100)
A = boundedConfidence(X, 5)

alpha = np.full(100, .5)
# Update the opinion matrix to get X(t=1)
X = np.matmul(X, A)

P = np.full(1, X.mean()-a*var*z_s)
P = simulate(X, A, alpha, P, 10)
print(P)