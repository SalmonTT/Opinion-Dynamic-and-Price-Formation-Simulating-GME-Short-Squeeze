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

def initialize(mu, sigma):
    G = np.random.normal(mu, sigma, 100)
    print(G)
    print(G.mean())
    print("Start")
    return G

def boundedConfidence(G):
    C = np.zeros((100, 100))
    for i in range(100):
        count = 0
        for j in range(100):
            if abs(G[i]-G[j]) <= 5 :
                count = count+1
                C[i][j] = 100
        tmp = np.full(100, 1/count)
        C[i] = np.where(C[i] == 100, tmp, C[i])
    return C

def simulate(G, A, alpha, P, n):
    for i in range(n):
        C = boundedConfidence(G)
        # print(C)
        A = alpha*C + (1-alpha)*A
        print(A)
        G = np.matmul(G, A)
        print("-------------------------------------------")
        # print(G)
        P = np.append(P, G.mean()-a*var*z_s)
    return P

mu, sigma = 30, 15
r = 0.5 # interest rate
a = 1 # constant absolute risk aversion coefficient
var = 0.5 # variance of stock in risk premium
z_s = 10 # supply per agent
G = initialize(mu, sigma)
A = boundedConfidence(G)
# print(A)
alpha = np.full(100, .5)
G = np.matmul(G, A)
P = np.full(1, G.mean()-a*var*z_s)
P = simulate(G, A, alpha, P, 10)
print(P)