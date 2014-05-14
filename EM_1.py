import numpy as np
import random as rd
import math

def initialize(K, data):
    # print data, not defined

    # initialize the mu_k randomly
    cols = (data.shape)[1]
    mu_k = np.zeros((K,cols))
    for row in range(K):
        idx = int(np.floor(rd.random()*len(data)))
        for col in range(cols):
            mu_k[row][col] += data[idx][col]

    # print mu_k

    # initialize the sigma_k
    sigma_k = []
    for k in range(K):
        sigma_k.append(np.cov(data.T))
        # sigma_k.append(np.mat(np.random.random((cols,cols))))

    # initialize the pi_k randomly
    sum_pi = 1.0
    pi_k = np.zeros(K)
    pi_k += sum_pi/K

    # print np.array(sigma_k)
    # evaluate the initial value of the log likelihood
    # evaluate(mu_k,sigma_k,pi_k)
    print pi_k
    return mu_k, sigma_k, pi_k

def e_step(K,mu,sigma,pi,data):
    """
    Evaluate the responsibility using the current parameter
    values
    """
    r_nk = np.zeros((len(data),K))

    for i in range(len(data)):
        for k in range(K):
            r_nk[i][k] = (pi[k]*gdf(data[i],mu[k],sigma[k]))/prior_prob(K,mu,sigma,pi,data[i])

    return r_nk

def m_step(r_nk, K, data):
    """
    # Re-estimate the parameters using the current
    # responsibility
    # """
    # calculate new_mu_k
    N_k = np.zeros(K)
    # print N_k
    cols = (data.shape)[1]
    new_mu_k = np.zeros((K,cols))
    for k in range(K):
        for n in range(len(data)):
            N_k[k] += r_nk[n][k]
            new_mu_k[k] += (r_nk[n][k]*data[n])

        # print new_mu_k
        new_mu_k[k] /= N_k[k]
    # print new_mu_k

    # calculate new_sigma_k
    new_sigma_k = np.zeros((K,cols,cols))
    # print new_sigma_k
    for k in range(K):  
        for n in range(len(data)):
            xn = np.zeros((1,4))
            mun = np.zeros((1,4))
            xn += data[n]
            mun += new_mu_k[k]
            x_mu = xn - mun
            # print xn,xn.shape
            # print x_mu*x_mu.T
            new_sigma_k[k] += (r_nk[n][k]*x_mu*x_mu.T)
        new_sigma_k[k] /= N_k[k]
    # print new_sigma_k

    # calculate new_pi_k
    new_pi_k = np.zeros(3)
    for k in range(K):
        new_pi_k[k] += (N_k[k]/len(data))

    # print new_pi_k
    return new_mu_k, new_sigma_k, new_pi_k

def likelihood(K,mu,sigma,pi,data):
    """
    Calculate the log likelihood using current mu, sigma, and
    pi. 
    """
    log_score = 0.0

    for n in range(len(data)):
        log_score += np.log(prior_prob(K,mu,sigma,pi,data[n]))
    # print log_score
    # epsilon = 0.0001
    return log_score

def prior_prob(K,mu,sigma,pi,data):
    pb = 0.0
    for k in range(K):
        # prior_prob += 1
        pb += pi[k]*gdf(data,mu[k],sigma[k])

    return pb

def gdf(x,mu,sigma):
    score = 0.0

    x_mu = np.matrix(x - mu)
    inv_sigma = np.linalg.inv(sigma)
    det_sqrt = np.linalg.det(sigma)**0.5

    norm_const = 1.0/((2*np.pi)**(len(x)/2)*det_sqrt)
    exp_value = math.pow(math.e,-0.5 * (x_mu * inv_sigma * x_mu.T))
    score = norm_const * exp_value

    # print score
    return score

def gmm(K,data,rst):
    mu, sigma, pi = initialize(K,data)
    # epsilon = 1e-30
    log_score = likelihood(K, mu, sigma, pi, data)
    log_score_0 = log_score
    threshold = 0.001
    i = 0
    max_iter = 100
    while i < max_iter:
        # expectation step
        r_nk = e_step(K,mu,sigma,pi,data)

        # maximization step
        mu, sigma, pi = m_step(r_nk, K, data)

        # evaluate the log likelihood
        new_log_score = likelihood(K, mu, sigma, pi, data)
        # print log_score, new_log_score
        if abs(new_log_score - log_score) < threshold:
            # print abs(new_log_score - log_score)
            break

        # print the process
        print rst+1, "  |  ", i+1 ,"  |  ", log_score, " | ", new_log_score 
        log_score = new_log_score

        i += 1

    print "convergenced"

def read_data(fn):
    with open(fn) as f:
        return np.loadtxt(f, delimiter= ',', dtype="float", 
            skiprows=0, usecols=(0,1,2,3))

def main():
    data = read_data("bezdekIris.data.txt")
    print "#restart | EM iteration | log likelihood | expected log likelihood"
    print "------------------------------------------------------------------"
    for i in range(5):
        gmm(3,data,i)

if __name__ == '__main__':
    main()