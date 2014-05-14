'''
@author: dsc
'''
import numpy as np
import random as rd
import math

def Estep(mu,sigma,pi,data):# the e step of EM, calculate the R_n_k
    rnk = []
    for d in data:
        rnk_temp = []
        for i in range(3):
            rnk_temp += [(pi[i]*normalpdf(d,mu[i],sigma[i]))/recordvalue(mu,sigma,pi,d)]#caculate the new rnk of the E step
        rnk += [rnk_temp]
    return rnk

def Mstep(rnk, data):# The M step of the EM and update the mu the sigma and the Nk
    
    N_k = np.zeros(3)#get the new N_k
    mu_k = np.zeros((3,4))#update the new mu
    for k in range(3):
        for n in range(len(data)):
            N_k[k] += rnk[n][k]
            mu_k[k] += (rnk[n][k]*data[n])
        mu_k[k] = mu_k[k] / N_k[k]
   
    sigma_k = np.zeros((3,4,4))#update the new sigma
    for k in range(3):  
        for i in range(len(data)):
            mu_temp = np.zeros((1,4))
            mu_temp += np.array(data[i]) - np.array(mu_k[k])
            sigma_k[k] += (rnk[i][k]*mu_temp*mu_temp.T)
        sigma_k[k] /= N_k[k]
            
    pi_k = np.zeros(3)#update the new PI
    for k in range(3):
        pi_k[k] += (N_k[k]/len(data))
    return mu_k, sigma_k, pi_k

def read_data():#loading the data from iris data
    f =open("bezdekIris.data.txt", 'r')
    lines=[line.strip() for line in f.readlines()]  
    f.close()
    lines=[line.split(",") for line in lines if line]
    data = np.array([line[:4] for line in lines ], dtype=np.float)#load the first 4 columns
    return data

def init(data):
    #lenth = (data.shape)[1]  
    mu = []#initialize the mu
    sigma = []# initialize the sigma 
    for i in range(3):#choose three as the initial mu
        index = rd.randint(0, len(data)-1)
        mu += [data[index]]
    sigma += [np.cov(data.T) for i in range(3)]#add 3 covar matix into sigma
    # initialize the pi_k randomly
    PI = np.ones(3)*1/3# initial the pi, since we have 3
    initlh = loglikelihood(mu,sigma,PI,data) 
    return initlh,mu, sigma, PI

def normalpdf(x,mu,sigma):#caculate the noramal distribution pdf value
    pdf = (1.0/((2*np.pi)**(len(x)/2)*(np.linalg.det(sigma)**0.5)))*math.pow(math.e ,-0.5 * (np.matrix(x - mu) * np.linalg.inv(sigma) * np.matrix(x - mu).T))
    #the formula is from http://en.wikipedia.org/wiki/Normal_distribution
    return pdf

def recordvalue(mu,sigma,pi,record):#caculate every score for each record
    re_value = 0.00#each record means each row in the data 
    for count in range(3):
        re_value += pi[count]*normalpdf(record,mu[count],sigma[count])
    return re_value

def loglikelihood(mu,sigma,pi,data):
    likelihood = 0.00
    for d in data:#read every record from the dataset 
        likelihood += np.log(recordvalue(mu,sigma,pi,d))#return the sum of the total likeli value
    return likelihood

if __name__ == '__main__':
    data = read_data()
    for i in range(5):
        print '\n' + 'This is the '+str(i+1)+'th iteration'
        likelihood , mu, sigma, PI = init(data)#get the first loglikelihood
        for it in range(1000000):
            rnk = Estep(mu,sigma,PI,data)#doing the E step
            mu, sigma, pi = Mstep(rnk,data)#proccess the M step

            likelihood_new = loglikelihood(mu, sigma, pi, data)#update the likelihood value
            if abs(likelihood_new - likelihood) < 0.01:#if the test is over
                break
            print i+1, it+1 , likelihood, likelihood_new#print the result  
            likelihood = likelihood_new#update the likelihood
    