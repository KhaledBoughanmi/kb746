# -*- coding: utf-8 -*-
"""
@author: Khaled Boughanmi
Fully NON PARAMETRIC DYNAMICS/SUPERVISED HIERARCHICAL DIRICHLET PROCESS
	*********NP-sHDP-SSS******** 
With automatic hyperpriors updates
### CHINESE RESTAURANT PROCESS / DIRECT ASSIGNMENT ALGORITHM ###
### STICK BREAKING PROCESS UPDATES AT EACH NEW THEMATIC DISCOVERY ###
"""

"""
Needed Building packages
"""
import os
import numpy as np
import numba as nb
# Make sure to update the Antoniak generator with the 
# Number of largest topic (max tag count in bag of tags)
import Antoniak_rvs 
import Utils 
import scipy.linalg.matfuncs as matfuncs
import pylab
import time as thetime
import datetime
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = (10, 6)
# Configure print settings
np.core.arrayprint._line_width  = 200
np.set_printoptions(suppress=True)
np.random.seed(1)

"""

        State Space Non Paramteric Regression Part
        
"""
# The augmented basis functions for the main polynomial effects
# The outcome of this function is the foloowing big matrix of covariates:
# [1, XL, XS, XS^2, ...., X^p]
@nb.jit(nopython = True)
def add_intercept(xL):
    res = np.zeros((xL.shape[0], xL.shape[1]+1))
    for i in range(xL.shape[0]):
        res[i,0] = 1.
        for j in range(xL.shape[1]):
            res[i, j+1] = xL[i,j]
    return res
            
@nb.jit(nopython = True)   
def append_matrix(A,B):
    res = np.zeros((A.shape[0],A.shape[1]+B.shape[1]))
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res[i,j] = A[i,j]
        for j in range(B.shape[1]):
            res[i, A.shape[1] + j] = B[i,j]
    return(res)
        
@nb.jit(nopython = True)   
def append_matrix_vec(A,x):
    res = np.zeros((A.shape[0],A.shape[1]+int(1)))
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res[i,j] = A[i,j]
        res[i, A.shape[1]] = x[i]
    return(res)

@nb.jit(nopython = True)
def get_X(xL, xS, p):
    #n = xL.shape[0]
    tmp = xL
    # Adding the intercept:: can be deleted later
    #tmp = add_intercept(xL) #.append(np.ones(n)[:,None], tmp, axis = 1)
    
    # Build the polynomial basis for the Splines covariates
    # xs + xs^2 + ... + xs^p for each s=1..J
    for j in range(xS.shape[1]):
        for i in range(int(1), int(p + 1)):
            xsp = np.power(xS[:,int(j)], i)
            tmp = append_matrix_vec(tmp, xsp)
    return(tmp)

          
# The augmented basis functions with the knots
# The number of knots is assumed the same for all the splines
# The outcome of this function is the following big matrix:
# [(xs1-k1)+^p, (xs1-k1)+^p, ..., (xs1-kK)+^p,(xs2-k1)+^p, (xs2-k1)+^p, ..., (xs2-kK)+^p, ... ]
@nb.jit(nopython = True) 
def get_Z(xS, K, p):

    # Find the K knots between the min and the max of each xs
    kappas = np.linspace(np.min(xS[:,0]), np.max(xS[:,0]), int(K))
    Zj = np.zeros((len(xS[:,0]),int(K)))
    
    for i, xi in enumerate(xS[:,0]):
        for k, kappa in enumerate(kappas):
            mult = 1
            if xi < 0:
                mult = -1
            Zj[i][k] =  mult * (xi >= kappa) * np.power((xi - kappa), p)
    Z = Zj
    
    for j in range(1, xS.shape[1]):
        # Find the K knots between the min and the max of each xs
        kappas = np.linspace(np.min(xS[:,j]), np.max(xS[:,j]), int(K))
        Zj = np.zeros((len(xS[:,j]),int(K)))
        
        for i, xi in enumerate(xS[:,j]):
            for k, kappa in enumerate(kappas):
                mult = 1
                if xi < 0:
                    mult = -1
                Zj[i][k] =  mult * (xi >= kappa) * np.power((xi - kappa), p)

        Z = append_matrix(Z, Zj) 

    return(Z)           

 
# The matrix of random effects relative to the penalized splines
# D = [0, 0, ..., 0 , sigma1 * 1_K1, sigma2 * 1_K2]

@nb.jit(nopython = True)
def append(x, y):
    # Make sure the input is an array
    res = np.zeros(x.shape[0]+y.shape[0])
    i = 0
    for item in x:
        res[i] = item
        i += 1
    for item in y:
        res[i] = item
        i += 1
    return(res)
    
@nb.jit(nopython = True) 
def get_D(num_zeros, num_splines, sigma_e_iter, sigma_u_iter, K):
    #JL: the number of linear effects
    #JpS: The numebr of non linear effects given their basis p*S 
    D = np.ones(int(num_zeros)) / 100.
    sigmas = 1 / sigma_u_iter
    for j in range(int(num_splines)):
        Dj = sigmas[j] * np.ones(int(K))
        D  = append(D, Dj)    
    return(np.diag(D))

"""
 The full conditionals
"""


"""
                            ###############################
                                State space part
                            ###############################
"""
# keep certain rows relative to indices in idx
@nb.jit(nopython=True)
def sub_row_matrix(X, idx):
    res = np.zeros((len(idx), X.shape[1]))
    for k in range(len(idx)):
        res[k] = X[int(idx[k])]
    return(res)

@nb.jit(nopython=True)
def sub_array(x, idx):
    n = idx.shape[0]
    res = np.zeros(n)
    for k in range(n):
        res[k] = x[int(idx[k])]
    return(res)
    
@nb.autojit(nopython=True)
def which(a, x):
    # Find number of elements in the array
    m = 0
    for k in range(len(a)):
        if x == a[k]:
            m += 1
    if m == 0:
        res = np.array([-1.])
    else:
        res = np.zeros(int(m))
        t = 0
        for k in range(len(a)):
            if x == a[k]:
                res[t] = k
                t += 1
    return(res)

# Full conditionals
@nb.jit(nopython=True)
def RanMNormalPrec(mub, inv_Vb):
    c      = np.linalg.cholesky(inv_Vb)
    rv     = np.random.normal(loc= 0, scale= 1, size= len(mub))
    draw   = mub + matfuncs.dot(np.linalg.inv(c.T), rv)
    return(draw)
    
@nb.jit(nopython=True)   
def full_conditional_sigma(Y, X, theta, T, time):
    a, b = 1., 1.
    
    N = len(Y)
    a1 = a + float(N)/2
    b1 = 0.
    for t in range(1,int(T)):
        idxt = which(time, t)
        Yt   = sub_array(Y, idxt)
        Xt   = sub_row_matrix(X, idxt)
        b1 += np.sum(np.power(Yt - matfuncs.dot(Xt, theta[t]), 2))
    b1 = 1/ (b1 /2. + 1/b)
    sigma_iter = 1/np.random.gamma(a1, b1, 1)
    return(sigma_iter[0])

@nb.jit(nopython=True)   
def full_conditional_W(thetat1, thetat):
    a, b = 1., 1.
    a1         = float(len(thetat1))/2 + a
    b1         = 1/(np.sum(np.square(thetat1-thetat))/2 + b)
    sigma_iter = 1/np.random.gamma(a1, b1, 1)
    return(sigma_iter[0])

@nb.jit(nopython=True)  
def gibbs_dlm_1iter(X, Y, time, theta, sigmae, W):
    
    T = np.max(time) + 1
    K = X.shape[1]
    
    #Forward filetring
    at = np.zeros((T,K))
    Rt = np.zeros((T,K,K))
    mt = np.zeros((T,K))
    Ct = np.zeros((T,K,K))
    # Initial state
    at[0] = np.zeros(K)
    Rt[0] = np.diag(np.ones(K))
    mt[0] = np.zeros(K)               # prior over the mean of the thetas
    Ct[0] = 100 * np.diag(np.ones(K)) # prior over the variance of the thetas
    for t in range(1,T):
        idxt = which(time, t)
        Xt   = sub_row_matrix(X, idxt)
        Yt   = sub_array(Y, idxt)
        at[t] = mt[t-1]
        Rt[t] = Ct[t-1] + W
        inv_Rt = np.linalg.inv(Rt[t])
        inv_Ct = inv_Rt + matfuncs.dot(Xt.T, Xt) / sigmae
        Ct[t] = np.linalg.inv(inv_Ct)
        mt[t] = matfuncs.dot(Ct[t], matfuncs.dot(inv_Rt, at[t]) + matfuncs.dot(Xt.T, Yt) / sigmae) 
    
    # Backward smoothing
    ht = np.zeros((T,K))
    Ht = np.zeros((T,K,K))
    theta[T-1] = RanMNormalPrec(mt[int(T-1)], np.linalg.inv(Ct[int(T-1)]))
    for t in range(T-2,-1, -1):
        Ht[t] = np.linalg.inv(np.linalg.inv(Ct[t]) + np.linalg.inv(W))
        ht[t] = matfuncs.dot(Ht[t], matfuncs.dot(np.linalg.inv(Ct[t]), mt[t]) + matfuncs.dot(np.linalg.inv(W), theta[t+1]))
        theta[t] = RanMNormalPrec(ht[t], np.linalg.inv(Ht[t]))
    
    sigmae = full_conditional_sigma(Y, X, theta, T, time)
    
    w = np.zeros(int(K))
    for k in range(K):
        thetat1 = theta[1:,k]
        thetat  = theta[:int(T-1),k]
        w[k] = full_conditional_W(thetat1, thetat)
    W = np.diag(w)
    
    return(theta, sigmae, W)
    
    
"""
                ###############################
                    Splines penalization part
                ###############################
"""


@nb.jit(nopython=True)
def full_conditional_bu(num_zeros, num_splines, Knots, Z, Zt, sigma_e_iter, sigma_u_iter, y_tild):
    # prior matrix over linear and non linear effects
    # the linear effects get the prior 1/100, the splines get the 
    # penalizing prior 1/sigma_u2
    D         = get_D(num_zeros, num_splines, 1 , sigma_u_iter, Knots) 
    inv_V_bu  = (D + matfuncs.dot(Zt,Z) / sigma_e_iter)
    V_bu      = np.linalg.inv(inv_V_bu)
    mu_bu     = matfuncs.dot(V_bu, matfuncs.dot(Zt,y_tild) / sigma_e_iter)
    bu        = RanMNormalPrec(mu_bu, inv_V_bu)
    return(bu)

@nb.jit(nopython=True)    
def full_conditional_sigma_u(u):
    au, bu = 1., 1.
    a1           = au + len(u) * 0.5 
    b1           = 1/(bu + 0.5 * np.sum(np.square(u)))
    sigma_u_iter = 1/np.random.gamma(a1, b1, 1)
    return(sigma_u_iter) 


@nb.jit(nopython=True)    
def full_conditional_sigma_e(y, X, Z, b, u):
    ae, be = 1., 1.
    a1           = ae + len(y) * 0.5
    b1           = 1/(be + 0.5 * np.sum(np.square(y - matfuncs.dot(X, b) - matfuncs.dot(Z, u))))
    sigma_e_iter = 1/np.random.gamma(a1, b1, 1)
    return(sigma_e_iter) 

"""
 One Gibbs Iteration
"""
@nb.jit(nopython=True) 
def SP_Reg_iter(num_zeros, num_splines, Knots, Z, Zt, sigmae, sigma_u_iter, y_tild, theta):
    u_iter   = full_conditional_bu(num_zeros, num_splines, Knots, Z, Zt, sigmae, sigma_u_iter, y_tild)

    for j in range(num_splines):
        u_iter_j = u_iter[int(j * Knots): int((j+1) * Knots) ]
        sigma_u_iter_j = full_conditional_sigma_u(u_iter_j)
        sigma_u_iter[j] = sigma_u_iter_j[0]
        
    return u_iter, sigma_u_iter 





"""
                #################################################
                    Supervised Hierarchical Dirichlet Process
                #################################################
"""

"""
Full conditionals
"""
# Tau sampler 
@nb.jit(nopython=True)
def dirichlet_rvs(a):
    X =np.zeros(a.shape[0])
    for k in range(a.shape[0]):
        X[k] = np.random.gamma(a[k], 1., 1)[0]
    tot = np.sum(X)
    return(X/tot)

# Sampling the number of tables
@nb.jit(nopython=True)
def get_mK(tau, gamma, alpha, n_m_z, K, U1):
    active = which(U1, 1)
    mK = np.zeros(int(K))
    for m in range(n_m_z.shape[0]):
        for ktild in range(int(K)):
            k = np.int(active[ktild])
            n_mk = int(n_m_z[m][k])
            if n_mk >= 1:
                mK[ktild] += Antoniak_rvs.rand_antoniak(alpha * tau[ktild], n_mk)
    return(mK)
    
@nb.jit(nopython=True)
def sample_tau(tau, gamma, alpha, n_m_z, K, U1):
    # Updating the new tau from a Dirichlet with the appropriate number of topics
    mK  = get_mK(tau, gamma, alpha, n_m_z, K, U1)
    a_   = np.zeros(int(K + 1.))
    for k in range(int(K)):
        a_[k] = mK[k]
    a_[int(K)] = gamma
    tau = dirichlet_rvs(a_)
    T = np.sum(mK)
    return(T, tau)

@nb.jit(nopython=True)
def local_sample_tau(tau, gamma, K):
    # Breaking the stick and updating the new priors
    new_tau = np.zeros(int(K+1))
    for k in range(int(K-1)):
        new_tau[k] = tau[k]     
    b = np.random.beta(1., gamma)
    tau_K_old = tau[int(K-1)]
    new_tau[int(K-1)] = b * tau_K_old
    new_tau[int(K)] = (1.-b) * tau_K_old
    return(new_tau)
    
# Sample gamma
@nb.jit(nopython = True)
def sample_gamma(T, gamma, K):
    a_gamma = 1.
    b_gamma = 1.
    D = 5
    for d in range(D): 
        nu = np.random.beta(gamma + 1., T)    
        p = T/(T + gamma)
        u = np.random.binomial(1, p)
        gamma = np.random.gamma(a_gamma + K - 1 + u, 1./(b_gamma - np.log(nu)))
    return(gamma)


# Sample alpha
@nb.jit(nopython = True)
def sample_alpha(T, alpha, n_m):
    a_alpha = 1.
    b_alpha = 1.
    D = 5
    for d in range(D):
        w_m = np.zeros(n_m.shape[0])
        s_m = np.zeros(n_m.shape[0])
        for m in range(n_m.shape[0]):
            p = n_m[m]/(n_m[m] + alpha)
            s_m[m] = np.random.binomial(1, p)
            w_m[m] = np.random.beta(alpha + 1, n_m[m])
        sum_s = np.sum(s_m)
        sum_w = np.sum(np.log(w_m))
        alpha = np.random.gamma(a_alpha + T - sum_s, 1./(b_alpha - sum_w))
    return(alpha)


@nb.jit(nopython=True)
def Init(data, K0, Kmax, V, num_docs, max_len_doc):
    # Initialization of the count statistics
    n_m_z = np.zeros((np.int64(num_docs), np.int64(Kmax)))         # Num. words in each topic in doc m 
    n_z_t = np.zeros((np.int64(Kmax), np.int64(V)))                # Num. of times word of vocab appears in topic
    n_z   = np.zeros( np.int64(Kmax ))                             # Total word (overlapping) count of each topic
    z_m_n = np.zeros((np.int64(num_docs), np.int64(max_len_doc)))  # Words topics assignments in doc m
    # Initialize documetns assignments
    for i in range(len(data)):
        m = np.int64(data[i][0])
        n = np.int64(data[i][1])
        t = np.int64(data[i][2])
        # Assign memberships randomly
        z = np.random.randint(0, K0)
        n_m_z[m][z] += 1
        n_z_t[z][t] += 1
        n_z[z]      += 1
        z_m_n[m][n]  = z 
    return(n_m_z, n_z_t, n_z, z_m_n)

@nb.jit(nopython = True)
def init_nm(n_m_z):
    res = np.zeros(int(n_m_z.shape[0]))
    for m in range(n_m_z.shape[0]):
        tot = 0.
        for j in range(n_m_z.shape[1]):
            tot += n_m_z[m,j]
        res[m] = tot
    return(res)

@nb.jit(nopython=True)
def increment(k_star, n_m_z, n_z_t, n_z, z_m_n, m, n, t):
    # Increase the counters
    z_m_n[m][n] = k_star
    n_m_z[m][k_star] += 1.
    n_z_t[k_star][t] += 1.
    n_z[k_star]      += 1.
    return(n_m_z, n_z_t, n_z, z_m_n)

@nb.jit(nopython=True)  
def decrement(k_star, n_m_z, n_z_t, n_z, z_m_n, m, n, t):
    # Decrement the counters
    z_m_n[m][n] = k_star
    n_m_z[m][k_star] -= 1.
    n_z_t[k_star][t] -= 1.
    n_z[k_star]      -= 1.
    return(n_m_z, n_z_t, n_z, z_m_n)
     

@nb.jit(nopython=True)
def get_prob(U1, n_z_t, n_m_z, n_z, alpha, beta, tau, K, V, m, t):
    active = which(U1, 1.)
    p = np.zeros(int(K)+1)
    for k_ in range(int(K)):
        k = int(active[int(k_)])
        p[k_]    = (n_z_t[int(k), int(t)] + beta) 
        p[k_]   *= (n_m_z[int(m), int(k)] + alpha * tau[k_])
        p[k_]   /= (n_z[int(k)] + float(V) * beta)
    p[int(K)] = alpha * tau[int(K)] /  float(V) 
    # We do not need to normalize here
    res = p
    return(res)

@nb.jit(nopython = True)
def update_topics(Kmax, k_tild, K, U1, n_m_z, n_z_t, n_z, z_m_n, tau, m, n, t, gamma, alpha, Beta_iter, Beta_new, BtM):
    
    # In order for the number of topics not to explode before convergence
    if k_tild >= Kmax:
        k_tild = Kmax-1
        
    if k_tild < int(K):
        # Preserve the topic in the used topics
        active = which(U1, 1)
        k_star = active[int(k_tild)]
        # and increase the counters
        n_m_z, n_z_t, n_z, z_m_n = increment(int(k_star), n_m_z, n_z_t, n_z, z_m_n, int(m), int(n), int(t))
        BtM[int(m)] += Beta_iter[int(k_star)]
    else:
        deactive   = which(U1, 0)
        k_star     = deactive[0]
        U1[int(k_star)] = 1. 
        Beta_iter[int(k_star)] = Beta_new
        Beta_new = np.random.normal(0., 1.)
        # Incerement the counters
        n_m_z, n_z_t, n_z, z_m_n = increment(int(k_star), n_m_z, n_z_t, n_z, z_m_n, int(m), int(n), int(t))
        K += 1
        tau = local_sample_tau(tau, gamma, K)
        
        # update scalar product
        BtM = matfuncs.dot(n_m_z, Beta_iter.T)
    return(K, U1, n_m_z, n_z_t, n_z, z_m_n, tau, Beta_iter, Beta_new, BtM)

@nb.jit(nopython=True)
def multinomial_rvs(prob):
    N = prob.shape[0]
    w = np.zeros(N)
    w[0] = prob[0]
    for i in range(1,N):
        w[i] = w[i-1] + prob[i] 
    wmax = np.max(w)
    u = np.random.uniform(0.,wmax, 1)[0]
    draw = 0
    for i in range(1,N):
        if u > w[i-1]:
            if u <= w[i]:
                draw = i
    return(draw)


""""
                        ################################################
                                      Themes and memberships
                        ################################################
"""
# Compute the phi (topics/ words distribution)
@nb.jit(nopython = True)
def Compute_phi(K, V, n_z_t, beta, active):
    tmp1 = np.zeros((np.int64(K), np.int64(V)))
    tmp2 = np.zeros( np.int64(K))
    phi  = np.zeros((np.int64(K), np.int64(V)))
    # Topic probability distribution over words
    for i in range(int(K)):
        i_ = int(active[int(i)])
        z = n_z_t[i_]
        tmp2[i] = np.sum(z +  beta)
        for j in range(len(z)):
            x = z[j]
            tmp1[i][j] = x + beta
            phi[i][j] += tmp1[i][j]/tmp2[i]
    return(phi)

# Compute the theta (documents/ topics distribution)
@nb.jit(nopython = True)
def Compute_theta(num_docs, K, n_m_z, active, fixed_alpha):
    tmp1  = np.zeros((np.int64(num_docs), np.int64(K)) )
    tmp2  = np.zeros( np.int64(num_docs))
    theta = np.zeros((np.int64(num_docs), np.int64(K))) 
    # Topic probability of every document
    for i in range(num_docs):
        z = np.zeros(int(K))
        for k in range(int(K)):
            k_ = int(active[k])
            z[k] = n_m_z[i][k_]
        tmp2[i] = np.sum(z + fixed_alpha) 
        for j in range(len(z)):
            x = z[j]
            tmp1[i][j]   = x + fixed_alpha[j] 
            theta[i][j] += tmp1[i][j]/tmp2[i]   
    return(theta)

@nb.jit(nopython = True)
def find_max_topic(L, last = 1.):
    IDX = np.array(list(set(L)))
    res = np.zeros(len(IDX))
    for i in range(int(last * len(L))-1,len(L)):
        res[int(which(IDX , L[i])[0])] += 1
    res_K = IDX[int(which(res, np.max(res))[0])]
    return(res_K)


""""
                                ===================================
                                        The supervision sHDP
                                ===================================
"""
@nb.jit(nopython=True)    
def full_conditional_Beta(y, X, sigma_iter, eta, inv_C):
    inv_Vb = inv_C + (matfuncs.dot(X.T,X))/sigma_iter 
    Vb     = np.linalg.inv(inv_Vb)
    X_tild = matfuncs.dot(inv_C, eta) + (matfuncs.dot(X.T, y))/ sigma_iter
    mub    = matfuncs.dot(Vb, X_tild)
    Beta_iter = RanMNormalPrec(mub, inv_Vb)
    return(Beta_iter)


# The regression error will be estimated with the splines
"""    
@nb.jit(nopython=True)    
def full_conditional_sigma(y, X, Beta_iter):
    a, b       = 1., 1.
    a1         = float(len(y)/2 + a)
    b1         = 1/(np.sum(np.square(y-matfuncs.dot(X, Beta_iter)))/2 + b)
    sigma_iter = 1/np.random.gamma(a1, b1, 1)
    return(sigma_iter[0])
"""
    
@nb.jit(nopython=True) 
def get_active_Zbar(n_m_z , n_m, U1, K):
    res = np.zeros((n_m.shape[0], int(K)))
    active = which(U1, 1.)
    for m in range(n_m.shape[0]):
        for k, k_ in enumerate(active):
            res[m,int(k)] = n_m_z[m,int(k_)] /n_m[m]
    return(res)
    

@nb.jit(nopython=True) 
def get_pshdp(phdp, BtMd, K, yd, Beta_iter, Beta_new, sigma_iter, Nd, active):
    beta = np.zeros(int(K+1))
    for t, k in enumerate(active):
            beta[int(t)] = Beta_iter[int(k)]
    beta[int(K)] = Beta_new
            
    res = phdp* np.exp(-0.5 * np.power(yd -  BtMd/Nd - beta/Nd, 2)  / sigma_iter)
    return(res)  

@nb.jit(nopython=True)
def Compute_beta(Beta_iter, active):
    res = np.zeros(active.shape)
    for k, k_ in enumerate(active):
        res[int(k_)] = Beta_iter[int(k)]
    return(res)
    
    
""""
                                ===================================
                                        The collapsed Gibbs
                                             --MAIN--
                                ===================================
"""
@nb.jit(nopython = True)
def DSPsHDP(Iter, y, time, data, max_len_doc, num_docs, V, xL, xS, Knots, p):
    """
    Non prametric Regression
    """
    
    """
    Dynamic part
    """
    maxT = np.max(time) + 1
    
    """
    Spline part
    """
    # We allow for an intercept here
    # this is the linear part of the matrix
    X = get_X(xL, xS, p)
    # this accounts for the fluctuation in the splines
    Z  = get_Z(xS, Knots, p)
    Zt = Z.T
    # we are spliting the estimation in two parts
    # thus we require the number of the lienar part (0 on the diagonal)
    # to be exactly 0
    num_zeros   = 0
    num_splines = xS.shape[1]
    
    """
    Initializing parameters
    """
    ### Dynamic part
    theta  =  np.zeros((maxT,X.shape[1]))
    sigmae = 1
    W      = 1 * np.diag(np.ones(X.shape[1])) 
    ### Spline part
    sigma_u_iter = np.ones(int(num_splines)) * 1000 #np.random.random(num_splines)
    U_iter       = np.random.random(Z.shape[1])
    
    """
    sHDP
    """
    # Priors fixed
    alpha = 0.01
    beta  = 0.50 # Dispersion of topics
    gamma = 0.01
    
    # Initial number of themes K0, maximum number of themes is Kmax
    K0   = 30
    Kmax = 200
    K    = K0
    
    # Init the state of the sample: dynamic/static
    Fixed_K = False
    
    # Lists of used (1) and unused topics (0)
    U1          =  np.zeros(int(Kmax)) # Activated topics
    U1[:int(K)] = 1.
    
    # Init 
    n_m_z, n_z_t, n_z, z_m_n = Init(data, K0, Kmax, V, num_docs, max_len_doc)
    n_m = init_nm(n_m_z)
    Zbar = get_active_Zbar(n_m_z , n_m, U1, K)
    
    # Initialization of dirichlet paramters tau
    tau   = np.ones(int(K)) / np.float(K)
    T,tau = sample_tau(tau, gamma, alpha, n_m_z, K, U1)
    gamma = sample_gamma(T, gamma, int(K))
    alpha = sample_alpha(T, alpha, n_m)
    
    # Regression history    
    Beta_iter          = np.zeros(int(Kmax))
    Beta_iter[int(K):] = 0
    Beta_iter_tmp      = np.zeros(int(K))
    Beta_new           = np.random.normal(0, 100.)
    
    """ History and caching """
    # History for saving the results    
    Hist = np.zeros(int(Iter))
    """
         Start the Markov chain
    """
    for iteration in range(Iter):
        """
        Sample Managerial, Acoustical and Sonor effects
        """
        
        # sample spline
        yhat0   = matfuncs.dot(Zbar, Beta_iter_tmp)
        yhat1   =  matfuncs.dot(X, theta.T)
        y_tild1 = np.zeros(Y.shape) 
        for t in range(Y.shape[0]):  
            y_tild1[t] = Y[t] - yhat1[t, time[t]] - yhat0[t]
           
        U_iter, sigma_u_iter = SP_Reg_iter(num_zeros, num_splines, Knots, Z, Zt, sigmae, sigma_u_iter, y_tild1, theta)
        
        # sample SSM
        yhat2   =  matfuncs.dot(Z, U_iter)
        y_tild2 = Y - yhat2 - yhat0
        theta, sigmae, W = gibbs_dlm_1iter(X, y_tild2, time, theta, sigmae, W)
        
        """
        Themal effects
        """
        # Themes effects before sampling
        BtM = matfuncs.dot(n_m_z, Beta_iter)
        # Keep what is not explained by the acoustics
        yres = np.zeros(Y.shape[0])
        for t in range(Y.shape[0]):
            yres[t] = Y[t] - (yhat1[t, time[t]] + yhat2[t])

        for i in range(len(data)):
            m = int(data[i][0])
            n = int(data[i][1])
            t = int(data[i][2])
            
            # Discount for n-th word t with topic z
            z = z_m_n[m][n]
            n_m_z, n_z_t, n_z, z_m_n = decrement(int(z), n_m_z, n_z_t, n_z, z_m_n, int(m), int(n), int(t))
            
            # Number of words in the document and the BtZ of the document
            Nd = np.int(n_m[m])
            yd = yres[m]
            # Update scalar product
            BtM[m] -= Beta_iter[int(z)]
            BtMd    = BtM[m]
        
            # Tockenrobabilties 
            phdp   = get_prob(U1, n_z_t, n_m_z, n_z, alpha, beta, tau, K, V,int(m), int(t)) 
            active = which(U1, 1.)
            prob   = get_pshdp(phdp, BtMd, K, yd, Beta_iter, Beta_new, sigmae, Nd, active)

            if Fixed_K == True:
                # Do not allow for sampling new topics
                prob[int(K)] = 0.
                
            k_tild = multinomial_rvs(prob)
            K, U1, n_m_z, n_z_t, n_z, z_m_n, tau, Beta_iter, Beta_new, BtM = update_topics(int(Kmax), int(k_tild), int(K), U1, n_m_z, n_z_t, n_z, z_m_n, tau, int(m), int(n), int(t), gamma, alpha, Beta_iter, Beta_new, BtM)
    
        # Check the topics still activated
        del_idx    = which(n_z, 0.)
        if del_idx[0]  >= 0.:
            for k in del_idx:
                U1[int(k)] = 0.
                Beta_iter[int(k)] = 0.
            K = np.sum(U1)
        
        """
        Update HDP priors
        """
        # Update the priors
        T,tau = sample_tau(tau, gamma, alpha, n_m_z, int(K), U1)
        gamma = sample_gamma(T, gamma, int(K))
        alpha = sample_alpha(T, alpha, n_m)
        

        """
        Sample Thematic parameters
        """
        Zbar = get_active_Zbar(n_m_z , n_m, U1, K)
        # The priors are dynamic and adjust to the number of topics that are sampled        
        eta   = np.zeros(int(K))
        inv_C = np.diag(np.ones(int(K))) * 0.01      
        Beta_iter_tmp = full_conditional_Beta(yres, Zbar, sigmae, eta, inv_C)
        # Relate the coefficients to the active topics and update the new topics sample
        active = which(U1, 1.)
        for t, k in enumerate(active):
            Beta_iter[int(k)] = Beta_iter_tmp[int(t)]
        Beta_new = np.random.normal(0, 100.)
        
        """
        Checking convergence and caching the results
        """
        # Once a reasonable number of iterations is reached 
        # we fix the topics and start sampling
        if iteration >= int(0.9 * Iter):
            if Fixed_K == False:
                # Guess which number of topics was sampled the most
                # in the last 90% iterations
                if Iter > 5:
                    # loc the best number of topics as soon as the counter gets back to it
                    best_K = find_max_topic(Hist[:iteration], 0.75)
                    if K == best_K:
                        iter_left = Iter - iteration
                        Fixed_K = True
                        # Update the priors accordingly
                        T,tau = sample_tau(tau, gamma, alpha, n_m_z, int(K), U1)
                        # Updating the hyperpriors
                        gamma = sample_gamma(T, gamma, int(K))
                        alpha = sample_alpha(T, alpha, n_m)
                        
                # Init Gibbs samples history
                SIGMA_E = np.zeros( np.int64(Iter - iteration))
                SIGMA_U = np.zeros((np.int64(Iter - iteration), np.int64(num_splines)))
                #BU      = np.zeros((np.int64(Iter - iteration), np.int64(C.shape[1]))) 
                iter_hist = 0
                # Init phi and theta
                phi   = np.zeros((np.int64(K), np.int64(V)))
                theta_topic = np.zeros((np.int64(num_docs), np.int64(K)))
                # Regression results
                B     = np.zeros((np.int64(Iter - iteration), np.int64(Kmax)))
                # The splines 
                thetaH   = np.zeros((np.int64(Iter - iteration), np.int64(maxT), X.shape[1]))
                WH       = np.zeros((np.int64(Iter - iteration), X.shape[1], X.shape[1]))
                UH       = np.zeros((np.int64(Iter - iteration), Z.shape[1]))

            if Fixed_K == True:                            
                # Fix the theta prior
                fixed_alpha = alpha * tau[:-1]
                # The estimated phi/theta
                active = which(U1, 1.)
                phi   += Compute_phi(K, V, n_z_t, beta, active) / float(iter_left)
                theta_topic += Compute_theta(num_docs, K, n_m_z, active, fixed_alpha) / float(iter_left)
                
                #S     += sigma_iter/ float(iter_left)
                B[int(iter_hist)]          = Beta_iter
                #BU[int(iter_hist)]         = BetaU_iter
                SIGMA_E[int(iter_hist)]    = sigmae
                SIGMA_U[int(iter_hist)]    = sigma_u_iter
                
                # saving the history of the SS plines
                thetaH[int(iter_hist)]   = theta
                #sigmae_H[iteration] = sigmae
                #sigmau_H[iteration] = sigma_u_iter
                WH[int(iter_hist)]       = W
                UH[int(iter_hist)]       = U_iter
                
                iter_hist += 1.
        Hist[iteration] = K
        
    return(Hist, phi, theta_topic, B, active, thetaH, WH, UH, SIGMA_E, SIGMA_U)



    
"""
                            ======================================
                                         Test Example
                            ======================================
"""
rawdocs = ["1 2 3 4 5 6 8 9 10 11 12",
           "a b c d e f g h i j k l",
           "1 2 3 4 5 6 8 9 10 11 12 a b c d e f g h i j k l" ]
         
ytopics = np.array([-2, 1, 0]) * 5


T      = 3 # max time
K      = 4   # number of parameters
Nt     = 500 # number of observations per time t
sigmae = 1   # regression noise
W      = 10 * np.diag(np.ones(K)) # state noise
theta = np.zeros((T,K))


# State updating equation
true_theta = np.zeros((T,K))
true_theta[0] =  np.random.multivariate_normal(np.zeros(K), 100*np.diag(np.ones(K)))
for t in range(1,T):
    true_theta[t] = true_theta[t-1] + np.random.multivariate_normal(np.zeros(K), W)
    
# Observations generation
X  = []
xL = []
xS = []
Y  = []
time = []
Id = range(T)
data_text = []

l = 0
for t in range(1,T):
    for l in range(Nt):
        # Independent variables
        # Linear part
        x1 = np.random.uniform(0, 5, 1)[:,None]
        x2 = np.random.uniform(0, 5, 1)[:,None]
        # Non linear part
        x3 = np.random.uniform(0, 5, 1)[:,None]
        x4 = np.random.uniform(0, 5, 1)[:,None]
        # error term
        e  = np.random.normal(0, sigmae, 1)[:,None]
        # dependent variable
        
        topic_idx = np.random.choice(3,1)[0]
        data_text.append(rawdocs[topic_idx])
        
        #y  =  x1 + x2 + 3 * x3 *np.cos((x3 - 3)*5) + 3*x4*np.cos(x4*5) + e
        #y  = y * np.power(-1, t)
        x = np.concatenate((x1, x2, x3, x4), axis =1)
        y = ytopics[topic_idx] + matfuncs.dot(x,true_theta[t]) + np.random.normal(0, sigmae, 1)[0]
        
        # The linear effects
        xl = np.concatenate((x1, x2), axis =1)
        # The non linear effects
        xs = np.concatenate((x3, x4), axis =1)
        # Saving the variables
        xL.extend(xl)
        xS.extend(xs)
        X.extend(x)
        Y.extend(y)
        time.append(t)

#X = np.matrix(X)
xL = np.matrix(xL)
xS = np.matrix(xS)
Y = np.array(Y)
time = np.array(time)
rawdocs = np.array(data_text)


# Numbers of linear and non linear effects
nL = xL.shape[1]
nS = xS.shape[1]

#Splines paramters
# The number of knots
Knots = 5
p     = 2

# Formatting the data
data_obj    = Utils.Format_data(rawdocs)
data        = data_obj['data']
max_len_doc = data_obj['max_len_doc']
num_docs    = data_obj['num_docs']
V           = data_obj['V']
vocab = data_obj['vocab']

"""
Estimate run time
"""
Iter = 10
time1 = thetime.time()
Hist, phi, theta_topic, B, active, thetaH, WH, UH, SIGMA_E, SIGMA_U = DSPsHDP(Iter, y, time, data, max_len_doc, num_docs, V, xL, xS, Knots, p)
time2 = thetime.time()
delta = round(time2-time1, 3) / float(Iter)


Iter = 10000
print ('The estimation will take about ' 
		+  str(delta * Iter/60.) 
		+  'minutes to run  from ' 
		+ str(datetime.datetime.now()))


"""
Run it
"""
time1 = thetime.time()
Hist, phi, theta_topic, B, active, thetaH, WH, UH, sigmae_H, sigmau_H = DSPsHDP(Iter, y, time, data, max_len_doc, num_docs, V, xL, xS, Knots, p)
time2 = thetime.time()
print ('The estimation took ' +  str(round(time2-time1, 3)/float(Iter)) +  's to run \n')
   
      
plt.figure(facecolor = "white")
plt.subplot(1, 2, 1)
plt.hist(Hist[int(0.75 * Iter):], color = "gray", bins =    10)
plt.subplot(1, 2, 2)
plt.plot(Hist, color = "red")

np.mean(B, axis = 0)
np.mean(thetaH, axis = 0)
np.mean(WH, axis = 0)
np.mean(UH, axis = 0)
np.mean(sigmau_H, axis = 0)
np.mean(sigmae_H, axis = 0)



"""
Describe the results
"""
topics = Utils.describe_topics(10, phi, vocab)
print("Topics description with vocabulary\n")
for k in range(phi.shape[0]):
    print(topics[k][:10])
    print("--------- \n")

docs_topics = Utils.describe_docs(2, 3, theta_topic, phi, vocab)

print("Documents description with words\n")
for d, doc in enumerate(rawdocs):
    print(doc)
    print(docs_topics[d])
    print("--------- \n")
    if d >= 10:
        break


"""
estimates
"""    
import seaborn    as sns; sns.set(color_codes=True)
# The estimated intercept and the linear effects
bL  = thetaH[:,:,: nL] 
np.mean(bL, axis = 0)






# plots of the non-linear effects
def get_x_range(x):
    x_simul = np.zeros((100, x.shape[1]))
    for j in range(x.shape[1]):
        x_simul[:,j] = np.linspace(np.min(x[:,j]), np.max(x[:,j]), num = 100)
    return(x_simul)
    

# time evolution
def plot_spline_pure_effect(j):

    xL_ = np.zeros((100, xL.shape[1]))
    xS_ = get_x_range(xS)
    for k in range(xS_.shape[1]):
        if k != j:
            xS_[:,k] = 0
    
    X = get_X(xL_, xS_, p)
    Z = get_Z(xS_, Knots, p)
    
    for t in range( np.max(time) + 1):
        yhat1   =  matfuncs.dot(X, thetaH[:,t,:].T)
        yhat2   =  matfuncs.dot(Z, UH.T)

        yhat = yhat1 + yhat2
        yhat = yhat - np.mean(yhat)
        sns.tsplot(data=yhat.T,ci=[95])
        plt.axhline(y = 0, color='black', linestyle='--')

plt.figure(facecolor = "white")
for j in range(xS.shape[1]):
    plt.subplot(1,2, 1+j)
    plot_spline_pure_effect(j)



  
#from mpl_toolkits.mplot3d import Axes3D 
#3D time evolution
    
from mpl_toolkits.mplot3d import Axes3D
def plot_spline_pure_effect(j):

    xL_ = np.zeros((100, xL.shape[1]))
    xS_ = get_x_range(xS)
    for k in range(xS_.shape[1]):
        if k != j:
            xS_[:,k] = 0
    
    X = get_X(xL_, xS_, p)
    Z = get_Z(xS_, Knots, p)
    
    ax = plt.subplot(projection='3d')
    for t in range(1, np.max(time) + 1):
        yhat1   =  matfuncs.dot(X, thetaH[:,t,:].T)
        yhat2   =  matfuncs.dot(Z, UH.T)

        yhat = yhat1 + yhat2
        yhat = yhat - np.mean(yhat)
        surf = np.mean(yhat, axis =1 )
        ax.plot(xS_[:,j], np.ones(100) * t ,surf, color='b')
        #ax.add_collection3d(pl.fill_between(x, 0.95*z, 1.05*z, color='r', alpha=0.3), zs=t, zdir='y')
        ax.set_ylabel('Decade')
        ax.set_zlabel('Success') 


for j in range(xS.shape[1]):
    plt.figure(facecolor = "white")
    plt.subplot(1,2, 1+j)
    plot_spline_pure_effect(j)
