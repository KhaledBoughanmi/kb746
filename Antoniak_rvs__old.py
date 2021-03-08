"""
### ANTONIAK RANDOM NUMBER GENERATION ####
### REQUIRED FOR DIRECT ASSIGNMENT ESTIMATION FOR sHDP ####
### SAMPLES NUMBER OF TABLES IN CHINESE RESTAURANT PROCESS ###
This code is tailored to the main algorithm and numba optimized from source:
https://gist.github.com/tdhopper/80dbf2582e12ab5d08e1
"""
        
        
import numpy as np
import numba as nb
np.core.arrayprint._line_width  = 200
np.set_printoptions(suppress=True)


@nb.jit(nopython=True)
def stirling(n, k):
    row = np.zeros(k+1)
    row[0] = 1 
    for i in xrange(1, n+1):
        new = np.zeros(k+1)
        for j in xrange(1, k+1):
            stirling_ = (i-1) * row[j] + row[j-1]
            new[j] = stirling_
        row = new
    return row[k]


max_nk = 100
@nb.jit(nopython=True)
def get_Stir_Hist(max_nk):
    res = np.zeros((int(max_nk + 1), int(max_nk + 1)))
    res[0,0] = 1
    for nk in range(max_nk + 1):
        for m in range(nk + 1):
            if m < nk:
                res[nk,m] = stirling(nk, m)
    return(res)    

# Global variable cache
# The Stirling number are generated and cached for 
# Time optimization 
get_Stir_Hist(10)
STIR_ARR = get_Stir_Hist(max_nk)

    
@nb.jit(nopython=True)
def normalized_stirling_numbers(nn):
    #  * stirling(nn) Gives unsigned Stirling numbers of the first
    #  * kind s(nn,*) in ss. ss[i] = s(nn,i). ss is normalized so that maximum
    #  * value is 1. After Teh (npbayes).
    ss = np.zeros(nn)
    for i in range(1, nn + 1):
        ss[i-1] = STIR_ARR[nn, i]
        #ss[i-1] = stirling(nn, i)
    max_val = np.max(ss)
    return(ss / max_val)

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
    
@nb.jit(nopython=True)
def rand_antoniak(alpha, n):
    # Sample from Antoniak Distribution
    # cf http://www.cs.cmu.edu/~tss/antoniak.pdf
    p = normalized_stirling_numbers(n)
    aa = 1
    for i, _ in enumerate(p):
        p[i] *= aa
        aa *= alpha
    # the division under numba is not ver stable
    # I replace the sampling from pribabilities to sampling from weights
    #p = p / np.sum(p)
    res = multinomial_rvs(p) + 1
    #res = np.random.multinomial(1, p).argmax() + 1
    return(res)
    
    
"""
# Different tests for the functions 
# to check if they work
assert stirling(9, 3) == 118124
assert stirling(9, 3) == 118124
assert stirling(0, 0) == 1
assert stirling(1, 1) == 1
assert stirling(2, 9) == 0
assert stirling(9, 6) == 4536

ss1 = np.array([1])
ss2 = np.array([1, 1])
ss10 = np.array([  3.09439754e-01,   8.75395242e-01,   1.00000000e+00,
         6.17105824e-01,   2.29662318e-01,   5.39549757e-02,
         8.05832694e-03,   7.41877718e-04,   3.83729854e-05,
         8.52733009e-07]) # Verified with Yee Whye Teh's code

assert np.sqrt(((normalized_stirling_numbers(1) - ss1)**2).sum()) < 0.00001
assert np.sqrt(((normalized_stirling_numbers(2) - ss2)**2).sum()) < 0.00001
assert np.sqrt(((normalized_stirling_numbers(10) - ss10)**2).sum()) < 0.00001

rand_antoniak(0.5, 10)
"""



