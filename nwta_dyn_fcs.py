'''
General nWTA dynamics for all-to-all coupled winner-take-all network as 
introduced in 
"Robust parallel decision-making in neural circuits with nonlinear 
inhibition" by B. Kriener, R. Chaudhuri, and Ila R. Fiete
 
Code for simulation of (n)WTA dynamics. 
Author: B Kriener, R Chaudhuri (September 2020)   
'''
import numpy as np
from numpy.random import randn
import scipy as sc
from scipy.stats import *
from scipy.special import *




def rect(v,theta):
    '''
    returns rectified vector x, s.t.
    0. if x[i] <= theta else x[i]
    '''
    hv=v.copy()
    hv[np.where(hv<theta)] = 0.
    return hv


def nWTAdyn_T(alpha, beta, N, x0, b, tau, tau_eta, sigma, theta, T=10000, dt=0.001, NL=True):
    '''
    Simulates nWTA-dynamics for a given time T in steps of dt (see Eqn. (1) in paper)
    Parameters:
    alpha    : strength of excitatory feedback (for stable WTA dynamics: alpha<1, 
               see Xie et al. (2002) Neural Computation 14, 2627-2646)
    beta     : strength of inhibitory feedback (for WTA dynamics: (1-alpha)<beta, 
               see Xie et al. (2002))
    N        : number of nodes in the network
    x0       : vector of initial conditions
    b        : vector of mean external input drives
    tau      : neural time constant
    tau_eta  : correlation time constant of OU noise
    sigma    : noise amplitude
    theta    : threshold for inhibition
    NL       : bool, if True nWTA (with nonlinear inhibition), 
               otherwise: conventional WTA
    '''

    # initialize storing vectors and dynamical variables:
    l1    = np.zeros((int(T/dt),N))
    l2    = np.zeros((int(T/dt),N))
    b     = np.sort(b)     # here: largest input should have index -1 (in paper has index 0)
    x     = x0
    eta   = np.zeros(N)
    sigma = sigma*np.sqrt(2*tau_eta) # rescale OU-amplitude to GWN amplitude for generation of OU noise  
    for it in range(int(T/dt)):
        # deterministic part:
        xrr = (alpha*x - beta*sum(rect(x,theta)) + beta*rect(x,theta)) if NL else ((alpha+beta)*x - beta*sum(x)) 
        # OU noise generation:
        eta = eta * np.exp(-dt/tau_eta) + sigma * randn(N) * np.sqrt((1. - np.exp(-2.*dt/tau_eta) ) / 2. / tau_eta)
        # rate:
        rr  = rect(b + xrr + eta, 0)
        # activation:
        x   = x +  dt/tau *(rr - x)
        # store in vectors:
        l1[it,:] = x
        l2[it,:] = rr
    return l1,l2



def nWTA_selfterm(alpha, beta, N, x0, b, sigma, tau, tau_eta, theta, dt=0.001, NL=True):
    '''
    Simulates nWTA-dynamics with self-terminating dynamics and time step dt (see Eqn. (1) in paper)
    Parameters:
    alpha    : strength of excitatory feedback (for stable WTA dynamics: alpha<1, 
               see Xie et al. (2002) Neural Computation 14, 2627-2646)
    beta     : strength of inhibitory feedback (for WTA dynamics: (1-alpha)<beta, see Xie et al. (2002))
    N        : number of nodes in the network
    x0       : vector of initial conditions
    b        : vector of mean external input drives
    tau      : neural time constant
    tau_eta  : correlation time constant of OU noise
    sigma    : noise amplitude
    theta    : threshold for inhibition
    NL       : bool, if True nWTA (with nonlinear inhibition), otherwise: conventional WTA
    '''
    b     = np.sort(b)     # here: largest input should have index -1 (in paper has index 0)
    x     = x0             # initial condition
    sx    = np.sort(x)     # initial condition sorted vector
    cnt   = 0              # keep count of steps
    eta   = np.zeros(N)    # initialize noise vector
    stopc = 0.8*max(b)/(1.-alpha) # stopping criterion (80% of asymptotic activity of node with
                                  # largest b in deterministic conventional WTA dynamics)
    sigma = sigma*np.sqrt(2*tau_eta) # rescale OU-amplitude to GWN amplitude for generation of OU noise                              
    # while largest current activation is smaller than stopc                           
    while(sx[-1]<stopc):    
        # deterministic part:
        xrr = (alpha*x - beta*sum(rect(x,theta)) + beta*rect(x,theta)) if NL else ((alpha+beta)*x - beta*sum(x)) 
        # OU noise generation:
        eta = eta * np.exp(-dt/tau_eta) + sigma * randn(N) * np.sqrt((1. - np.exp(-2.*dt/tau_eta) ) / 2. / tau_eta)
        # rate:
        rr  = rect(b + xrr + eta, 0)
        # activation:
        x   = x +  dt/tau *(rr - x)
        sx  = np.sort(x)
        # abort simulation, if WTA fails:
        if (cnt>100./dt) and (mod(cnt,100./dt)==0) and (sx[-2]/sx[-1]>0.5):
            break
        cnt = cnt + 1
    # return time, activation of node with largest and second largest drive, activation of largest and second largest activation
    # if sx[-1]==x[-1] accurate; x[-2],sx[-2] help to assess wta-fraction
    return cnt*dt, x[-1], x[-2], sx[-1], sx[-2] 



