'''
Code to compute stability of the convential winner-take-all regime in the 
presence of noise as discussed in the paper 
"Robust parallel decision-making in neural circuits with nonlinear 
inhibition" by B. Kriener, R. Chaudhuri, and Ila R. Fiete

See SI Appendix S2.6 for reference.
 
Author: B Kriener (September 2020)   
'''

import numpy as np
from numpy.random import *
import scipy 
from scipy.stats import *
from scipy.special import erf,erfc
import matplotlib.pyplot as pl

## Effective mean of rectified Gaussian noise with mean mu and std sigma,
## see SI Appendix S2.6, Eqn.(19)
def mu_trunc(mu,sigma):
    '''
    Returns mean of truncated Gaussian noise of mean mu and standard deviation 
    sigma. See SI appendix S2.6, Eqn.(19)

    '''
    return (mu + np.exp(-mu**2/2./sigma**2)*np.sqrt(2./np.pi)*sigma + mu*erf(mu/np.sqrt(2.)/sigma))/2.


### Function to find fixed point rates:
def fp_xw_xl(m0i, m1i, alpha, beta, N, bl, bw, sigma):
    '''
    Computes fixed point of asymptotic winner and loser activity x(t), assuming 
    quasi-2d case (max b=bw, all other b=bl<bw) and small auto-correlation time
    constant of OUP noise tau_eta.
    See SI appendix S2.6, Eqn.(19)

    Parameters:

    alpha    : strength of excitatory feedback (for stable WTA dynamics: alpha<1, 
               see Xie et al. (2002) Neural Computation 14, 2627-2646)
    beta     : strength of inhibitory feedback (for WTA dynamics: (1-alpha)<beta, 
               see Xie et al. (2002))
    N        : number of nodes in the network
    bw,bl    : mean external input drives of winner and loser, respectively
    sigma    : noise amplitude (of OUP)

    '''
    def stat(m):
        return [m[0]-mu_trunc(alpha*m[0] - beta*(N-1)*m[1] + bw, sigma),\
                m[1]-mu_trunc(alpha*m[1] - beta*(N-2)*m[1] - beta*m[0] + bl, sigma)]
    m = scipy.optimize.fsolve(stat,[m0i, m1i])
    return m    


N           = 100                # number of neurons
alpha, beta = 0.6, 0.5           # strength of self-excitation, lateral inhibition
#alpha, beta = 1.-1./2./N, 1./N  # 1/N scaling of inh with alpha<1 for stab
Delta       = 0.05               # gap size
b           = np.ones(N)-Delta   # create quasi 2d input drives
b[-1]       = 1.                 # largest drive is Delta bigger than rest
tau_eta     = 0.005              # noise auto-correlation time constant
tau         = 1.                 # neural time constant
SGMAS = np.arange(0.05,0.18,0.001) # OUP noise amplitudes
l1    = np.zeros((len(SGMAS),3))
l2    = np.zeros((len(SGMAS),3))
cnt   = 0
for sigma in SGMAS:
    # WTA branches:
    a1,a2      = fp_xw_xl(1./(1-alpha),0.,alpha,beta,N,b[0],b[-1],sigma)
    l1[cnt,:]  = sigma,a1,a2
    # no-WTA branches:
    a1,a2      = fp_xw_xl(1.,1.,alpha,beta,N,b[0],b[-1],sigma)
    l2[cnt,:]  = sigma,a1,a2
    cnt+=1

    
pl.figure()
pl.plot(l1[:,0],l1[:,1],'--',color='k',lw=5)   # WTA branch, largest b=1 (winner)
pl.plot(l1[:,0],l1[:,2],'--',color='0.6',lw=5) # WTA branch, b=1-Delta (losers)
pl.plot(l2[:,0],l2[:,1],color='k',lw=5)        # no-WTA branch, largest b=1 (winner)
pl.plot(l2[:,0],l2[:,2],color='0.6',lw=5)      # no-WTA branch, b=1-Delta (losers)
pl.xlabel(r'noise amplitude $\sigma$', size=24)
pl.ylabel(r'activity x$^\infty$', size=20)
pl.xticks(size=18)
pl.yticks(size=18)
pl.tight_layout()
pl.show()
