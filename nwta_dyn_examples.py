'''
General nWTA dynamics for all-to-all coupled winner-take-all network as 
introduced in 
"Robust parallel decision-making in neural circuits with nonlinear 
inhibition" by B. Kriener, R. Chaudhuri, and Ila R. Fiete
 
Examples of how to use simulation code of (n)WTA dynamics. See Eqns.(1),(4).
Author: B Kriener, R Chaudhuri (September 2020)   
'''

import numpy as np
from numpy.random import randn
import scipy as sc
from scipy.stats import *
from scipy.special import *
import matplotlib.pyplot as pl 
import nwta_dyn_fcs as nw

# Plot examples:

alpha, beta = 0.5, 0.6    # strength of self-excitation, lateral inhibition
N           = 10          # network size
Delta       = 0.05        # gap size between largest and second-largest input
b           = np.ones(N)*(1.-Delta) # mean input drives, quasi-2d
b[-1]       = 1.          # largest input drive is Delta bigger than second drive
x0          = np.zeros(N) # initial conditions 
tau         = 1.          # neural time constant
tau_eta     = 0.05        # noise auto-correlation time constant
sigma       = 0.22        # noise amplitude
theta       = 0.2         # threshold of nonlinear inhibition
T,dt        = 50,tau_eta/5. # total simulation time, simulation time step
x,rr        = nw.nWTAdyn_T(alpha, beta, N, x0, b, tau, tau_eta, sigma, theta, T, dt, NL=True)
times       = dt*np.arange(0,len(x))

# example traces:

pl.figure()
pl.plot(times, x, lw=2, color='0.6')
pl.plot(times, x[:,-1], lw=2, color='k')
pl.xlabel('time',size=24)
pl.ylabel('activity x(t)',size=24)
pl.xticks(size=18)
pl.yticks([0,1,2],size=18)
pl.tight_layout()

pl.figure()
pl.plot(times, rr, lw=2, color='0.6')
pl.plot(times, rr[:,-1], lw=2, color='k')
pl.xlabel('time',size=24)
pl.ylabel('rate r(t)',size=24)
pl.xticks(size=18)
pl.yticks([0,1,2],size=18)
pl.tight_layout()
pl.show()

# self-terminating dynamics:

time, x1, x2, sx1, sx2 = nw.nWTA_selfterm(alpha, beta, N, x0, b, sigma, tau, tau_eta, theta, dt, NL=True)

print ('decision time: ',time)
print ('correct winner: ', x1==sx1)
