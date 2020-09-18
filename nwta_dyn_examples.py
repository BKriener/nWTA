import numpy as np
from numpy.random import randn
import scipy as sc
from scipy.stats import *
from scipy.special import *
import matplotlib.pyplot as pl 
import nwta_dyn_fcs as nw

# Plot examples:

alpha, beta = 0.5, 0.6
N           = 10
Delta       = 0.05
b           = np.ones(N)*(1.-Delta)
b[-1]       = 1.
x0          = np.zeros(N)
tau         = 1.
tau_eta     = 0.05
sigma       = 0.22
theta       = 0.2
T,dt        = 50,tau_eta/5.
x,rr        = nw.nWTAdyn_T(alpha, beta, N, x0, b, tau, tau_eta, sigma, theta, T, dt, NL=True)
times       = dt*np.arange(0,len(x))

# example traces:

pl.figure()
pl.plot(times, x, lw=2, color='0.6')
pl.plot(times, x[:,-1], lw=2, color='k')
pl.xlabel('time',size=24)
pl.ylabel('activation',size=24)
pl.xticks(size=18)
pl.yticks([0,1,2],size=18)

pl.figure()
pl.plot(times, rr, lw=2, color='0.6')
pl.plot(times, rr[:,-1], lw=2, color='k')
pl.xlabel('time',size=24)
pl.ylabel('rate',size=24)
pl.xticks(size=18)
pl.yticks([0,1,2],size=18)

pl.show()

# self-terminating dynamics:

time, x1, x2, sx1, sx2 = nw.nWTA_selfterm(alpha, beta, N, x0, b, sigma, tau, tau_eta, theta, dt, NL=True)

print 'decision time: ',time
print 'correct winner: ', x1==sx1
