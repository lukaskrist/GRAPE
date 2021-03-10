import matplotlib.pyplot as plt
import numpy as np
import GRAPE
import scipy.optimize as opt
import pdb
import time as t
# Here I play with optimization 

G = GRAPE.GRAPE()
TQSL = 2.4#np.arcsin(np.sqrt(0.5))*np.sqrt(2)#np.pi/np.sqrt(2)

T = 1.0*TQSL

Nlist = np.array([1,2,3,4,5,10,50,100])
Tlist = np.arange(0,11)/5
N = 20

#pulse = np.random.rand(1,N)

H0 = -0.5*np.array([[1,0],[0,-1]], dtype = complex)
Hc = -0.5*np.array([[[0,1],[1,0]]], dtype = complex)

#psi0 = np.array([1,0], dtype = complex)
#psi_target = np.array([0,1], dtype = complex)

amax = 4.0
amin = -4.0

eigvals, eigvecs = np.linalg.eigh(H0 -2*Hc[0])
psi0 = eigvecs[:,0]

eigvals, eigvecs = np.linalg.eigh(H0 + 2*Hc[0])
psi_target = eigvecs[:,0]

#pdb.set_trace()

bounds = opt.Bounds(lb = amin*np.ones((N)), ub = amax*np.ones((N)))

options={'disp': None, 'maxcor': 10, 'ftol': 1e-15, 
'gtol': 1e-15, 'eps': 1e-15, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 
'maxls': 20}
t0 = t.time()
for i in range(10):
    def optim(x):
    
        inf, g  = G.infidelity(np.reshape(x,(1,N)), N, T, H0, Hc, psi0, psi_target)
    
        return inf, g[0]

    pulse = (amax-amin)*np.random.rand(N)+amin

    res = opt.minimize(optim, pulse, method = 'L-BFGS-B', jac = True, bounds = bounds, tol = 1e-15, options = options)
t1 = t.time()
print(t1-t0)
print(res)