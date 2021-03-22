import matplotlib.pyplot as plt
import numpy as np
import GRAPE
import scipy.optimize as opt
import pdb
import time as t
import spin_chain
# Here I play with optimization 
Nqubits = 5 # number of qubits
sp = spin_chain.Spin_chain(Nqubits) # initialize the class

G = GRAPE.GRAPE()
TQSL = 3.33#np.arcsin(np.sqrt(0.5))*np.sqrt(2)#np.pi/np.sqrt(2)

T = 1.0*TQSL

Nlist = np.array([1,2,3,4,5,10,50,100])
Tlist = np.arange(0,11)/5
N = 20
tol = 0.02

#pulse = np.random.rand(1,N)

#H0 = -0.5*np.array([[1,0],[0,-1]], dtype = complex)
#Hc = -0.5*np.array([[[0,1],[1,0]]], dtype = complex)
H0 = -1.0*sp.ZZ
Hc = np.copy(sp.X)
#psi0 = np.array([1,0], dtype = complex)
#psi_target = np.array([0,1], dtype = complex)

amax = 4
amin = -4

#eigvals, eigvecs = np.linalg.eigh(H0 -2*Hc[0])
#psi0 = eigvecs[:,0]

#eigvals, eigvecs = np.linalg.eigh(H0 + 2*Hc[0])
#psi_target = eigvecs[:,0]
psi0 = np.zeros((sp.dim), dtype = complex)
psi0[0] = 1.0 # Initial state is |0,0,...,0>
psi_t = np.zeros((sp.dim), dtype = complex)
psi_t[-1] = 1.0 # Target state is |1,1,...,1>
#pdb.set_trace()

bounds = opt.Bounds(lb = amin*np.ones((N)), ub = amax*np.ones((N)))

options={'disp': None, 'maxcor': 10, 'ftol': 1e-15, 
'gtol': 1e-15, 'eps': 1e-15, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 
'maxls': 20}
t0 = t.time()

def optim(x):
    
    inf, g  = G.infidelity(np.reshape(x,(1,N)), N, T, H0, Hc, psi0, psi_t)
    
    return inf, g[0]
fidel = 0
i = 0
#while fidel<1-tol or i<10:
i+= 1
pulse = (amax-amin)*np.random.rand(N)+amin
print(pulse)
res = opt.minimize(optim, pulse, method = 'L-BFGS-B', jac = True, bounds = bounds, tol = 1e-15, options = options)
    
t1 = t.time()
