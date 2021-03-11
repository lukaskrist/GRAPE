import numpy as np
import scipy.linalg as la
import pdb
import time

class GRAPE:
    
    def __init__(self):
        print("GRAPE initialized")
        
        
    def infidelity(self, pulse, N, T, H0, Hc, psi0, psi_target):
        # H0, dxd matrix
        # Hc m,dxd matrix, with m being the number of controls
        # pulse [m,n] amplitudes
        M = Hc.shape[0]
        d = H0.shape[0]
        dt = T/N
        assert H0.shape == (d,d), 'H0 must be square matrix'
        #assert Hc.shape == (M,d,d), 'Hc dimension mismatch'
        assert pulse.shape == (M,N), 'pulse dimension mismatch'
        assert psi0.shape == (d,), 'psi0 mismatch'
        assert psi_target.shape == (d,), 'psi_target mismatch'
        


        # Forward propagation
        psi = np.copy(psi0)
        psi_forward = np.zeros((N+1,d), dtype = complex)
        psi_backward = np.zeros((N+1,d), dtype = complex)
        psi_forward[0] = np.copy(psi0)
        psi_backward[N] = np.copy(psi_target)
        Ulist = np.zeros((N,d,d), dtype = complex)
        
        eigvecs_list = np.zeros((N,d,d), dtype = complex)
        eigvals_list = np.zeros((N,d), dtype = complex)
        exp_eigvals_list = np.zeros((N,d), dtype = complex)
        psi_story = []
        
        for step in range(0,N):
            H = np.copy(H0)
            for control in range(0,M):
                H += pulse[control,step]*Hc[control]
            
            eigvals, eigvecs = np.linalg.eigh(H)

            eigvecs_list[step] = eigvecs
            eigvals_list[step] = eigvals

            exp_eigvals = np.exp(-1j*eigvals*dt)
            
            exp_eigvals_list[step] = exp_eigvals 
            
            
            U = np.matmul(np.matmul( eigvecs , np.diag(exp_eigvals)), la.inv(eigvecs)) 
            
            psi = np.dot(U,psi)
            psi_story.append(psi)
            psi_forward[step+1] = psi
            Ulist[step] = U


        psi_T = np.vdot(psi, psi_target)    
        
        infidelity = 1-np.abs(psi_T)**2
        #print(infidelity)
        #print(psi)
        
        # Backwards propagation
        psi = np.copy(psi_target)
        psi_backward[N-1] = np.copy(psi)
        
        for step in range(N-2,-1,-1):
                    
            psi = np.dot(np.transpose(np.conjugate(Ulist[step+1])), psi)
            psi_backward[step] = psi
            

        # testing if backpropagation is done correctly
        #for step in range(0, N):
        #    print(np.vdot(psi_backward[step], np.dot(Ulist[step],psi_forward[step])) - np.vdot(psi_target,psi_forward[-1])) # this is the syntax

      
        # calculating gradient
        gradient = np.zeros(pulse.shape)
        
        for step in range(0, N):
            
            R = eigvecs_list[step]
            Rd = np.conjugate(np.transpose(R))
            I = -1j*dt*np.diag(exp_eigvals_list[step])
            
            for idx1 in range(0, d):
                for idx2 in range(idx1 + 1, d):
                    I[idx1, idx2] = (exp_eigvals_list[step,idx1]-exp_eigvals_list[step,idx2])/(eigvals_list[step,idx1]-eigvals_list[step,idx2])
                    I[idx2, idx1] = I[idx1, idx2]
            
            for control in range(0, M):
                dH = Hc[control]
                #pdb.set_trace()
                dH = np.matmul(Rd,np.matmul(dH,R))
                dH = dH*I
                dH = np.matmul(R,np.matmul(dH,Rd))
                
                dpsi = np.vdot(psi_backward[step], np.dot(dH, psi_forward[step]))
                
                gradient[control, step] = -2*np.real(dpsi*psi_T)
                
        return infidelity, gradient
        














            
