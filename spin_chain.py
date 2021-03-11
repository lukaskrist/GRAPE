import numpy as np
import pdb
# Here I make a class that initialized a qubit chain

class Spin_chain:

    def __init__(self, N):
        self.N = N
        self.dim = 2**N
        self.initialize_basis()

        self.Sx = np.array([[0,1],[1,0]],dtype = complex)
        self.Sy = np.array([[0,-1j],[1j,0]],dtype = complex)
        self.Sz = np.array([[1,0],[0,-1]],dtype = complex)
        self.id = np.eye(2)

        self.calculate_magnetization_operator()
        self.initialize_Hamiltonians()


    def initialize_basis(self):
        self.basis = np.zeros((self.dim, self.N), dtype = int)
        
        for idx in range(1, self.dim):
            psi = np.copy(self.basis[idx-1])
            for idx2 in range(self.N-1,-1, -1):

                if psi[idx2] == 0:
                    psi[idx2] = 1
                    break
                else:
                    psi[idx2] = 0
            
            self.basis[idx] = psi

    def calculate_magnetization_operator(self):
        M = np.zeros((self.dim, self.dim))
        for idx in range(0, self.dim):
            M[idx, idx] = 2*(np.sum(self.basis[idx]))-self.N

        self.M = -M #|0> is spin up

        number_of_projectors = self.N+1
        self.projectors = np.zeros((number_of_projectors, self.dim, self.dim))
        self.proj_vals = -np.arange(-self.N, self.N+2, 2)
        it = 0

        for idx in range(0, number_of_projectors):
            
            proj_val = self.proj_vals[it]
            M_diag = np.diag(self.M)
            self.projectors[idx] = np.diag( proj_val == M_diag )
            it += 1

    
    def initialize_Hamiltonians(self):
        self.initialize_ZZ()
        self.initialize_X()
        self.initialize_Y()
        self.initialize_XX()
        self.initialize_ZZ_nextnearest()


    def initialize_ZZ(self):
        # ZZ terms is sum_j sigma_z(j) sigma_z(j+1) with periodic bounderies
        self.ZZ = np.zeros((self.dim, self.dim), dtype = complex)
        
        for idx in range(0, self.dim):
            for idx2 in range(0,self.N-1):
                if self.basis[idx,idx2] == self.basis[idx,idx2+1]:
                    self.ZZ[idx,idx] += 1
                else:
                    self.ZZ[idx,idx] -= 1

            # assuming periodic boundery conditions
            if self.basis[idx, 0 ] == self.basis[idx,self.N-1]:
                self.ZZ[idx,idx] +=1
            else:
                self.ZZ[idx,idx] -= 1

    def initialize_ZZ_nextnearest(self):
        # ZZ terms is sum_j sigma_z(j) sigma_z(j+2) with periodic bounderies
        self.ZZ_nextnearest = np.zeros((self.dim, self.dim), dtype = complex)
        
        for idx in range(0, self.dim):
            for idx2 in range(0,self.N):
                
                if self.basis[idx,idx2] == self.basis[idx, np.mod(idx2+2,self.N)]:
                    self.ZZ_nextnearest[idx,idx] += 1
                else:
                    self.ZZ_nextnearest[idx,idx] -= 1


    def initialize_X(self):
        # X is just sum_j sigma_x(j)
        X = np.zeros((self.dim, self.dim), dtype = complex)

        for idx1 in range(0, self.dim):
            for idx2 in range(0, self.N):
                psi = np.copy(self.basis[idx1])
                # apply x gate on idx2 qubit
                psi[idx2] = np.mod(psi[idx2] + 1, 2)

                indx = np.where((psi == self.basis).all(axis = 1))[0][0]
                X[idx1, indx] += 1.0

        self.X = X
        
        
    
    def initialize_Y(self):
        # Y is just sum_j sigma_y(j)
        Y = np.zeros((self.dim, self.dim), dtype = complex)

        for idx1 in range(0, self.dim):
            for idx2 in range(0, self.N):
                psi = np.copy(self.basis[idx1])
                # apply x gate on idx2 qubit
                if psi[idx2] == 0:
                    a = -1j
                else:
                    a = 1j
                        
                    
                psi[idx2] = np.mod(psi[idx2] + 1, 2)

                indx = np.where((psi == self.basis).all(axis = 1))[0][0]
                Y[idx1, indx] += a

        self.Y = Y
    
    def initialize_XX(self):
        # XX terms is  sigma_z(j) sigma_z(j+1) with no periodic bounderies
        XX = np.zeros((self.N-1, self.dim, self.dim), dtype = complex)
        
        
        for step in range(0, self.N-1):
            # calculate the interaction between step and step + 1
            for idx in range(0, self.dim):
                psi = np.copy(self.basis[idx])
                psi[step] = np.mod(psi[step] + 1, 2)
                psi[step + 1] = np.mod(psi[step + 1] + 1, 2)
                indx = np.where((psi == self.basis).all(axis = 1))[0][0]
                XX[step, indx, idx] += 1.0
        self.XX = XX
        
        
    
    def measure(self, psi):
        # simulate a measurement of total magnetization
        measure_probs = np.zeros((self.proj_vals.shape))

        for idx in range(0, measure_probs.shape[0]):
            p = np.real( np.vdot(psi, np.dot(self.projectors[idx],psi) ) )
            measure_probs[idx] = p

        measure_indx = np.random.choice(np.arange(0, self.proj_vals.shape[0]), size = 1, p = measure_probs)[0]

        measure_result = self.proj_vals[measure_indx] # value of measurement

        new_psi = np.dot(self.projectors[measure_indx], psi)

        new_psi /= np.real( np.sqrt(np.vdot(psi,new_psi)) )


        return measure_result, measure_probs, new_psi


        
    def initialize_memory(self, Nsteps, Nsize):
        # Here Nsize denotes the maximum size of allocated space
        self.pulses = np.zeros((Nsize, Nsteps))
        self.infidelities = np.zeros((Nsize))
        self.Ncount = 0
        
    def store_memory(self, pulse, inf):
        self.pulses[self.Ncount] = pulse
        self.infidelities[self.Ncount] = inf
        self.Ncount += 1
