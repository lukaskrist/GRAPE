import matplotlib.pyplot as plt
import numpy as np
import GRAPE
import scipy.linalg as la
import pdb
import matplotlib as mpl
#from pylab import *
#from qutip import *
from matplotlib import cm
import imageio
'''
def animate_bloch(states,number, duration=0.1, save_all=True):

    b = Bloch()
    b.vector_color = ['r']
    b.view = [-40,30]
    images=[]
    try:
        length = len(states)
    except:
        length = 1
        states = [states]
    ## normalize colors to the length of data ##
    nrm = mpl.colors.Normalize(0,length)
    colors = cm.cool(nrm(range(length))) # options: cool, summer, winter, autumn etc.

    ## customize sphere properties ##
    b.point_color = list(colors) # options: 'r', 'g', 'b' etc.
    b.point_marker = ['o']
    b.point_size = [30]
    
    for i in range(length):
        b.clear()
        b.add_states(states[i])
        b.add_states(states[:(i+1)],'point')
        if save_all:
            b.save(dirc='tmp') #saving images to tmp directory
            filename="tmp/bloch_%01d.png" % i
        else:
            filename='temp_file.png'
            b.save(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave('bloch_anim%s.gif' %number, images, duration=duration)
'''
G = GRAPE.GRAPE()

N = 50
T = 3.5

pulse = np.random.rand(1,N)
#print(pulse)

H0 = np.array([[1,0],[0,-1]], dtype = complex)
Hc = np.array([[[0,1],[1,0]]], dtype = complex)

psi0 = np.array([1,0], dtype = complex)
psi_target = np.array([0,1j], dtype = complex)

#psi_target = 1j*np.dot(H0+Hc[0],psi0)

psi_target = np.dot(la.expm(-1j*(H0 + Hc[0])*T), psi0)

inf, g, eigval  = G.infidelity(pulse, N, T, H0, Hc, psi0, psi_target)

#pdb.set_trace()
# Test the gradient

coeffs = np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60])
weights = np.array([-3,-2,-1,0,1,2,3])

iterations = 100

N_exp = np.linspace(-0.5,-4,iterations)
eps_list = np.zeros((iterations))
error_list = np.zeros((iterations))
psi_new = []
inf_story = []
for iteration in range(0, iterations):
    eps = 10**N_exp[iteration]
    
    g_new = np.zeros((1,N))
    
    for step in range(0, N):
        for idx in range(0, coeffs.shape[0]):
            new_pulse = np.copy(pulse)
            new_pulse[0,step] += weights[idx]*eps
            #print(new_pulse)
            inf_new, grad ,psi_every = G.infidelity(new_pulse, N, T, H0, Hc, psi0, psi_target)
            inf_story.append(inf_new)
            g_new[0,step] += coeffs[idx]*inf_new/eps
            psi_new.append(psi_every)
            
    if iteration < 5:   
        
        psi_animate = []
        for step in psi_every:
            psi_animate.append(Qobj(step))
    
        animate_bloch(psi_animate,iteration)
    #print(grad)
    error = np.sqrt(np.sum(np.abs(g_new-g)**2))
    error_list[iteration] = error
    eps_list[iteration] = eps
print(inf_story)

plt.loglog(eps_list, error_list)
plt.show()




#