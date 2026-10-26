import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


E      = 18000e6     # instantaneous Young's modulus, Pa (170 MPa ~ paper's bone)
E_pag  = 0.86*E
phi    = 0.94      # E_inf / E_0, fraction of stiffness that does not relax
tau    = 5.5       # relaxation time, s
strain0 = 0.0022   # fixed compressive strain (2200 microstrain)

L = 5e-3                   # m
A = np.pi*(5e-3)**2        # m^2

k1 = phi*E*A/L
k2 = (1-phi)*E*A/L
c  = tau*k2

############  PAGETS ----------------

k1_pag = phi*E_pag*A/L
k2_pag = (1-phi)*E_pag*A/L
c_pag  = tau*k2_pag
x0_pag = strain0*L             # displacement, m

#force_pag  = x0_pag*(k1_pag + k2_pag*np.exp(-k2_pag*times/c_pag))
#stress_pag = force_pag/A           # Pa

############  PAGETS  -----------------



m = 500000000

def Amat(k1, k2, c, m):
    return np.array([[0, 1, 0],
                     [-(k1+k2)/m, 0, k2/m],
                     [k2/c, 0, -k2/c]])

def Bmat(k1_pag, k2_pag, c_pag, m):
    return np.array([[0, 1, 0],
                     [-(k1_pag+k2_pag)/m, 0, k2_pag/m],
                     [k2_pag/c_pag, 0, -k2_pag/c_pag]])

v0 = 1 # initial velocity (downwards)
A = Amat(k1, k2, c, m) # choose k1, k2 and c here
print(A)

v0 = 1 # initial velocity (downwards)
B = Bmat(k1_pag, k2_pag, c_pag, m) # choose k1, k2 and c here
print(B)

eig = np.linalg.eig(A)
ev = eig.eigenvalues
print(ev) # should all have -ve real part (for stability)

eig_pag = np.linalg.eig(B)
ev_pag = eig_pag.eigenvalues
print(ev_pag) # should all have -ve real part (for stability)

# solution (x vector)
def xv(t):
    return sp.linalg.expm(A*t) @ np.array([0, v0, 0])

def xv_pag(t):
    return sp.linalg.expm(B*t) @ np.array([0, v0, 0])

# compute and plot solution at a bunch of times
times = np.arange(0, 60, 0.1)
out = np.array(list(map(xv, times)))
print(out.shape)

out_pag = np.array(list(map(xv_pag, times)))
print(out_pag.shape)

#for i in range(3):
plt.plot(times, out[:,0], label = 'Healthy Bone')
plt.plot(times, out_pag[:,0], label = 'Pagets Bone')
plt.title(f'Cortical Bone Impact Responce')
plt.legend()
plt.xlabel('Time (t)')
plt.show()
