import numpy as np
import matplotlib.pyplot as plt

E      = 170e6    # instantaneous Young's modulus, Pa (170 MPa ~ paper's bone)
E_pag  = 0.86*E
phi    = 0.75      # E_inf / E_0, fraction of stiffness that does not relax
tau    = 2.5       # relaxation time, s
strain0 = 0.0022   # fixed compressive strain (2200 microstrain)

L = 5e-3                   # m
A = np.pi*(5e-3)**2        # m^2

k1 = phi*E*A/L
k2 = (1-phi)*E*A/L
c  = tau*k2
x0 = strain0*L             # displacement, m

times  = np.arange(0, 60, 0.1)
force  = x0*(k1 + k2*np.exp(-k2*times/c))
stress = force/A           # Pa

####PAGETS-----------------------

k1_pag = phi*E_pag*A/L
k2_pag = (1-phi)*E_pag*A/L
c_pag  = tau*k2_pag
x0_pag = strain0*L             # displacement, m

force_pag  = x0_pag*(k1_pag + k2_pag*np.exp(-k2_pag*times/c))
stress_pag = force_pag/A           # Pa

####PAGETS---------------------------

plt.plot(times, stress/1e6, label='Healthy')
plt.plot(times, stress_pag/1e6, label='Pagets')
plt.xlabel('Time (s)')
plt.ylabel('Stress (MPa)')
plt.title(f'Healthy vs Pagets Trabecular Bone')
plt.legend()
plt.show()
