import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

########################
# Helper for saving figs
########################
plot_id = 1
def savefig():
    global plot_id
    fname = f"plot_{plot_id:02d}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    plot_id += 1


n_grid=200
x=np.linspace(-5,5,n_grid)
y=np.sin(x)

plt.plot(y)
savefig()


h=x[1]-x[0]
D=-np.eye(n_grid)+np.diagflat(np.ones(n_grid-1),1)
D = D / h

D2=D.dot(-D.T)
D2[-1,-1]=D2[0,0]

sns.set_style("white")
plt.plot(x,y, label="f")
plt.plot(x[:-1],D.dot(y)[:-1], label="D")
plt.plot(x[1:-1],D2.dot(y)[1:-1], label="D2")
plt.legend()
savefig()


eig_non, psi_non=np.linalg.eigh(-D2/2)

for i in range(5):
    plt.plot(x,psi_non[:,i], label=f"{eig_non[i]:.4f}")
plt.legend(loc=1)
savefig()


X=np.diagflat(x*x)

eig_harm, psi_harm = np.linalg.eigh(-D2/2+X)

for i in range(5):
    plt.plot(x,psi_harm[:,i], label=f"{eig_harm[i]:.4f}")
plt.legend(loc=1)
savefig()


w=np.full_like(x,1e10)
w[np.logical_and(x>-2,x<2)]=0.
plt.plot(w)
savefig()

eig_well, psi_well= np.linalg.eigh(-D2/2+np.diagflat(w))

for i in range(5):
    plt.plot(x,psi_well[:,i], label=f"{eig_well[i]:.4f}")
plt.legend(loc=1)
savefig()


###############
# integral
###############
def integral(x,y,axis=0):
    dx=x[1]-x[0]
    return np.sum(y*dx, axis=axis)

num_electron=17

def get_nx(num_electron, psi, x):
    # normalization
    I=integral(x,psi**2,axis=0)
    normed_psi=psi/np.sqrt(I)[None, :]
    
    # occupation num
    fn=[2 for _ in range(num_electron//2)]
    if num_electron % 2:
        fn.append(1)

    # density
    res=np.zeros_like(normed_psi[:,0])
    for ne, psi  in zip(fn,normed_psi.T):
        res += ne*(psi**2)
    return res


plt.plot(get_nx(num_electron,psi_non, x), label="non")
plt.plot(get_nx(num_electron,psi_harm, x), label="harm")
plt.plot(get_nx(num_electron,psi_well, x), label="well")
plt.legend(loc=1)
savefig()


###############
# XC + Hartree
###############
def get_exchange(nx,x):
    energy=-3./4.*(3./np.pi)**(1./3.)*integral(x,nx**(4./3.))
    potential=-(3./np.pi)**(1./3.)*nx**(1./3.)
    return energy, potential

def get_hatree(nx,x, eps=1e-1):
    h=x[1]-x[0]
    energy=np.sum(nx[None,:]*nx[:,None]*h**2/np.sqrt((x[None,:]-x[:,None])**2+eps)/2)
    potential=np.sum(nx[None,:]*h/np.sqrt((x[None,:]-x[:,None])**2+eps),axis=-1)
    return energy, potential

def print_log(i,log):
    print(f"step: {i:<5} energy: {log['energy'][-1]:<10.4f} energy_diff: {log['energy_diff'][-1]:.10f}")


##########################################
# Self-consistent field loop
##########################################
max_iter=1000
energy_tolerance=1e-5
log={"energy":[float("inf")], "energy_diff":[float("inf")]}

nx=np.zeros(n_grid)
for i in range(max_iter):
    ex_energy, ex_potential=get_exchange(nx,x)
    ha_energy, ha_potential=get_hatree(nx,x)
    
    # Hamiltonian
    H=-D2/2+np.diagflat(ex_potential+ha_potential+x*x)
    
    energy, psi= np.linalg.eigh(H)
    
    # log
    log["energy"].append(energy[0])
    energy_diff=energy[0]-log["energy"][-2]
    log["energy_diff"].append(energy_diff)
    print_log(i,log)
    
    # convergence
    if abs(energy_diff) < energy_tolerance:
        print("converged!")
        break
    
    # update density
    nx=get_nx(num_electron,psi,x)
else:
    print("not converged")


# Plot SCF eigenstates
for i in range(5):
    plt.plot(x,psi[:,i], label=f"{energy[i]:.4f}")
plt.legend(loc=1)
savefig()


plt.plot(nx)
plt.plot(get_nx(num_electron,psi_harm,x), label="no-interaction")
plt.legend()
savefig()
