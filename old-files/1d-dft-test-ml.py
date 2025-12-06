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


###############################################################################
# 1. Grid + Derivative matrices
###############################################################################
n_grid = 200
x = np.linspace(-5, 5, n_grid)
h = x[1] - x[0]

# First derivative
D = -np.eye(n_grid) + np.diagflat(np.ones(n_grid-1), 1)
D = D / h

# Second derivative operator
D2 = D.dot(-D.T)
D2[-1, -1] = D2[0, 0]


###############################################################################
# 2. Basic tests of D, D2 (saved plots)
###############################################################################
y = np.sin(x)
plt.plot(y)
savefig()

plt.plot(x, y, label="f")
plt.plot(x[:-1], D.dot(y)[:-1], label="D[f]")
plt.plot(x[1:-1], D2.dot(y)[1:-1], label="D2[f]")
plt.legend()
savefig()


###############################################################################
# 3. Solve simple Hamiltonians
###############################################################################
eig_non, psi_non = np.linalg.eigh(-D2/2)

plt.figure()
for i in range(5):
    plt.plot(x, psi_non[:, i], label=f"{eig_non[i]:.4f}")
plt.legend()
savefig()

Xharm = np.diagflat(x*x)
eig_harm, psi_harm = np.linalg.eigh(-D2/2 + Xharm)

plt.figure()
for i in range(5):
    plt.plot(x, psi_harm[:, i], label=f"{eig_harm[i]:.4f}")
plt.legend()
savefig()

w = np.full_like(x, 1e10)
w[(x > -2) & (x < 2)] = 0
plt.plot(w)
savefig()

eig_well, psi_well = np.linalg.eigh(-D2/2 + np.diagflat(w))

plt.figure()
for i in range(5):
    plt.plot(x, psi_well[:, i], label=f"{eig_well[i]:.4f}")
plt.legend()
savefig()


###############################################################################
# 4. Density routines
###############################################################################
def integral(x, y, axis=0):
    dx = x[1] - x[0]
    return np.sum(y * dx, axis=axis)

num_electron = 17

def get_nx(num_electron, psi, x):
    I = integral(x, psi**2, axis=0)
    psi_norm = psi / np.sqrt(I)[None, :]

    occupations = [2]*(num_electron//2)
    if num_electron % 2:
        occupations.append(1)

    rho = np.zeros_like(psi[:, 0])
    for occ, orb in zip(occupations, psi_norm.T):
        rho += occ * orb**2
    return rho


plt.figure()
plt.plot(get_nx(num_electron, psi_non, x), label="non")
plt.plot(get_nx(num_electron, psi_harm, x), label="harm")
plt.plot(get_nx(num_electron, psi_well, x), label="well")
plt.legend()
savefig()


###############################################################################
# 5. Exchange + Hartree
###############################################################################
def get_exchange(nx, x):
    energy = -3/4 * (3/np.pi)**(1/3) * integral(x, nx**(4/3))
    potential = -(3/np.pi)**(1/3) * nx**(1/3)
    return energy, potential

def get_hartree(nx, x, eps=1e-1):
    h = x[1] - x[0]
    diff = x[:, None] - x[None, :]
    energy = np.sum(nx[:, None]*nx[None, :]*h*h / np.sqrt(diff**2 + eps)) / 2
    potential = np.sum(nx[None, :] * h / np.sqrt(diff**2 + eps), axis=-1)
    return energy, potential


###############################################################################
# 6. MACHINE LEARNING: Train linear model ρ ≈ Xw
###############################################################################
# Build training dataset of potentials
potentials = []
densities = []

def solve_density_for_potential(V):
    H = -D2/2 + np.diagflat(V)
    eig, psi = np.linalg.eigh(H)
    return get_nx(num_electron, psi, x)

# Make several random training potentials
np.random.seed(0)
for _ in range(20):
    V = 0.5*x*x + 0.2*np.random.randn(len(x))         # harmonic + noise
    rho = solve_density_for_potential(V)
    potentials.append(V)
    densities.append(rho)

# Stack into matrices
X_train = np.stack(potentials)       # shape (N, n_grid)
Y_train = np.stack(densities)        # shape (N, n_grid)

# Linear model for each grid point: Y = XW
# Solve W = pseudo-inverse(X) @ Y (using SVD)
U, S, VT = np.linalg.svd(X_train, full_matrices=False)
W = VT.T @ np.diag(1/S) @ U.T @ Y_train   # shape (n_grid, n_grid)


###############################################################################
# 7. Self-consistent field using ML initial guess
###############################################################################
# Input potential for main SCF
V_SHO = x*x 
V_linear = x
V_inf = np.zeros(n_grid) #inf square 
V_main = V_linear

print("V_main=\n",V_main)

# ML initial density guess:
nx = V_main @ W     # shape (n_grid,)

print("ML initial density constructed!")

max_iter = 1000
energy_tolerance = 1e-5
log = {"energy": [float("inf")], "energy_diff": [float("inf")]}

for i in range(max_iter):
    ex_energy, ex_pot = get_exchange(nx, x)
    ha_energy, ha_pot = get_hartree(nx, x)

    H = -D2/2 + np.diagflat(ex_pot + ha_pot + V_main)
    energy, psi = np.linalg.eigh(H)

    log["energy"].append(energy[0])
    diff = energy[0] - log["energy"][-2]
    log["energy_diff"].append(diff)
    print(f"Step {i:4d}  E = {energy[0]:.6f}  dE = {diff:.4e}")

    if abs(diff) < energy_tolerance:
        print("Converged!")
        break

    nx = get_nx(num_electron, psi, x)
else:
    print("NOT converged!")


###############################################################################
# 8. Plots of final solution
###############################################################################
plt.figure()
for i in range(5):
    plt.plot(x, psi[:, i], label=f"{energy[i]:.4f}")
plt.legend()
savefig()

plt.figure()
plt.plot(nx, label="interacting density")
plt.plot(get_nx(num_electron, psi_harm, x), label="non-interacting")
plt.legend()
savefig()
