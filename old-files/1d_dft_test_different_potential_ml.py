import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("white")

# Create output directory
os.makedirs("results_potentials", exist_ok=True)

###############################################################################
# Grid + Derivative matrices
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
# Utility functions
###############################################################################
def integral(x, y, axis=0):
    dx = x[1] - x[0]
    return np.sum(y * dx, axis=axis)

def get_nx(num_electron, psi, x):
    """Compute electron density from orbitals"""
    I = integral(x, psi**2, axis=0)
    psi_norm = psi / np.sqrt(I)[None, :]
    
    occupations = [2]*(num_electron//2)
    if num_electron % 2:
        occupations.append(1)
    
    rho = np.zeros_like(psi[:, 0])
    for occ, orb in zip(occupations, psi_norm.T):
        rho += occ * orb**2
    return rho

def get_exchange(nx, x):
    """Exchange energy and potential (LDA)"""
    nx_safe = np.maximum(nx, 1e-12)  # Prevent negative densities
    energy = -3/4 * (3/np.pi)**(1/3) * integral(x, nx_safe**(4/3))
    potential = -(3/np.pi)**(1/3) * nx_safe**(1/3)
    return energy, potential

def get_hartree(nx, x, eps=1e-1):
    """Hartree energy and potential"""
    h = x[1] - x[0]
    diff = x[:, None] - x[None, :]
    R = np.sqrt(diff**2 + eps)
    energy = np.sum(nx[:, None]*nx[None, :]*h*h / R) / 2
    potential = np.sum(nx[None, :] * h / R, axis=-1)
    return energy, potential


###############################################################################
# Define different potential types
###############################################################################
def get_potential(potential_type, x):
    """
    Returns V_confining (for eigenvalue problems) and V_external (for SCF)
    
    For wells: Use large barriers for confinement, but zeros for SCF external potential
    """
    if potential_type == "harmonic":
        V_confining = x * x
        V_external = x * x
        
    elif potential_type == "infinite_well":
        # Infinite well from -2 to 2
        V_confining = np.full_like(x, 1e10)
        V_confining[(x > -2) & (x < 2)] = 0.0
        V_external = np.zeros_like(x)  # External potential is zero inside well
        
    elif potential_type == "finite_well":
        # Finite square well: depth = -5, width from -2 to 2
        V_confining = np.zeros_like(x)
        V_confining[(x >= -2) & (x <= 2)] = -5.0
        V_external = V_confining.copy()
        
    elif potential_type == "step":
        # Step potential: 0 for x < 0, height 3 for x >= 0
        V_confining = np.zeros_like(x)
        V_confining[x >= 0] = 3.0
        V_external = V_confining.copy()
        
    elif potential_type == "double_well":
        # Double well potential
        V_confining = 0.5 * x**4 - 4 * x**2
        V_external = V_confining.copy()
        
    elif potential_type == "asymmetric":
        # Asymmetric potential
        V_confining = 0.5 * x**2 + 0.3 * x**3
        V_external = V_confining.copy()
        
    else:
        raise ValueError(f"Unknown potential type: {potential_type}")
    
    return V_confining, V_external


###############################################################################
# Machine Learning: Train on multiple potentials
###############################################################################
def train_ml_model(x, D2, num_electron, n_training=20):
    """
    Train a linear model to predict density from potential
    Uses multiple random harmonic + noise potentials
    """
    print("\n" + "="*70)
    print("TRAINING ML MODEL")
    print("="*70)
    
    potentials = []
    densities = []
    
    def solve_density_for_potential(V_conf):
        H = -D2/2 + np.diagflat(V_conf)
        eig, psi = np.linalg.eigh(H)
        return get_nx(num_electron, psi, x)
    
    # Generate training data: harmonic + noise
    np.random.seed(42)
    for i in range(n_training):
        V_train = 0.5*x*x + 0.3*np.random.randn(len(x))
        rho = solve_density_for_potential(V_train)
        potentials.append(V_train)
        densities.append(rho)
    
    X_train = np.stack(potentials)   # (n_training, n_grid)
    Y_train = np.stack(densities)    # (n_training, n_grid)
    
    # Solve for weight matrix using SVD
    U, S, VT = np.linalg.svd(X_train, full_matrices=False)
    W = VT.T @ np.diag(1/S) @ U.T @ Y_train  # (n_grid, n_grid)
    
    print(f"Trained on {n_training} examples")
    print(f"Weight matrix shape: {W.shape}")
    
    return W


###############################################################################
# SCF iteration (single run)
###############################################################################
def run_scf(x, D2, V_external, num_electron, initial_density, 
            max_iter=1000, tol=1e-5, verbose=True, label=""):
    """
    Run self-consistent field iteration
    
    Returns: (converged, num_iterations, final_energy, energy_history)
    """
    nx = initial_density.copy()
    energy_history = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCF: {label}")
        print(f"{'='*70}")
    
    for i in range(max_iter):
        # Ensure non-negative density
        nx = np.maximum(nx, 1e-12)
        
        # Compute potentials
        ex_energy, ex_pot = get_exchange(nx, x)
        ha_energy, ha_pot = get_hartree(nx, x)
        
        # Build Hamiltonian
        H = -D2/2 + np.diagflat(ex_pot + ha_pot + V_external)
        energy, psi = np.linalg.eigh(H)
        
        energy_history.append(energy[0])
        
        # Check convergence
        if i > 0:
            dE = energy[0] - energy_history[-2]
            if verbose and i % 10 == 0:
                print(f"Step {i:4d}  E = {energy[0]:12.8f}  dE = {dE:+.4e}")
            
            if abs(dE) < tol:
                if verbose:
                    print(f"Step {i:4d}  E = {energy[0]:12.8f}  dE = {dE:+.4e}")
                    print(f"✓ Converged in {i+1} iterations!")
                return True, i+1, energy[0], energy_history
        
        # Update density
        nx = get_nx(num_electron, psi, x)
    
    if verbose:
        print(f"✗ Did NOT converge after {max_iter} iterations")
    return False, max_iter, energy_history[-1], energy_history


###############################################################################
# Main comparison function
###############################################################################
def compare_potentials(potential_types, num_electron=17, use_ml=True):
    """
    Compare SCF convergence for different potentials with/without ML
    """
    results = {}
    
    # Train ML model once
    if use_ml:
        W = train_ml_model(x, D2, num_electron, n_training=20)
    
    for pot_type in potential_types:
        print("\n" + "="*70)
        print(f"TESTING POTENTIAL: {pot_type.upper()}")
        print("="*70)
        
        V_confining, V_external = get_potential(pot_type, x)
        
        # Plot the potential
        plt.figure(figsize=(10, 4))
        plt.plot(x, V_confining, 'b-', linewidth=2, label='Confining potential')
        if not np.allclose(V_confining, V_external):
            plt.plot(x, V_external, 'r--', linewidth=2, label='External potential (SCF)')
        plt.xlabel('x')
        plt.ylabel('V(x)')
        plt.title(f'Potential: {pot_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([-10, 20])
        plt.savefig(f"results_potentials/{pot_type}_potential.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Initial guess 1: Zero density
        nx_zero = np.zeros_like(x)
        converged_zero, iter_zero, E_zero, hist_zero = run_scf(
            x, D2, V_external, num_electron, nx_zero,
            label=f"{pot_type} - Zero initial guess", verbose=False
        )
        
        # Initial guess 2: ML prediction (if enabled)
        if use_ml:
            nx_ml = V_external @ W
            # Ensure non-negative and normalize
            nx_ml = np.maximum(nx_ml, 0)
            current_integral = integral(x, nx_ml)
            if current_integral > 0:
                nx_ml *= (num_electron / current_integral)
            else:
                nx_ml = np.ones_like(x) * num_electron / (x[-1] - x[0])
            
            converged_ml, iter_ml, E_ml, hist_ml = run_scf(
                x, D2, V_external, num_electron, nx_ml,
                label=f"{pot_type} - ML initial guess", verbose=False
            )
        else:
            converged_ml, iter_ml, E_ml, hist_ml = False, 0, 0, []
        
        # Store results
        results[pot_type] = {
            'zero_converged': converged_zero,
            'zero_iterations': iter_zero,
            'zero_energy': E_zero,
            'zero_history': hist_zero,
            'ml_converged': converged_ml,
            'ml_iterations': iter_ml,
            'ml_energy': E_ml,
            'ml_history': hist_ml,
        }
        
        # Print summary
        print(f"\n{'─'*70}")
        print(f"RESULTS for {pot_type}:")
        print(f"  Zero initial guess: {iter_zero} iterations, E = {E_zero:.8f}")
        if use_ml:
            print(f"  ML initial guess:   {iter_ml} iterations, E = {E_ml:.8f}")
            if converged_zero and converged_ml:
                savings = (iter_zero - iter_ml) / iter_zero * 100
                print(f"  Savings: {iter_zero - iter_ml} iterations ({savings:.1f}%)")
        print(f"{'─'*70}")
        
        # Plot convergence comparison
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(len(hist_zero)), 
                     np.abs(np.array(hist_zero) - E_zero), 
                     'b-o', linewidth=2, markersize=4, label='Zero initial guess')
        if use_ml:
            plt.semilogy(range(len(hist_ml)), 
                        np.abs(np.array(hist_ml) - E_ml), 
                        'r-s', linewidth=2, markersize=4, label='ML initial guess')
        plt.xlabel('Iteration')
        plt.ylabel('|E - E_final|')
        plt.title(f'Convergence: {pot_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"results_potentials/{pot_type}_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results


###############################################################################
# Summary visualization
###############################################################################
def plot_summary(results):
    """Create summary bar chart comparing all potentials"""
    pot_types = list(results.keys())
    
    zero_iters = [results[p]['zero_iterations'] for p in pot_types]
    ml_iters = [results[p]['ml_iterations'] for p in pot_types]
    savings_pct = [(z - m)/z * 100 if z > 0 else 0 
                   for z, m in zip(zero_iters, ml_iters)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Iterations comparison
    x_pos = np.arange(len(pot_types))
    width = 0.35
    
    ax1.bar(x_pos - width/2, zero_iters, width, label='Zero initial', color='skyblue')
    ax1.bar(x_pos + width/2, ml_iters, width, label='ML initial', color='salmon')
    ax1.set_xlabel('Potential Type')
    ax1.set_ylabel('Iterations to Converge')
    ax1.set_title('SCF Convergence Speed')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(pot_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Savings percentage
    colors = ['green' if s > 0 else 'red' for s in savings_pct]
    ax2.bar(x_pos, savings_pct, color=colors, alpha=0.7)
    ax2.set_xlabel('Potential Type')
    ax2.set_ylabel('Iteration Savings (%)')
    ax2.set_title('ML Performance Improvement')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pot_types, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(savings_pct):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results_potentials/summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Potential':<20} {'Zero':<10} {'ML':<10} {'Savings':<15}")
    print("─"*70)
    for i, pot in enumerate(pot_types):
        print(f"{pot:<20} {zero_iters[i]:<10} {ml_iters[i]:<10} "
              f"{zero_iters[i]-ml_iters[i]:>5} ({savings_pct[i]:>5.1f}%)")
    print("="*70)


###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    # Configuration
    num_electron = 17  # Can increase to 50, 100 for harder problems
    
    # List of potentials to test
    potential_types = [
        "harmonic",
        "infinite_well",
        "finite_well",
        "step",
        "double_well",
        "asymmetric"
    ]
    
    print("="*70)
    print("TESTING DIFFERENT POTENTIAL TYPES WITH ML")
    print(f"Number of electrons: {num_electron}")
    print(f"Grid points: {n_grid}")
    print("="*70)
    
    # Run comparison
    results = compare_potentials(potential_types, num_electron=num_electron, use_ml=True)
    
    # Create summary visualization
    plot_summary(results)
    
    print("\n✓ All results saved to 'results_potentials/' directory")