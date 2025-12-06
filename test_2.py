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
def get_potential(potential_type, x, params=None):
    """
    Returns V_confining (for eigenvalue problems) and V_external (for SCF).
    Accepts optional 'params' dictionary to allow varying potentials for ML training.
    """
    # Default parameters if none provided
    if params is None:
        params = {}

    if potential_type == "harmonic":
        # k = m * omega^2
        # Default k=1.0 (omega=1.0, m=1.0)
        k = params.get('k', 1.0) 
        V_confining = 0.5 * k * x**2
        V_external = V_confining.copy()
        
    elif potential_type == "infinite_well":
        # Infinite well usually defined by boundaries
        width = params.get('width', 10.0) # default spans full -5 to 5
        center = params.get('center', 0.0)
        
        # Calculate left and right walls
        a = center - width/2
        b = center + width/2
        
        large_value = 1e12
        V_confining = np.zeros_like(x)
        V_confining[x < a] = large_value
        V_confining[x > b] = large_value
        
        # Add small random bumps inside well if requested (for ML diversity)
        if 'noise_amp' in params:
            V_confining += params['noise_amp'] * np.random.randn(len(x))
            # Keep walls infinite
            V_confining[x < a] = large_value
            V_confining[x > b] = large_value
            
        V_external = np.zeros_like(x) 
        # External potential for SCF is usually the physical potential inside the box
        # We can map the internal bumps to external potential
        mask_inside = (x >= a) & (x <= b)
        V_external[mask_inside] = V_confining[mask_inside]
        
    elif potential_type == "finite_well":
        # Finite square well
        depth = params.get('depth', -1.0)
        width = params.get('width', 6.0) # Default -3 to 3
        center = params.get('center', 0.0)
        
        a = center - width/2
        b = center + width/2
        y_level = 0
        
        V_confining = np.full_like(x, y_level)
        mask = (x > a) & (x < b)
        V_confining[mask] = depth
        
        # Add noise if requested
        if 'noise_amp' in params:
            V_confining += params['noise_amp'] * np.random.randn(len(x))
            
        V_external = V_confining.copy()
        
    elif potential_type == "step":
        # Step potential
        x_jump = params.get('x_jump', 0.0)
        y_lower = 0
        y_upper = params.get('height', 5.0)
        
        V_confining = np.where(x < x_jump, y_lower, y_upper)
        if 'noise_amp' in params:
            V_confining += params['noise_amp'] * np.random.randn(len(x))
            
        V_external = V_confining.copy()
        
    elif potential_type == "double_square_well":
        # Two separate wells
        depth_L = params.get('depth_L', -1.0)
        depth_R = params.get('depth_R', -1.0)
        
        # Fixed positions for simplicity, or could vary
        left_well_a = -4
        left_well_b = -1.5
        right_well_a = 1.5
        right_well_b = 4
        
        y_level = 0
        V_confining = np.full_like(x, y_level)
        
        V_confining[(x > left_well_a) & (x < left_well_b)] = depth_L
        V_confining[(x > right_well_a) & (x < right_well_b)] = depth_R
        
        if 'noise_amp' in params:
             V_confining += params['noise_amp'] * np.random.randn(len(x))
             
        V_external = V_confining.copy()
        
    else:
        raise ValueError(f"Unknown potential type: {potential_type}")
    
    return V_confining, V_external


###############################################################################
# Machine Learning: Train on multiple potentials
###############################################################################
def train_ml_model(x, D2, num_electron, potential_type, n_training=50):
    """
    Train a linear model to predict density from potential.
    
    UPDATED STRATEGY: 
    Instead of adding white noise to a static potential, we vary the 
    parameters (depth, width, k) of the potential. This teaches the ML
    model the physics of how the potential shape affects density.
    """
    print(f"  Training ML model for {potential_type} ({n_training} samples)...")
    
    potentials = []
    densities = []
    
    def solve_density_for_potential(V_conf):
        H = -D2/2 + np.diagflat(V_conf)
        # For training, we need stable eigenvalues. 
        # Sometimes random noise creates deep holes; we clip very large negative values
        V_conf = np.maximum(V_conf, -100) 
        eig, psi = np.linalg.eigh(H)
        return get_nx(num_electron, psi, x)
    
    np.random.seed(42)
    
    for i in range(n_training):
        params = {}
        
        # Vary parameters based on potential type
        if potential_type == "harmonic":
            # Vary spring constant k between 0.5 and 1.5
            params['k'] = 1.0 + 0.5 * (np.random.rand() - 0.5) 
            
        elif potential_type == "infinite_well":
            # Vary width slightly and add internal bumps
            params['noise_amp'] = 0.5 # Add bumps to the floor
            
        elif potential_type == "finite_well":
            # Vary depth significantly (-0.5 to -2.0)
            # This is crucial for ML to learn "deeper = more density"
            params['depth'] = -1.0 + 1.0 * (np.random.rand() - 0.5)
            # Vary width slightly
            params['width'] = 6.0 + 1.0 * (np.random.rand() - 0.5)
            
        elif potential_type == "step":
            # Vary step height
            params['height'] = 5.0 + 2.0 * (np.random.rand() - 0.5)
            # Vary jump position
            params['x_jump'] = 0.0 + 1.0 * (np.random.rand() - 0.5)
            
        elif potential_type == "double_square_well":
            # Vary depths of left and right wells independently
            params['depth_L'] = -1.0 + 0.8 * (np.random.rand() - 0.5)
            params['depth_R'] = -1.0 + 0.8 * (np.random.rand() - 0.5)
        
        # Generate potential with these parameters
        V_train, _ = get_potential(potential_type, x, params)
        
        # Add a tiny bit of white noise to everything to prevent overfitting to exact shapes
        V_train += 0.05 * np.random.randn(len(x))
        
        rho = solve_density_for_potential(V_train)
        
        potentials.append(V_train)
        densities.append(rho)
    
    X_train = np.stack(potentials)
    Y_train = np.stack(densities)
    
    # Solve for weight matrix using SVD
    # Regularization (rcond/limit) helps numerical stability
    U, S, VT = np.linalg.svd(X_train, full_matrices=False)
    
    # Filter small singular values (Tikhonov Regularization equivalent)
    limit = 1e-3
    S_inv = np.zeros_like(S)
    S_inv[S > limit] = 1.0 / S[S > limit]
    
    W = VT.T @ np.diag(S_inv) @ U.T @ Y_train
    
    return W


###############################################################################
# SCF iteration (single run)
###############################################################################
def run_scf(x, D2, V_external, num_electron, initial_density, 
            max_iter=1000, tol=1e-5, verbose=True, label=""):
    
    nx = initial_density.copy()
    energy_history = []
    
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
            if abs(dE) < tol:
                return True, i+1, energy[0], energy_history
        
        # Update density
        nx = get_nx(num_electron, psi, x)
    
    return False, max_iter, energy_history[-1], energy_history


###############################################################################
# Initial guess generators
###############################################################################
def get_initial_guess(guess_type, x, V_external, W, num_electron):
    
    if guess_type == "zero":
        nx = np.zeros_like(x)
        
    elif guess_type == "ones":
        nx = np.ones_like(x)
        
    elif guess_type == "random":
        # Random starting density
        nx = np.random.rand(len(x))
        
    elif guess_type == "gaussian":
        sigma = 1.0
        nx = np.exp(-x**2 / (2 * sigma**2))
        
    elif guess_type == "ml":
        nx = V_external @ W
        nx = np.maximum(nx, 0) # Enforce physical positivity
        
    else:
        raise ValueError(f"Unknown guess type: {guess_type}")
    
    # Normalize to correct number of electrons
    current_integral = integral(x, nx)
    if current_integral > 1e-10:
        nx *= (num_electron / current_integral)
    else:
        # Fallback to uniform if integral is essentially zero
        nx = np.ones_like(x) * num_electron / (x[-1] - x[0])
    
    return nx


###############################################################################
# Main comparison function
###############################################################################
def compare_potentials(potential_types, num_electron=17, use_ml=True):
    
    results = {}
    # Added "random" to the list of guesses
    guess_types = ["zero", "ones", "random", "gaussian", "ml"] if use_ml else ["zero", "ones", "random", "gaussian"]
    
    for pot_type in potential_types:
        print("\n" + "="*70)
        print(f"TESTING POTENTIAL: {pot_type.upper()}")
        print("="*70)
        
        # 1. Train ML Model specific to this potential type
        W = None
        if use_ml:
            W = train_ml_model(x, D2, num_electron, pot_type, n_training=50)
        
        # 2. Get the actual potential for SCF (Standard params)
        V_confining, V_external = get_potential(pot_type, x)
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(x, V_confining, 'b-', linewidth=2, label='Confining potential')
        plt.xlabel('x')
        plt.title(f'Potential: {pot_type}')
        plt.ylim([-10, 20])
        plt.savefig(f"results_potentials/{pot_type}_potential.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test each initial guess type
        results[pot_type] = {}
        
        for guess_type in guess_types:
            nx_init = get_initial_guess(guess_type, x, V_external, W, num_electron)
            
            converged, iterations, final_E, history = run_scf(
                x, D2, V_external, num_electron, nx_init,
                label=f"{pot_type} - {guess_type} initial", verbose=False
            )
            
            results[pot_type][guess_type] = {
                'converged': converged,
                'iterations': iterations,
                'energy': final_E,
                'history': history,
            }
            
            print(f"  {guess_type:<10} initial: {iterations:>4} iterations")
        
    return results


###############################################################################
# Summary visualization and Table
###############################################################################
def plot_summary(results):
    """Create summary and relative improvement table"""
    pot_types = list(results.keys())
    guess_types = list(results[pot_types[0]].keys())
    
    # Calculate Improvements relative to zero
    print("\n" + "="*120)
    print(f"{'COMPREHENSIVE SUMMARY TABLE (Iterations & % Improvement over Zero)':^120}")
    print("="*120)
    
    # Create Header
    # Format: Potential | Zero (iter) | Ones (iter, %) | Random (iter, %) ...
    header = f"{'Potential':<20} | {'Zero':<8}"
    for gt in guess_types:
        if gt == 'zero': continue
        header += f" | {gt.capitalize():<18}"
    print(header)
    print("-" * 120)
    
    for pot in pot_types:
        zero_iters = results[pot]['zero']['iterations']
        row = f"{pot:<20} | {zero_iters:<8}"
        
        for gt in guess_types:
            if gt == 'zero': continue
            
            iters = results[pot][gt]['iterations']
            
            # Calculate improvement: (Zero - Current) / Zero
            if zero_iters > 0:
                imp = (zero_iters - iters) / zero_iters * 100
            else:
                imp = 0.0
                
            # Formatting: Show iteration and percentage
            row += f" | {iters:<4} ({imp:>5.1f}%)     "
            
        print(row)
    print("="*120)
    
    # Plotting code
    data = {gt: [results[pt][gt]['iterations'] for pt in pot_types] for gt in guess_types}
    fig, ax1 = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(pot_types))
    width = 0.15
    colors = {'zero': '#3498db', 'ones': '#2ecc71', 'random': '#9b59b6', 'gaussian': '#f1c40f', 'ml': '#e74c3c'}
    
    for i, gt in enumerate(guess_types):
        offset = (i - len(guess_types)/2 + 0.5) * width
        ax1.bar(x_pos + offset, data[gt], width, label=f'{gt}', color=colors.get(gt, 'gray'))
    
    ax1.set_xlabel('Potential Type')
    ax1.set_ylabel('Iterations to Converge')
    ax1.set_title('SCF Convergence Speed Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(pot_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results_potentials/summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    num_electron = 17 
    potential_types = ["harmonic", "infinite_well", "finite_well", "step", "double_square_well"]
    
    print("="*70)
    print("TESTING DIFFERENT POTENTIAL TYPES")
    print(f"Number of electrons: {num_electron}")
    print("="*70)
    
    results = compare_potentials(potential_types, num_electron=num_electron, use_ml=True)
    plot_summary(results)
    
    print("\n✓ All results saved to 'results_potentials/' directory")