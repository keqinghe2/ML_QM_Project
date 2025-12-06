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
        k = params.get('k', 1.0) 
        V_confining = 0.5 * k * x**2
        V_external = V_confining.copy()
        
    elif potential_type == "infinite_well":
        # Infinite well usually defined by boundaries
        # For grid based, we set high potential outside
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
            # Note: Changing infinite walls on fixed grid is tricky, 
            # so we focus on internal perturbations
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
    # Regularization (rcond) helps numerical stability
    U, S, VT = np.linalg.svd(X_train, full_matrices=False)
    
    # Filter small singular values (Regularization)
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
# Initial guess generators
###############################################################################
def get_initial_guess(guess_type, x, V_external, W, num_electron):
    """
    Generate different types of initial density guesses
    
    Parameters:
    - guess_type: "zero", "gaussian", "random", "ml"
    - x: grid
    - V_external: external potential
    - W: ML weight matrix (only used for "ml")
    - num_electron: number of electrons to normalize to
    
    Returns: initial density array
    """
    if guess_type == "zero":
        # All zeros
        nx = np.zeros_like(x)
        
    elif guess_type == "gaussian":
        # Gaussian centered at x=0 with width sigma=1
        sigma = 1.0
        nx = np.exp(-x**2 / (2 * sigma**2))
        
    elif guess_type == "random":
        # Random values (uniform between 0 and 1)
        np.random.seed(None)  # Use different seed each time
        nx = np.random.rand(len(x))
        
    elif guess_type == "ml":
        # Machine learning prediction
        nx = V_external @ W
        # Ensure non-negative
        nx = np.maximum(nx, 0)
        
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
    """
    Compare SCF convergence for different potentials with 4 different initial guesses:
    - zero: all zeros
    - gaussian: Gaussian distribution
    - random: random values
    - ml: Machine learning prediction (trained separately for each potential type)
    """
    results = {}
    
    # Train ML models separately for each potential type
    ml_models = {}
    if use_ml:
        for pot_type in potential_types:
            ml_models[pot_type] = train_ml_model(x, D2, num_electron, pot_type, n_training=20)
        print("\n" + "="*70)
        print("NOTE: ML models trained SEPARATELY for each potential type")
        print("="*70)
    
    # Define the initial guess types to test
    guess_types = ["zero", "gaussian", "random", "ml"] if use_ml else ["zero", "gaussian", "random"]
    
    for pot_type in potential_types:
        print("\n" + "="*70)
        print(f"TESTING POTENTIAL: {pot_type.upper()}")
        print("="*70)
        
        V_confining, V_external = get_potential(pot_type, x)
        
        # Get the appropriate ML model for this potential type
        W = ml_models.get(pot_type) if use_ml else None
        
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
        
        # Test each initial guess type
        results[pot_type] = {}
        
        for guess_type in guess_types:
            # Generate initial guess
            nx_init = get_initial_guess(guess_type, x, V_external, W, num_electron)
            
            # Run SCF
            converged, iterations, final_E, history = run_scf(
                x, D2, V_external, num_electron, nx_init,
                label=f"{pot_type} - {guess_type} initial", verbose=False
            )
            
            # Store results
            results[pot_type][guess_type] = {
                'converged': converged,
                'iterations': iterations,
                'energy': final_E,
                'history': history,
            }
            
            print(f"  {guess_type:<10} initial: {iterations:>4} iterations, E = {final_E:.8f}")
        
        # Calculate savings relative to zero initial guess
        zero_iter = results[pot_type]['zero']['iterations']
        print(f"\n  Savings relative to zero initial:")
        for guess_type in guess_types:
            if guess_type == 'zero':
                continue
            iters = results[pot_type][guess_type]['iterations']
            savings = (zero_iter - iters) / zero_iter * 100 if zero_iter > 0 else 0
            print(f"    {guess_type:<10}: {zero_iter - iters:>4} iterations ({savings:>6.1f}%)")
        
        # Plot convergence comparison for all initial guesses
        plt.figure(figsize=(12, 6))
        colors = {'zero': 'blue', 'gaussian': 'orange', 'random': 'purple', 'ml': 'red'}
        markers = {'zero': 'o', 'gaussian': '^', 'random': 'v', 'ml': 'D'}
        
        for guess_type in guess_types:
            hist = results[pot_type][guess_type]['history']
            final_E = results[pot_type][guess_type]['energy']
            plt.semilogy(range(len(hist)), 
                        np.abs(np.array(hist) - final_E), 
                        color=colors[guess_type],
                        marker=markers[guess_type],
                        linewidth=2, 
                        markersize=4, 
                        label=f'{guess_type} initial',
                        markevery=max(1, len(hist)//20))
        
        plt.xlabel('Iteration')
        plt.ylabel('|E - E_final|')
        plt.title(f'Convergence Comparison: {pot_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"results_potentials/{pot_type}_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results


###############################################################################
# Summary visualization
###############################################################################
def plot_summary(results):
    """Create summary bar chart comparing all potentials and initial guesses"""
    pot_types = list(results.keys())
    guess_types = list(results[pot_types[0]].keys())
    
    # Collect iteration counts for each combination
    data = {gt: [results[pt][gt]['iterations'] for pt in pot_types] for gt in guess_types}
    
    # Calculate savings relative to zero
    zero_iters = data['zero']
    savings = {}
    for gt in guess_types:
        if gt != 'zero':
            savings[gt] = [(z - i)/z * 100 if z > 0 else 0 
                          for z, i in zip(zero_iters, data[gt])]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Iterations comparison
    x_pos = np.arange(len(pot_types))
    width = 0.2  # Adjusted for 4 bars
    colors = {'zero': 'skyblue', 'gaussian': 'orange', 'random': 'purple', 'ml': 'salmon'}
    
    for i, gt in enumerate(guess_types):
        offset = (i - len(guess_types)/2 + 0.5) * width
        ax1.bar(x_pos + offset, data[gt], width, label=f'{gt} initial', color=colors[gt])
    
    ax1.set_xlabel('Potential Type')
    ax1.set_ylabel('Iterations to Converge')
    ax1.set_title('SCF Convergence Speed')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(pot_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Savings percentage (relative to zero)
    x_pos = np.arange(len(pot_types))
    width = 0.25  # Adjusted for 3 bars
    colors_savings = {'gaussian': 'orange', 'random': 'purple', 'ml': 'red'}
    
    for i, gt in enumerate([g for g in guess_types if g != 'zero']):
        offset = (i - (len(guess_types)-2)/2 + 0.5) * width
        bars = ax2.bar(x_pos + offset, savings[gt], width, 
                      label=f'{gt} vs zero', color=colors_savings[gt], alpha=0.7)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, savings[gt])):
            if abs(val) > 5:  # Only label if savings significant
                ax2.text(bar.get_x() + bar.get_width()/2, val + (2 if val > 0 else -2),
                        f'{val:.0f}%', ha='center', va='bottom' if val > 0 else 'top', 
                        fontsize=7)
    
    ax2.set_xlabel('Potential Type')
    ax2.set_ylabel('Iteration Savings (%)')
    ax2.set_title('Improvement Over Zero Initial Guess')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pot_types, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results_potentials/summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive table
    print("\n" + "="*110)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*110)
    print(f"{'Potential':<20} {'Zero':<8} {'Gaussian':<10} {'Random':<8} {'ML':<8} "
          f"{'Gauss vs Zero':<15} {'Random vs Zero':<16} {'ML vs Zero':<15}")
    print("─"*110)
    
    for pot in pot_types:
        zero_i = results[pot]['zero']['iterations']
        gauss_i = results[pot]['gaussian']['iterations']
        random_i = results[pot]['random']['iterations']
        ml_i = results[pot]['ml']['iterations'] if 'ml' in results[pot] else 0
        
        gauss_sav = (zero_i - gauss_i) / zero_i * 100 if zero_i > 0 else 0
        random_sav = (zero_i - random_i) / zero_i * 100 if zero_i > 0 else 0
        ml_sav = (zero_i - ml_i) / zero_i * 100 if zero_i > 0 and ml_i > 0 else 0
        
        print(f"{pot:<20} {zero_i:<8} {gauss_i:<10} {random_i:<8} {ml_i:<8} "
              f"{gauss_i-zero_i:>5} ({gauss_sav:>5.1f}%)  "
              f"{random_i-zero_i:>5} ({random_sav:>5.1f}%)   "
              f"{ml_i-zero_i:>5} ({ml_sav:>5.1f}%)")
    
    print("="*110)


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
        "double_square_well"
    ]
    
    print("="*70)
    print("TESTING DIFFERENT POTENTIAL TYPES")
    print("Comparing 4 initial guesses: zero, gaussian, random, ML")
    print(f"Number of electrons: {num_electron}")
    print(f"Grid points: {n_grid}")
    print("="*70)
    
    # Run comparison
    results = compare_potentials(potential_types, num_electron=num_electron, use_ml=True)
    
    # Create summary visualization
    plot_summary(results)
    
    print("\n✓ All results saved to 'results_potentials/' directory")