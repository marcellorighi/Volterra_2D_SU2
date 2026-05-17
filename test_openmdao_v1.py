import numpy as np
import openmdao.api as om
from scipy.special import eval_laguerre

# ============================================================================
# Component 1: Residual Computation
# ============================================================================

class VoltterraResidualComp(om.ExplicitComponent):
    """
    Compute residuals for Volterra kernel identification
    
    Residual = A @ h - y
    """
    
    def initialize(self):
        self.options.declare('A', types=np.ndarray, desc='System matrix')
        self.options.declare('y', types=np.ndarray, desc='Output vector')
        self.options.declare('kernel_name', types=str, desc='Name for this kernel (e.g., cl, cm)')
        self.options.declare('n_coeffs', types=int, desc='Number of kernel coefficients')
    
    def setup(self):
        A = self.options['A']
        y = self.options['y']
        kernel_name = self.options['kernel_name']
        n_coeffs = self.options['n_coeffs']
        
        # Design variable: kernel coefficients
        self.add_input(f'h_{kernel_name}', shape=n_coeffs, desc=f'Kernel coefficients for {kernel_name}')
        
        # Output: residual vector
        self.add_output(f'residual_{kernel_name}', shape=len(y), desc=f'Residual for {kernel_name}')
        
        # Declare partials (Jacobian is just A)
        self.declare_partials(f'residual_{kernel_name}', f'h_{kernel_name}', val=A)
    
    def compute(self, inputs, outputs):
        A = self.options['A']
        y = self.options['y']
        kernel_name = self.options['kernel_name']
        
        h = inputs[f'h_{kernel_name}']
        
        # Compute residual
        outputs[f'residual_{kernel_name}'] = A @ h - y


# ============================================================================
# Component 2: Regularization Penalty
# ============================================================================

class RegularizationComp(om.ExplicitComponent):
    """
    Compute regularization penalties:
    - Smoothness (second derivative penalty)
    - Decay enforcement (exponential penalty on later coefficients)
    - L2 norm (ridge penalty)
    """
    
    def initialize(self):
        self.options.declare('kernel_name', types=str)
        self.options.declare('n_coeffs', types=int)
        self.options.declare('m', types=int, desc='Memory length per kernel order')
        self.options.declare('lambda_smooth', default=0.01)
        self.options.declare('lambda_decay', default=0.1)
        self.options.declare('lambda_ridge', default=0.001)
        self.options.declare('decay_rate', default=5.0)
    
    def setup(self):
        kernel_name = self.options['kernel_name']
        n_coeffs = self.options['n_coeffs']
        m = self.options['m']
        
        self.add_input(f'h_{kernel_name}', shape=n_coeffs)
        
        # Outputs: different penalty terms
        self.add_output(f'smoothness_penalty_{kernel_name}', val=0.0)
        self.add_output(f'decay_penalty_{kernel_name}', val=0.0)
        self.add_output(f'ridge_penalty_{kernel_name}', val=0.0)
        
        # Build regularization matrices
        self._build_regularization_matrices()
        
        # Declare partials
        self.declare_partials(f'smoothness_penalty_{kernel_name}', f'h_{kernel_name}')
        self.declare_partials(f'decay_penalty_{kernel_name}', f'h_{kernel_name}')
        self.declare_partials(f'ridge_penalty_{kernel_name}', f'h_{kernel_name}')
    
    def _build_regularization_matrices(self):
        """Build D2 (smoothness) and L (decay) matrices"""
        from scipy.linalg import block_diag
        
        m = self.options['m']
        n_coeffs = self.options['n_coeffs']
        n_orders = n_coeffs // m  # Number of kernel orders
        
        # Second derivative operator for one order
        D2_single = np.diff(np.eye(m), n=2, axis=0)
        
        # Decay penalty for one order
        decay_rate = self.options['decay_rate']
        L_single = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
        
        # Block diagonal for all orders
        D2_blocks = [D2_single for _ in range(n_orders)]
        L_blocks = [L_single for _ in range(n_orders)]
        
        self.D2 = block_diag(*D2_blocks)
        self.L = block_diag(*L_blocks)
        
        # Precompute D2^T @ D2 and store
        self.D2TD2 = self.D2.T @ self.D2
        
    def compute(self, inputs, outputs):
        kernel_name = self.options['kernel_name']
        lambda_smooth = self.options['lambda_smooth']
        lambda_decay = self.options['lambda_decay']
        lambda_ridge = self.options['lambda_ridge']
        
        h = inputs[f'h_{kernel_name}']
        
        # Smoothness penalty: ||D2 @ h||^2
        D2h = self.D2 @ h
        outputs[f'smoothness_penalty_{kernel_name}'] = lambda_smooth * np.dot(D2h, D2h)
        
        # Decay penalty: h^T @ L @ h
        outputs[f'decay_penalty_{kernel_name}'] = lambda_decay * np.dot(h, self.L @ h)
        
        # Ridge penalty: ||h||^2
        outputs[f'ridge_penalty_{kernel_name}'] = lambda_ridge * np.dot(h, h)
    
    def compute_partials(self, inputs, partials):
        kernel_name = self.options['kernel_name']
        lambda_smooth = self.options['lambda_smooth']
        lambda_decay = self.options['lambda_decay']
        lambda_ridge = self.options['lambda_ridge']
        
        h = inputs[f'h_{kernel_name}']
        
        # Gradient of smoothness penalty: 2 * lambda_smooth * D2^T @ D2 @ h
        partials[f'smoothness_penalty_{kernel_name}', f'h_{kernel_name}'] = \
            2 * lambda_smooth * (self.D2TD2 @ h)
        
        # Gradient of decay penalty: 2 * lambda_decay * L @ h
        partials[f'decay_penalty_{kernel_name}', f'h_{kernel_name}'] = \
            2 * lambda_decay * (self.L @ h)
        
        # Gradient of ridge penalty: 2 * lambda_ridge * h
        partials[f'ridge_penalty_{kernel_name}', f'h_{kernel_name}'] = \
            2 * lambda_ridge * h


# ============================================================================
# Component 3: Laguerre Basis Transformation (Optional)
# ============================================================================

class LaguerreTransformComp(om.ExplicitComponent):
    """
    Transform Laguerre coefficients to time-domain kernel
    
    h(t) = L @ c, where L is Laguerre basis matrix, c are coefficients
    """
    
    def initialize(self):
        self.options.declare('kernel_name', types=str)
        self.options.declare('m', types=int, desc='Memory length per order')
        self.options.declare('n_orders', types=int, desc='Number of kernel orders')
        self.options.declare('n_basis', default=15, desc='Number of Laguerre functions')
        self.options.declare('alpha', default=0.7, desc='Laguerre pole (0 < alpha < 1)')
    
    def setup(self):
        kernel_name = self.options['kernel_name']
        m = self.options['m']
        n_orders = self.options['n_orders']
        n_basis = self.options['n_basis']
        
        # Input: Laguerre coefficients (much fewer than time-domain coeffs)
        n_laguerre_coeffs = n_orders * n_basis
        self.add_input(f'c_{kernel_name}', shape=n_laguerre_coeffs, 
                      desc='Laguerre coefficients')
        
        # Output: time-domain kernel coefficients
        n_time_coeffs = n_orders * m
        self.add_output(f'h_{kernel_name}', shape=n_time_coeffs,
                       desc='Time-domain kernel')
        
        # Build Laguerre basis
        self._build_laguerre_basis()
        
        # Jacobian is just the basis matrix
        self.declare_partials(f'h_{kernel_name}', f'c_{kernel_name}', val=self.L_full)
    
    def _build_laguerre_basis(self):
        """Generate Laguerre basis functions"""
        from scipy.linalg import block_diag
        
        m = self.options['m']
        n_basis = self.options['n_basis']
        alpha = self.options['alpha']
        n_orders = self.options['n_orders']
        
        # Create basis for one kernel order
        t = np.arange(m)
        L_single = np.zeros((m, n_basis))
        
        for n in range(n_basis):
            # Discrete Laguerre functions
            L_single[:, n] = (np.sqrt(1 - alpha**2) * (alpha**t) * 
                             eval_laguerre(n, 2*alpha*t/(1-alpha**2)))
        
        # Block diagonal for all orders
        L_blocks = [L_single for _ in range(n_orders)]
        self.L_full = block_diag(*L_blocks)
    
    def compute(self, inputs, outputs):
        kernel_name = self.options['kernel_name']
        
        c = inputs[f'c_{kernel_name}']
        outputs[f'h_{kernel_name}'] = self.L_full @ c


# ============================================================================
# Component 4: Objective Function (Sum of Squared Residuals)
# ============================================================================

class ObjectiveComp(om.ExplicitComponent):
    """
    Total objective: sum of residual norms and regularization penalties
    """
    
    def initialize(self):
        self.options.declare('kernel_names', types=list, desc='List of kernel names (e.g., [cl, cm])')
        self.options.declare('n_samples', types=int, desc='Number of samples in residual vector')
    
    def setup(self):
        kernel_names = self.options['kernel_names']
        n_samples = self.options['n_samples']
        
        # Inputs: residuals and penalties for each kernel
        for name in kernel_names:
            self.add_input(f'residual_{name}', shape=n_samples)
            self.add_input(f'smoothness_penalty_{name}', val=0.0)
            self.add_input(f'decay_penalty_{name}', val=0.0)
            self.add_input(f'ridge_penalty_{name}', val=0.0)
        
        # Output: total objective
        self.add_output('objective', val=0.0)
        
        # Declare partials
        for name in kernel_names:
            self.declare_partials('objective', f'residual_{name}')
            self.declare_partials('objective', f'smoothness_penalty_{name}', val=1.0)
            self.declare_partials('objective', f'decay_penalty_{name}', val=1.0)
            self.declare_partials('objective', f'ridge_penalty_{name}', val=1.0)
    
    def compute(self, inputs, outputs):
        kernel_names = self.options['kernel_names']
        
        total_obj = 0.0
        
        for name in kernel_names:
            # Sum of squared residuals
            residual = inputs[f'residual_{name}']
            total_obj += np.dot(residual, residual)
            
            # Add penalties
            total_obj += inputs[f'smoothness_penalty_{name}']
            total_obj += inputs[f'decay_penalty_{name}']
            total_obj += inputs[f'ridge_penalty_{name}']
        
        outputs['objective'] = total_obj
    
    def compute_partials(self, inputs, partials):
        kernel_names = self.options['kernel_names']
        
        for name in kernel_names:
            # Gradient w.r.t. residual: 2 * residual
            residual = inputs[f'residual_{name}']
            partials['objective', f'residual_{name}'] = 2 * residual


# ============================================================================
# Main Optimization Problem Setup
# ============================================================================

class VoltterraIdentificationProblem(om.Group):
    """
    Complete Volterra identification problem as OpenMDAO group
    """
    
    def initialize(self):
        self.options.declare('A', types=np.ndarray)
        self.options.declare('cl', types=np.ndarray)
        self.options.declare('cm', types=np.ndarray)
        self.options.declare('m', types=int, desc='Memory length per kernel order')
        self.options.declare('use_laguerre', default=False, types=bool)
        self.options.declare('n_basis', default=15, types=int)
        self.options.declare('lambda_smooth', default=0.01)
        self.options.declare('lambda_decay', default=0.1)
        self.options.declare('lambda_ridge', default=0.001)
        self.options.declare('decay_rate', default=5.0)
        self.options.declare('alpha_laguerre', default=0.7)
    
    def setup(self):
        A = self.options['A']
        cl = self.options['cl']
        cm = self.options['cm']
        m = self.options['m']
        use_laguerre = self.options['use_laguerre']
        
        n_samples, n_coeffs = A.shape
        n_orders = n_coeffs // m
        
        kernel_names = ['cl', 'cm']
        
        # ==================================================================
        # Design Variables
        # ==================================================================
        
        if use_laguerre:
            n_basis = self.options['n_basis']
            alpha = self.options['alpha_laguerre']
            
            # Laguerre coefficients as design variables
            for name in kernel_names:
                self.add_subsystem(f'laguerre_{name}',
                                  LaguerreTransformComp(kernel_name=name,
                                                       m=m,
                                                       n_orders=n_orders,
                                                       n_basis=n_basis,
                                                       alpha=alpha),
                                  promotes_inputs=[(f'c_{name}', f'c_{name}')],
                                  promotes_outputs=[(f'h_{name}', f'h_{name}')])
        else:
            # Direct kernel coefficients as design variables (via IndepVarComp)
            ivcomp = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
            for name in kernel_names:
                ivcomp.add_output(f'h_{name}', val=np.zeros(n_coeffs))
        
        # ==================================================================
        # Residual Computation
        # ==================================================================
        
        for name, y in zip(kernel_names, [cl, cm]):
            self.add_subsystem(f'residual_{name}',
                              VoltterraResidualComp(A=A, y=y, 
                                                   kernel_name=name,
                                                   n_coeffs=n_coeffs),
                              promotes_inputs=[(f'h_{name}', f'h_{name}')],
                              promotes_outputs=[(f'residual_{name}', f'residual_{name}')])
        
        # ==================================================================
        # Regularization
        # ==================================================================
        
        for name in kernel_names:
            self.add_subsystem(f'regularization_{name}',
                              RegularizationComp(kernel_name=name,
                                                n_coeffs=n_coeffs,
                                                m=m,
                                                lambda_smooth=self.options['lambda_smooth'],
                                                lambda_decay=self.options['lambda_decay'],
                                                lambda_ridge=self.options['lambda_ridge'],
                                                decay_rate=self.options['decay_rate']),
                              promotes_inputs=[(f'h_{name}', f'h_{name}')],
                              promotes_outputs=[(f'smoothness_penalty_{name}', f'smoothness_penalty_{name}'),
                                              (f'decay_penalty_{name}', f'decay_penalty_{name}'),
                                              (f'ridge_penalty_{name}', f'ridge_penalty_{name}')])
        
        # ==================================================================
        # Objective Function
        # ==================================================================
        
        self.add_subsystem('objective',
                          ObjectiveComp(kernel_names=kernel_names,
                                       n_samples=n_samples),
                          promotes_inputs=[f'residual_{name}' for name in kernel_names] +
                                        [f'{penalty}_{name}' 
                                         for name in kernel_names 
                                         for penalty in ['smoothness_penalty', 'decay_penalty', 'ridge_penalty']],
                          promotes_outputs=['objective'])


# ============================================================================
# Setup and Run Functions
# ============================================================================

def run_volterra_identification(A, cl, cm, m, 
                                use_laguerre=False,
                                n_basis=6,
                                lambda_smooth=0.01,
                                lambda_decay=0.1,
                                lambda_ridge=0.001,
                                decay_rate=5.0,
                                alpha_laguerre=0.7,
                                optimizer='SLSQP',
                                max_iter=4000,
                                tol=1e-6,
                                verbose=True):
    """
    Run Volterra kernel identification optimization
    
    Parameters:
    -----------
    A : ndarray, shape (ntsteps, 8m)
        System matrix
    cl, cm : ndarray, shape (ntsteps,)
        Output vectors
    m : int
        Memory length per kernel order
    use_laguerre : bool
        If True, use Laguerre basis expansion
    n_basis : int
        Number of Laguerre basis functions (if use_laguerre=True)
    lambda_smooth : float
        Smoothness regularization weight
    lambda_decay : float
        Decay penalty weight
    lambda_ridge : float
        Ridge penalty weight
    decay_rate : float
        Exponential decay rate for penalty
    alpha_laguerre : float
        Laguerre pole parameter (0 < alpha < 1)
    optimizer : str
        Optimizer to use ('SLSQP', 'IPOPT', 'SNOPT', etc.)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print optimization progress
    
    Returns:
    --------
    h_cl, h_cm : ndarray
        Identified kernel coefficients
    prob : om.Problem
        OpenMDAO problem object (for inspection)
    """
    
    # Create problem
    prob = om.Problem()
    
    # Add the group
    prob.model = VoltterraIdentificationProblem(
        A=A, cl=cl, cm=cm, m=m,
        use_laguerre=use_laguerre,
        n_basis=n_basis,
        lambda_smooth=lambda_smooth,
        lambda_decay=lambda_decay,
        lambda_ridge=lambda_ridge,
        decay_rate=decay_rate,
        alpha_laguerre=alpha_laguerre
    )
    
    # Setup driver
    if optimizer == 'SLSQP':
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['maxiter'] = max_iter
        prob.driver.options['tol'] = tol
    elif optimizer == 'IPOPT':
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'IPOPT'
        prob.driver.opt_settings['max_iter'] = max_iter
        prob.driver.opt_settings['tol'] = tol
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")
    
    # if verbose:
    #     prob.driver.options['disp'] = True
    
    # Add design variables
    n_coeffs = A.shape[1]
    
    if use_laguerre:
        n_laguerre_coeffs = (n_coeffs // m) * n_basis
        prob.model.add_design_var('c_cl', lower=-10.0, upper=10.0)
        prob.model.add_design_var('c_cm', lower=-10.0, upper=10.0)
    else:
        prob.model.add_design_var('h_cl', lower=-10.0, upper=10.0)
        prob.model.add_design_var('h_cm', lower=-10.0, upper=10.0)
    
    # Add objective
    prob.model.add_objective('objective')
    
    # Setup and run
    prob.setup()
    
    # Set initial guess (small random values)
    np.random.seed(42)
    if use_laguerre:
        prob.set_val('c_cl', 0.01 * np.random.randn(n_laguerre_coeffs))
        prob.set_val('c_cm', 0.01 * np.random.randn(n_laguerre_coeffs))
    else:
        prob.set_val('h_cl', 0.01 * np.random.randn(n_coeffs))
        prob.set_val('h_cm', 0.01 * np.random.randn(n_coeffs))
    
    # Run optimization
    prob.run_driver()
    
    # Extract results
    h_cl = prob.get_val('h_cl')
    h_cm = prob.get_val('h_cm')
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Final objective value: {prob.get_val('objective')[0]:.6e}")
        print(f"h_cl norm: {np.linalg.norm(h_cl):.6e}")
        print(f"h_cm norm: {np.linalg.norm(h_cm):.6e}")
        
        # Compute final residuals
        residual_cl = A @ h_cl - cl
        residual_cm = A @ h_cm - cm
        nmse_cl = np.mean(residual_cl**2) / np.var(cl)
        nmse_cm = np.mean(residual_cm**2) / np.var(cm)
        
        print(f"NMSE (cl): {nmse_cl:.6e}")
        print(f"NMSE (cm): {nmse_cm:.6e}")
    
    return h_cl, h_cm, prob


# ============================================================================
# Example Usage
# ============================================================================

# if __name__ == "__main__":
    
#     # Example: Create synthetic data
#     print("Creating synthetic test data...")
    
#     ntsteps = 500
#     m = 20  # Memory per order
#     n_orders = 8  # You mentioned 8m total coefficients
#     n_coeffs = n_orders * m  # 8m total
    
#     # Random system matrix
#     np.random.seed(42)
#     A = np.random.randn(ntsteps, n_coeffs) * 0.1
    
#     # True kernels (with decay)
#     h_true_cl = np.zeros(n_coeffs)
#     h_true_cm = np.zeros(n_coeffs)
    
#     for order in range(n_orders):
#         decay = np.exp(-2 * np.linspace(0, 1, m))
#         h_true_cl[order*m:(order+1)*m] = (order + 1) * decay * 0.1
#         h_true_cm[order*m:(order+1)*m] = (order + 1) * decay * 0.15
    
#     # Generate outputs with noise
#     cl = A @ h_true_cl + 0.01 * np.random.randn(ntsteps)
#     cm = A @ h_true_cm + 0.01 * np.random.randn(ntsteps)
    
#     print(f"Data shape: A={A.shape}, cl={cl.shape}, cm={cm.shape}")
#     print(f"Memory length m={m}, Total coefficients={n_coeffs}")
    
#     # ========================================================================
#     # Run identification - Direct coefficients
#     # ========================================================================
    
#     print("\n" + "="*70)
#     print("RUNNING OPTIMIZATION: Direct Coefficients")
#     print("="*70)
    
#     h_cl_direct, h_cm_direct, prob_direct = run_volterra_identification(
#         A, cl, cm, m,
#         use_laguerre=False,
#         lambda_smooth=0.01,
#         lambda_decay=0.1,
#         lambda_ridge=0.001,
#         decay_rate=5.0,
#         optimizer='SLSQP',
#         max_iter=500,
#         verbose=True
#     )
    
#     # ========================================================================
#     # Run identification - Laguerre basis
#     # ========================================================================
    
#     print("\n" + "="*70)
#     print("RUNNING OPTIMIZATION: Laguerre Basis")
#     print("="*70)
    
#     h_cl_laguerre, h_cm_laguerre, prob_laguerre = run_volterra_identification(
#         A, cl, cm, m,
#         use_laguerre=True,
#         n_basis=8,  # Fewer parameters than direct (8 vs 20 per order)
#         alpha_laguerre=0.7,
#         lambda_smooth=0.005,  # Less smoothing needed with Laguerre
#         lambda_decay=0.05,
#         lambda_ridge=0.01,
#         optimizer='SLSQP',
#         max_iter=500,
#         verbose=True
#     )
    
#     # ========================================================================
#     # Compare results
#     # ========================================================================
    
#     import matplotlib.pyplot as plt
    
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     # Plot kernels for first two orders
#     for order in range(min(2, n_orders)):
#         # CL kernels
#         axes[0, order].plot(h_true_cl[order*m:(order+1)*m], 
#                            'k-', linewidth=2, label='True')
#         axes[0, order].plot(h_cl_direct[order*m:(order+1)*m], 
#                            'b--', linewidth=2, label='Direct')
#         axes[0, order].plot(h_cl_laguerre[order*m:(order+1)*m], 
#                            'r:', linewidth=2, label='Laguerre')
#         axes[0, order].set_xlabel('Lag index')
#         axes[0, order].set_ylabel('Coefficient value')
#         axes[0, order].set_title(f'CL Kernel - Order {order}')
#         axes[0, order].legend()
#         axes[0, order].grid(True, alpha=0.3)
        
#         # CM kernels
#         axes[1, order].plot(h_true_cm[order*m:(order+1)*m], 
#                            'k-', linewidth=2, label='True')
#         axes[1, order].plot(h_cm_direct[order*m:(order+1)*m], 
#                            'b--', linewidth=2, label='Direct')
#         axes[1, order].plot(h_cm_laguerre[order*m:(order+1)*m], 
#                            'r:', linewidth=2, label='Laguerre')
#         axes[1, order].set_xlabel('Lag index')
#         axes[1, order].set_ylabel('Coefficient value')
#         axes[1, order].set_title(f'CM Kernel - Order {order}')
#         axes[1, order].legend()
#         axes[1, order].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('volterra_openmdao_results.png', dpi=150)
#     plt.show()
    
#     print("\n" + "="*70)
#     print("COMPARISON COMPLETE")
#     print("="*70)

import csv 

def read_csv(filename):
    with open(filename, newline='') as f_input:
        next(f_input)
        next(f_input)
        return [list(map(float, row)) for row in csv.reader(f_input)]


def generate_schroeder_multisine(freqs, dt, duration, amp=1.0):
    t = np.arange(0, duration, dt)
    N = len(freqs)
    # Schroeder Phase Formula: phi_k = -k*(k-1)*pi / N
    phases = [-k * (k - 1) * np.pi / N for k in range(  1, N + 1)]
    
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += amp * np.sin(2 * np.pi * f * t + phases[i])
    
    # Normalize to keep within original amplitude constraints
    # signal = signal / np.max(np.abs(signal)) * amp
    return t, signal


home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA06" 
filename = home_pitch + "/history_05_00002.dat"    

mach = 0.8
chord = 0.4064 #1. 
v_inf = 123. #mach * 10 / 8 * 277.24 # 123. 

# mach = 0.8
# v_inf = mach * 10 / 8 * 277.24 # 123. 
# chord = 1. 

tref = chord / (2 * v_inf)
subs = 48 

# dt = 7.5e-4 
dt = 2.5e-4 

iCd = 11
iCl = 12
iCm = 16

print(filename)
data = np.array(read_csv(filename)) 
cd = (data[0*subs::subs,iCd] ) #/ daoa_plunge
cl_output = (data[0*subs::subs,iCl] ) #/ daoa_plunge
cm_output = - (data[0::subs,iCm] ) # sign differs in 2 and 3D?
it_05 = data[0*subs::subs,0]
tau_vector = (it_05-2)*dt/tref
tt_05 = (it_05-2)*dt

# cm_output = -cm_output + 0.25*cl_output

ntsteps = len(cl_output)

target_freqs_pitching = 12.5 / (2.*np.pi) * np.array([1.4048, 2.6437, 5.0113]) 
target_freqs_plunging = 12.5 / (2.*np.pi) * np.array([1.0, 1.9226, 3.6286])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.50 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.10)

zdot_input = np.gradient(plunge_input,tt_05)/v_inf # SIGN!!!!!
thetadot_input = np.gradient(pitch_input, tt_05) /v_inf

# -----
m = 180 #180 #160 #80 #80

cl0 = cl_output[0]
cm0 = cm_output[0]

cl_output -= cl0
cm_output -= cm0 

A_pitch = np.zeros((ntsteps,m))
A_thetadot = np.zeros((ntsteps,m))
A_zdot = np.zeros((ntsteps,m))

A_pitch2 = A_pitch*A_pitch
A_thetadot2 = A_thetadot*A_thetadot
A_zdot2 = A_zdot*A_zdot
A_pitch_zdot = A_pitch*A_zdot
A_thetadot_zdot = A_thetadot*A_zdot

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn] 
    A_thetadot[icolumn:,icolumn] = thetadot_input[:ntsteps-icolumn] 
    A_zdot[icolumn:,icolumn] = zdot_input[:ntsteps-icolumn] 

A = np.hstack( (A_pitch, A_thetadot, A_zdot, A_pitch2,A_thetadot2,A_zdot2,A_pitch_zdot,A_thetadot_zdot))

# Load your data
# A: your system matrix (ntsteps, 8m)
# cl, cm: your output vectors

# Example with your data
h_cl, h_cm, prob = run_volterra_identification(
    A=A,           # Your A matrix
    cl=cl_output,         # Your cl vector
    cm=cm_output,         # Your cm vector
    m=m,          # Your memory length
    use_laguerre=False,  # or True for basis expansion
    lambda_smooth=0.01,
    lambda_decay=0.1,
    lambda_ridge=0.01,
    decay_rate=2.0,
    optimizer='IPOPT',
    max_iter=2500,
    verbose=True
)

# Access results
print(f"Identified h_cl shape: {h_cl.shape}")
print(f"Identified h_cm shape: {h_cm.shape}")

# Visualize specific kernel orders
import matplotlib.pyplot as plt

# m = 50
order = 0  # First order
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(h_cl[order*m:(order+1)*m*3], 'b.-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(h_cm[order*m:(order+1)*m*3], 'r.-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cm - Order {order}')
plt.grid(True)

plt.tight_layout()
plt.show()

order = 1  # First order
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(h_cl[order*3*m:], 'b.-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(h_cm[3*order*m:], 'r.-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cm - Order {order}')
plt.grid(True)

plt.tight_layout()
plt.show()

cl_check = A@h_cl
cm_check = A@h_cm

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cl_output, 'b-', linewidth=2)
plt.plot(cl_check, 'k-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cm_output, 'r-', linewidth=2)
plt.plot(cm_check, 'k-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cm - Order {order}')
plt.grid(True)

plt.tight_layout()
plt.show()

print(cl_check)
