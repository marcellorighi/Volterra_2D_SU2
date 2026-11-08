import numpy as np
import openmdao.api as om
from scipy.special import eval_laguerre
from glob import glob
import pyvista as pv
from tqdm import tqdm

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
# Main Optimization Problem Setup
# ============================================================================

class VoltterraIdentificationProblemModes(om.Group):
    """
    Complete Volterra identification problem as OpenMDAO group
    """
    
    def initialize(self):
        self.options.declare('A', types=np.ndarray)
        self.options.declare('cl', types=np.ndarray)
        self.options.declare('cm', types=np.ndarray)
        self.options.declare('modes_coeffs', types=np.ndarray)
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
        modes_coeffs = self.options['modes_coeffs']
        m = self.options['m']
        use_laguerre = self.options['use_laguerre']
        
        n_samples, n_coeffs = A.shape
        n_orders = n_coeffs // m
        r = modes_coeffs.shape[1]
        
        kernel_names = ['cl', 'cm']
        kernel_names += [
            f'mode_{i+1}'
            for i in range(r)
        ]            
        outputs = [
            cl,
            cm,
            *[modes_coeffs[:, i] for i in range(r)]
        ]

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
        
        # for name, y in zip(kernel_names, [cl, cm]):
        #     self.add_subsystem(f'residual_{name}',
        #                       VoltterraResidualComp(A=A, y=y, 
        #                                            kernel_name=name,
        #                                            n_coeffs=n_coeffs),
        #                       promotes_inputs=[(f'h_{name}', f'h_{name}')],
        #                       promotes_outputs=[(f'residual_{name}', f'residual_{name}')])

        # for i, name in enumerate(kernel_names):

        #     y = modes_coeffs[:, i]

        #     self.add_subsystem(
        #         f'residual_{name}',
        #         VoltterraResidualComp(
        #             A=A,
        #             y=y,
        #             kernel_name=name,
        #             n_coeffs=n_coeffs
        #         ),
        #         promotes_inputs=[
        #             (f'h_{name}', f'h_{name}')
        #         ],
        #         promotes_outputs=[
        #             (f'residual_{name}',
        #             f'residual_{name}')
        #         ]
        #     )

        for name, y in zip(kernel_names, outputs):

            self.add_subsystem(
                f'residual_{name}',
                VoltterraResidualComp(
                    A=A,
                    y=y,
                    kernel_name=name,
                    n_coeffs=n_coeffs
                ),
                promotes_inputs=[
                    (f'h_{name}', f'h_{name}')
                ],
                promotes_outputs=[
                    (f'residual_{name}', f'residual_{name}')
                ]
            )    
        
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
                                tol=1e-8,
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
        prob.driver = om.pyOptSparseDriver(
            optimizer='SLSQP'
        )

        prob.driver.opt_settings['MAXIT'] = max_iter
        prob.driver.opt_settings['ACC'] = tol
        # prob.driver = om.ScipyOptimizeDriver()
        # prob.driver.options['optimizer'] = 'SLSQP'
        # prob.driver.options['maxiter'] = max_iter
        # prob.driver.options['tol'] = tol
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
    
    # r = q.shape[1]
    kernel_names = ['cl', 'cm'] #+ [f'mode_{i+1}' for i in range(r)]

    if use_laguerre:
        n_laguerre_coeffs = (n_coeffs // m) * n_basis
        # prob.model.add_design_var('c_cl', lower=-45.0, upper=45.0)
        # prob.model.add_design_var('c_cm', lower=-45.0, upper=45.0)
        for name in kernel_names:
            prob.model.add_design_var(
                f'c_{name}',
                lower=-45.0,
                upper=45.0
            )
    else:
        # prob.model.add_design_var('h_cl', lower=-45.0, upper=45.0)
        # prob.model.add_design_var('h_cm', lower=-45.0, upper=45.0)
        for name in kernel_names:
            prob.model.add_design_var(
                f'h_{name}',
                lower=-45.0,
                upper=45.0
            )
        
    # Add objective
    prob.model.add_objective('objective')
    
    # Setup and run
    prob.setup()
    
    # Set initial guess (small random values)
    np.random.seed(42)
    if use_laguerre:
        # prob.set_val('c_cl', 0.01 * np.random.randn(n_laguerre_coeffs))
        # prob.set_val('c_cm', 0.01 * np.random.randn(n_laguerre_coeffs))
        for name in kernel_names:
            prob.set_val(
                f'c_{name}',
                0.01 * np.random.randn(n_laguerre_coeffs)
            )

    else:
        # prob.set_val('h_cl', 0.01 * np.random.randn(n_coeffs))
        # prob.set_val('h_cm', 0.01 * np.random.randn(n_coeffs))
        for name in kernel_names:
            prob.set_val(
                f'h_{name}',
                0.01 * np.random.randn(n_coeffs)
            )
    
    # Run optimization
    prob.run_driver()
    
    prob.model.list_outputs()

    # Extract results
    # h_cl = prob.get_val('h_cl')
    # h_cm = prob.get_val('h_cm')
    h = {}

    for name in kernel_names:
        h[name] = prob.get_val(f'h_{name}').copy()

    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Final objective value: {prob.get_val('objective')[0]:.6e}")
        print(f"h_cl norm: {np.linalg.norm(h['cl']):.6e}")
        print(f"h_cm norm: {np.linalg.norm(h['cm']):.6e}")
        
        # Compute final residuals
        residual_cl = A @ h['cl'] - cl
        residual_cm = A @ h['cm'] - cm
        nmse_cl = np.mean(residual_cl**2) / np.var(cl)
        nmse_cm = np.mean(residual_cm**2) / np.var(cm)
        
        print(f"NMSE (cl): {nmse_cl:.6e}")
        print(f"NMSE (cm): {nmse_cm:.6e}")
    
    return h, prob


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



def plot_aero_responses(h_cl_theta, h_cl_zdot, h_cl_thetadot, 
                        h_cm_theta, h_cm_zdot, h_cm_thetadot, 
                        omega, nt_int, dt):
    """
    Calculates and plots the aerodynamic lift (Cl) and moment (Cm) responses 
    for pitch, heave, and theta-dot components based on a sinusoidal AoA.
    """
    
    # --- 1. Define Time and Input Signal ---
    time_test = dt * np.arange(nt_int)
    # Sinusoidal Angle of Attack (1 degree amplitude)
    aoa = np.deg2rad(1.) * np.sin(omega * time_test)
    aoa_2 = np.deg2rad(1.) * np.sin(2 * omega * time_test)
    aoa_3 = np.deg2rad(1.) * np.sin(3 * omega * time_test)

    # --- 2. Calculate Responses ---
    # Assuming 'response' is a pre-defined function in your workspace
    cl_pitch    = response(h_cl_theta, aoa, nt_int)
    cl_heave    = response(h_cl_zdot, aoa, nt_int)
    cl_thetadot = response(h_cl_thetadot, aoa, nt_int)

    cm_pitch    = response(h_cm_theta, aoa, nt_int)
    cm_heave    = response(h_cm_zdot, aoa, nt_int)
    cm_thetadot = response(h_cm_thetadot, aoa, nt_int)

    cl_pitch_2    = response(h_cl_theta, aoa_2, nt_int)
    cm_pitch_2    = response(h_cm_theta, aoa_2, nt_int)

    cl_pitch_3    = response(h_cl_theta, aoa_3, nt_int)
    cm_pitch_3    = response(h_cm_theta, aoa_3, nt_int)

    # --- 3. Visualization ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # Subplot 1: Input AoA
    ax[0].plot(time_test, aoa, label="AOA")
    ax[0].set_xlabel(r'$\tau\,[\,]$')
    ax[0].set_ylabel(r'$w\, [\,]$')
    ax[0].legend()
    ax[0].grid(True)

    # Subplot 2: Cl Components
    ax[1].plot(time_test, cl_pitch, label="Cl PITCH")
    ax[1].plot(time_test, cl_pitch_2, label="Cl PITCH 2")
    ax[1].plot(time_test, cl_pitch_3, label="Cl PITCH 3")
    ax[1].plot(time_test, cl_heave, label="Cl HEAVE")
    ax[1].plot(time_test, cl_thetadot, label="Cl Thetadot")
    ax[1].set_xlabel(r'$\tau\,[\,]$')
    ax[1].set_ylabel(r'$Cl\, [\,]$')
    ax[1].legend()
    ax[1].grid(True)

    # Subplot 3: Cm Components
    ax[2].plot(time_test, cm_pitch, label="Cm PITCH")
    ax[2].plot(time_test, cm_pitch_2, label="Cm PITCH 2")
    ax[2].plot(time_test, cm_pitch_3, label="Cm PITCH 3")
    ax[2].plot(time_test, cm_heave, label="Cm HEAVE")
    ax[2].plot(time_test, cm_thetadot, label="Cm Thetadot")
    ax[2].set_xlabel(r'$\tau\,[\,]$')
    ax[2].set_ylabel(r'$Cm\, [\,]$')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Optional: Return data for further analysis
    return time_test, aoa
    

def response(kernels, aoa, nt):
    ndof = kernels.shape[0] 
    
    AA = np.diag(np.ones(ndof-1),-1)
    BB = np.zeros((ndof))
    BB[0] = 1.
    CC = kernels
    
    q = np.zeros((ndof, nt))
    cn = np.zeros(nt)
    
    for it in range(1,nt):
        q[:,it] = AA@q[:,it-1] + BB*aoa[it] 
        cn[it] = CC@q[:,it]    
    
    return cn 
    
AOA = 5
MACH = 80
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_linear_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_test_2" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_verylarge" % AOA
home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine" % AOA
home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_surfaceprint_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_3" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_2" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/M0%s_veryfine" % MACH
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_veryfine" % AOA
filename = home_pitch + "/history_05_00002.dat"    

chord = 0.4064 #1. 
v_inf = 123. * MACH / 80 # * 277.24 # 123. 
print("V freestream", v_inf)

# mach = 0.8
# v_inf = mach * 10 / 8 * 277.24 # 123. 
# chord = 1. 

tref = chord / (2 * v_inf)
subs = 48 
# subs = 76 

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

print(cl_output.shape,it_05.shape,it_05[0],it_05[-1])

"""
cl_0 and cm_0 values 
AOA.    Cl          Cd          Cm
0       0.477260|    0.026601|    0.032471
1       0.669071|    0.041779|   -0.004728
2       0.742936|    0.058152|   -0.032949|
3       0.867393|    0.079169|   -0.062860
4       0.942390|    0.101759|   -0.081949|
5       1.042210|    0.126311|   -0.102394
6       1.116924|    0.155741|   -0.116118|

"""

cl0 = 0.942390
cm0 = -0.081949

# cm_output = -cm_output + 0.25*cl_output

ntsteps = len(cl_output)

# filter 
from scipy import signal

def butter_filter(data, btype, cutoff, fs, order=4):
    """
    btype   : 'low', 'high', or 'band'
    cutoff  : scalar for low/high, [low, high] for band  [Hz]
    fs      : sampling frequency [Hz]
    order   : filter order (4 is a safe default)
 
    Uses second-order sections (sosfilt) for numerical stability —
    always prefer this over ba-form for order > 2.
    """
    nyq = 0.5 * fs
    if btype == 'band':
        Wn = [cutoff[0] / nyq, cutoff[1] / nyq]
    else:
        Wn = cutoff / nyq
 
    sos = signal.butter(order, Wn, btype=btype, output='sos')
    return signal.sosfiltfilt(sos, data)   # zero-phase (forward + backward)


cl_output = butter_filter(cl_output, 'low', cutoff = 120., fs = 1./dt)
cm_output = butter_filter(cm_output, 'low', cutoff = 120., fs = 1./dt)


# test 

target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.849, 3.6304, 5.30]) 
target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226, 3.43, 5.436])

# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.85 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.08)

# veryfine 

# target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.8849]) 
# target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.95 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.065)

# verylarge
# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 1.90 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.130)

# test 2

# target_freqs_pitching = 20.9 / (2.*np.pi) * np.array([1.0, 1.849, 3.6304, 5.30]) 
# target_freqs_plunging = 14.1 / (2.*np.pi) * np.array([1.0, 1.9226, 3.43, 5.436])

# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 1.20 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.12)


# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 1.00 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.10)

norm_factor = 20. 

zdot_input = np.gradient(plunge_input,tt_05)/v_inf * norm_factor 
thetadot_input = np.gradient(pitch_input, tt_05) /v_inf * norm_factor 
pitch_input *= norm_factor 

# ============================================================================
# cp values 
# ============================================================================

files = sorted(glob(home_pitch + "/surface_*.vtu"))

Cp_list = []
x_list = []
y_list = []

for f in tqdm(files, desc="Reading SU2 files"):

    mesh = pv.read(f)

    Cp_list.append(mesh['Pressure_Coefficient'])

    x_list.append(mesh.points[:,0])
    y_list.append(mesh.points[:,1])

Cp = np.array(Cp_list)
x  = np.array(x_list)
y  = np.array(y_list)

X = Cp.copy()

Cp_mean = Cp.mean(axis=0)
X = Cp - Cp_mean

U, S, Vt = np.linalg.svd(X, full_matrices=False)
Phi = Vt.T
print("SVD complete")
#

# Use ALL modes
A_full = X @ Phi
Cp_full = Cp_mean + A_full @ Phi.T

error_full = np.linalg.norm(Cp - Cp_full) / np.linalg.norm(Cp)

print("Cp shape:", Cp.shape)
print("Phi shape:", Phi.shape)
print("A_full shape:", A_full.shape)
print("Full-rank reconstruction error:", error_full)
# 

energy = S**2
energy_fraction = energy / np.sum(energy)
cumulative_energy = np.cumsum(energy_fraction)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(
    np.arange(1, len(cumulative_energy)+1),
    100*cumulative_energy,
    '-o'
)

plt.xlabel('Number of modes')
plt.ylabel('Captured energy [%]')
plt.grid(True)

plt.show()

# plot modes 

r = 6

x_ref = x[0, :]

plt.figure(figsize=(8, 5))

for j in range(r):
    plt.plot(x_ref, Phi[:, j], label=f"Mode {j+1}")

plt.xlabel("x")
plt.ylabel("POD mode")
plt.grid(True)
plt.legend()
plt.show()

# for i in range(Cp.shape[0]):
#     plt.plot(x[i,:], Cp[i,:])
# plt.show()

# for i in range(Cp.shape[0]):
#     plt.plot(x[i,:], y[i,:])
# plt.show()

# spot check 

Phi_r = Phi[:, :r]

A_modes = X @ Phi_r

Cp_r = Cp_mean + A_modes @ Phi_r.T

print("Cp:     ", Cp.shape)
print("Phi_r:  ", Phi_r.shape)
print("A:      ", A_modes.shape)
print("Cp_r:   ", Cp_r.shape)

print("t:      ", t.shape)
print("pitch_input:   ", pitch_input.shape)

i = 7300

x_ref = x[i, :]

plt.figure()
plt.plot(x_ref, Cp[i, :], label="CFD")
plt.plot(x_ref, Cp_r[i, :], "--", label=f"POD ({r} modes)")

plt.xlabel("x")
plt.ylabel("$C_p$")
plt.legend()
plt.grid()
plt.show()

# time histories modes coeffs

plt.figure(figsize=(8, 5))

for j in range(r):
    plt.plot(A_modes[:, j], label=f"Mode {j+1}")

plt.xlabel("x")
plt.ylabel("POD mode")
plt.grid(True)
plt.legend()
plt.show()


# ============================================================================
# ID
# ============================================================================



# n_discard = 200 
# ntsteps -= n_discard
# cl_output = cl_output[n_discard:]
# cm_output = cm_output[n_discard:]
# zdot_input = zdot_input[n_discard:]
# thetadot_input = thetadot_input[n_discard:]
# pitch_input = pitch_input[n_discard:]


# -----
# m = 240 #180 #180 #160 #80 #80
m = 96 #320 

# cl0 = cl_output[0]
# cm0 = cm_output[0]

# detrend

cl_output -= cl0
cm_output -= cm0 

cl_avg = np.average(cl_output)
cm_avg = np.average(cm_output)

cl_output -= cl_avg
cm_output -= cm_avg 

A_pitch = np.zeros((ntsteps,m))
A_thetadot = np.zeros((ntsteps,m))
A_zdot = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn] 
    A_thetadot[icolumn:,icolumn] = thetadot_input[:ntsteps-icolumn] 
    A_zdot[icolumn:,icolumn] = zdot_input[:ntsteps-icolumn] 

A = np.hstack( (A_pitch, A_zdot))

A_pitch2 = A_pitch*A_pitch
# A_pitch2 = np.abs(A_pitch)*A_pitch
A_zdot2 = A_zdot*A_zdot
#A_zdot2 = np.abs(A_zdot)*A_zdot
A_pitch_zdot = A_pitch*A_zdot

A2 = np.hstack( (A_pitch2,A_zdot2,A_pitch_zdot))

# third order
A_pitch3 = A_pitch*A_pitch*A_pitch
A_zdot3 = A_zdot*A_zdot*A_zdot
A_pitch_zdot2 = A_pitch*A_zdot2
A_zdot_pitch2 = A_zdot*A_pitch2

A3 = np.hstack( (A_pitch3,A_zdot3,A_pitch_zdot2, A_zdot_pitch2))

# fourth order
A_pitch4 = A_pitch*A_pitch*A_pitch*A_pitch
A_zdot4 = A_zdot*A_zdot*A_zdot*A_zdot
A_pitch2_zdot2 = A_pitch*A_pitch*A_zdot*A_zdot
A_pitch_zdot3 = A_pitch*A_zdot3
A_pitch3_zdot = A_zdot*A_pitch3

A4 = np.hstack( (A_pitch4,A_zdot4,A_pitch2_zdot2, A_pitch3_zdot,A_pitch_zdot3))

# TEMP
cl_norm_factor = 4. #4. #5. #5.
cm_norm_factor = 30. #30. # 20. 

delta_cl_norm_factor = 1. #30. 
delta_cm_norm_factor = 1. #30. 

cl_output *= cl_norm_factor
cm_output *= cm_norm_factor 

# A_123 = np.hstack( (A, A2, A3) )
A_123 = np.hstack( (A, A2, A3, A4) )

A_123 = A * 1.0

# Example with your data
h, prob = run_volterra_identification(
    A=A_123,           # Your A matrix
    cl=cl_output,         # Your cl vector
    cm=cm_output,         # Your cm vector
    m=m,          # Your memory length
    use_laguerre=True,  # or True for basis expansion
    n_basis=16,  # Fewer parameters than direct (8 vs 20 per order)
    alpha_laguerre=0.36,
    lambda_smooth=0.000,
    lambda_decay=0.0625, # 0.1
    lambda_ridge=0.000, # 0.01
    decay_rate= 9.0, #5.0,
    #optimizer='IPOPT',
    optimizer='SLSQP',
    tol = 1e-10, 
    max_iter= 12500,
    verbose=True
)

print("Kernels in h:")

for name, kernel in h.items():
    print(
        f"{name}: "
        f"shape = {kernel.shape}, "
        f"min = {kernel.min():.6g}, "
        f"max = {kernel.max():.6g}, "
        f"norm = {np.linalg.norm(kernel):.6g}"
    )
    
# Access results
print(f"Identified h_cl shape: {h['cl'].shape}")
print(f"Identified h_cm shape: {h['cm'].shape}")

h_cl = h['cl']
h_cm = h['cm']
# h_mode_1 = h['mode_1']
# h_mode_2 = h['mode_2']

h_cl_1 = h_cl[:2*m]
h_cm_1 = h_cm[:2*m]

cl_check = A_123@h_cl / cl_norm_factor
cm_check = A_123@h_cm / cm_norm_factor

# mode_1_check = A_123@h_mode_1
# mode_2_check = A_123@h_mode_2

# h_cl_2 = h_cl[2*m:5*m]
# h_cm_2 = h_cm[2*m:5*m]

# h_cl_3 = h_cl[5*m:9*m]
# h_cm_3 = h_cm[5*m:9*m]

# h_cl_4 = h_cl[9*m:]
# h_cm_4 = h_cm[9*m:]


# Visualize specific kernel orders

# m = 50
order = 0  # First order
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(h_cl_1[order*m:(order+1)*m*3], 'b.-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(h_cm_1[order*m:(order+1)*m*3], 'r.-', linewidth=2)
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cm - Order {order}')
plt.grid(True)

plt.tight_layout()
plt.show()

# modes??? 
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(mode_1_check, 'b.-', linewidth=1)
# plt.plot(A_modes[:,0])
# plt.xlabel('Lag index')
# plt.ylabel('Coefficient')
# plt.title(f'h_cl - Order {order}')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(mode_2_check, 'r.-', linewidth=1)
# plt.plot(A_modes[:,1])
# plt.xlabel('Lag index')
# plt.ylabel('Coefficient')
# plt.title(f'h_cm - Order {order}')
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# #
# cl_check = (A@h_cl_1)  / cl_norm_factor
# cm_check = (A@h_cm_1)  / cm_norm_factor

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cl_output / cl_norm_factor, 'b-', linewidth=1)
plt.plot(cl_check, 'k-', linewidth=1)
plt.xlabel(f't')
plt.ylabel(f'C_l')
plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cm_output / cm_norm_factor, 'b-', linewidth=1)
plt.plot(cm_check, 'k-', linewidth=1)
plt.xlabel(f't')
plt.ylabel(f'C_m')
plt.title(f'h_cm - Order {order}')
plt.grid(True)

plt.tight_layout()
plt.show()

# temp


# # Example with your data
# h, prob = run_volterra_identification(
#     A=A_123,           # Your A matrix
#     cl=A_modes[:,4],         # Your cl vector
#     cm=A_modes[:,5],         # Your cm vector
#     m=m,          # Your memory length
#     use_laguerre=True,  # or True for basis expansion
#     n_basis=16,  # Fewer parameters than direct (8 vs 20 per order)
#     alpha_laguerre=0.36,
#     lambda_smooth=0.000,
#     lambda_decay=0.0625, # 0.1
#     lambda_ridge=0.000, # 0.01
#     decay_rate= 9.0, #5.0,
#     #optimizer='IPOPT',
#     optimizer='SLSQP',
#     tol = 1e-10, 
#     max_iter= 12500,
#     verbose=True
# )

A_123 = np.hstack( (A, A2, A3, A4) )

U, S, Vt = np.linalg.svd(A_123, full_matrices=False)
h_mode_5 = Vt.T @ (np.diag(1/S) @ (U.T @ A_modes[:,4]))

U, S, Vt = np.linalg.svd(A_123, full_matrices=False)
h_mode_6 = Vt.T @ (np.diag(1/S) @ (U.T @ A_modes[:,5]))


mode_5_check = A_123@h_mode_5 
mode_6_check = A_123@h_mode_6

# Visualize specific kernel orders

# m = 50
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mode_5_check, 'b.-', linewidth=1)
plt.plot(A_modes[:,4])
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(mode_6_check, 'r.-', linewidth=1)
plt.plot(A_modes[:,5])
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cm - Order {order}')
plt.grid(True)

plt.tight_layout()
plt.show()


# #
# cl_check = (A@h_cl_1)  / cl_norm_factor
# cm_check = (A@h_cm_1)  / cm_norm_factor








# Visualize specific kernel orders

# m = 50
order = 0  # First order
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(h_cl_1, 'r-', linewidth=1,label="1st order")
plt.plot(h_cl_2, 'b-', linewidth=1,label="2nd order")
plt.plot(h_cl_3, 'g-', linewidth=1,label="3rd order")
plt.plot(h_cl_4, 'k-', linewidth=1,label="4th order")
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.legend()
plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(h_cm_1, 'r-', linewidth=1,label="1st order")
plt.plot(h_cm_2, 'b-', linewidth=1,label="2nd order")
plt.plot(h_cm_3, 'g-', linewidth=1, label="3rd order")
plt.plot(h_cm_4, 'k-', linewidth=1, label="4th order")
plt.xlabel('Lag index')
plt.ylabel('Coefficient')
plt.title(f'h_cm - Order {order}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

h_cl_extended = np.zeros(3*m)
h_cl_extended[:m] = h_cl_1[:m]
h_cl_extended[2*m:] = h_cl_1[m:]

h_cm_extended = np.zeros(3*m)
h_cm_extended[:m] = h_cm_1[:m]
h_cm_extended[2*m:] = h_cm_1[m:]

h_cl_2_extended = np.zeros(6*m)
h_cl_2_extended[:m] = h_cl_2[:m]
h_cl_2_extended[2*m:3*m] = h_cl_2[m:2*m]
h_cl_2_extended[3*m:4*m] = h_cl_2[2*m:]

h_cm_2_extended = np.zeros(6*m)
h_cm_2_extended[:m] = h_cm_2[:m]
h_cm_2_extended[2*m:3*m] = h_cm_2[m:2*m]
h_cm_2_extended[3*m:4*m] = h_cm_2[2*m:]

