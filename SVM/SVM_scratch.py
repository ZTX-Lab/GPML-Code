import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from cvxopt import matrix, solvers

def gen_linear_data(n_samples=100):
    x, y = make_blobs(n_samples=n_samples, centers=2, random_state=6, cluster_std=1.2)
    y = np.where(y == 0, -1, 1).astype(float)
    return x, y

def get_nonlinear_data(n_samples=100, noise=0.1):
    x, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=6)
    y = np.where(y == 0, -1, 1).astype(float)
    return x, y

def print_figure(x1, x2, y1, y2):
    plt.subplot(1, 2, 1)
    plt.scatter(x1[:, 0], x1[:, 1], c=y1, cmap='bwr', edgecolors='k')
    plt.title('Linear Data')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.scatter(x2[:, 0], x2[:, 1], c=y2, cmap='bwr', edgecolors='k')
    plt.title('Non-linear Data')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()

def print_margin_boundary(w, b, x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', edgecolors='k')

    # Create a grid to plot decision boundary
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Calculate decision boundary
    Z = grid.dot(w) + b
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='k', linestyles='dashed')

    plt.title('SVM Decision Boundary with Margins')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def print_dual_boundary(model, x, y, resolution=100):
    """
    Visualizes the Non-Linear Decision Boundary learned by the Dual SVM.
    
    [Mechanism]
    Unlike the Linear Primal SVM, we cannot simply use 'w.x + b'.
    Instead, we evaluate the decision function using the Kernel Trick:
      f(z) = \sum (alpha_i * y_i * K(x_i, z)) + b
    
    This allows us to draw contours for complex non-linear boundaries.
    """
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', edgecolors='k')

    # 1. Create a meshgrid to cover the feature space
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), 
                         np.linspace(y_min, y_max, resolution))
    
    # Flatten grid for iteration
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 2. Calculate Decision Function Value for each grid point
    # We evaluate f(z) for the visualization grid.
    Z_val = []
    for point in grid_points:
        val = 0
        # Sum over Support Vectors
        for i in range(len(model['sv_alphas'])):
            val += model['sv_alphas'][i] * model['sv_y'][i] * model['k_func'](model['sv_X'][i], point)
        Z_val.append(val + model['b'])
    
    Z = np.array(Z_val).reshape(xx.shape)

    # 3. Plot Contours (Decision Boundary & Margins)
    # Level 0: The Decision Boundary
    # Level -1, 1: The Margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    # 4. Highlight Support Vectors
    # Draw a circle around the support vectors to distinguish them
    plt.scatter(model['sv_X'][:, 0], model['sv_X'][:, 1], 
                s=200, linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.title('Dual SVM Decision Boundary (Non-Linear)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.show()


"""
Overview: SVM Primal Solver Implementations

1. Limitation: Linearity (Linear Boundary Only)
   Primal solvers inherently produce a straight, linear hyperplane. 
   They cannot capture non-linear patterns (such as concentric circles) unless explicit feature engineering is applied.
   Even with Soft Margin, which allows for outliers, the decision boundary itself remains linear.

2. Mechanism: Direct Optimization
   These solvers compute the optimal weight vector (w) and bias term (b) directly in the primal feature space.

3. Constraint: Incompatibility with Kernel Methods
   Primal formulations assume we can explicitly calculate and store 'w'. 
   However, in Kernel methods (e.g., RBF), the data is mapped to an infinite-dimensional space where 'w' cannot be explicitly computed. 
   Thus, Primal Solvers cannot effectively utilize the "Kernel Trick."

4. Computational Complexity: Dimensionality Sensitivity
   The computational cost scales with the feature dimension (d). 
   This makes primal solvers computationally expensive for high-dimensional data (d >> n), whereas Dual solvers scale with the number of samples (n).

Available Implementations:
1. Hard Margin QP Solver (solve_primal_qpsolver) - Uses `cvxopt` (Strict separation)
2. Soft Margin QP Solver (solve_primal_soft_margin_qpsolver) - Uses `cvxopt` (Robust to noise)
3. Gradient Descent Solver (solve_primal_gradient) - Iterative optimization
4. Newton's Method Solver (solve_primal_newton_vectorized) - Fast converging iterative method
"""
# 1. Hard Margin QP Solver using cvxopt
def solve_primal_qpsolver(x,y):
    n_samples, n_features = x.shape


    # ---------------------------------------------------------
    # Mathematical Derivation: Hard Margin SVM
    # ---------------------------------------------------------
    # 1. Decision Boundary Definition: 
    #    The hyperplane is defined by the equation: w.x + b = 0
    #
    # 2. Classification Constraint: 
    #    For correct classification, all samples must lie on the correct side of the boundary:
    #    y_i (w.x_i + b) >= 1 for all i.
    #
    # 3. Geometric Margin Derivation: 
    #    We aim to maximize the distance (margin) between the boundary and the nearest data points.
    #    Formula: Margin = |w.x + b| / ||w||
    #
    #    [Step-by-Step Geometric Proof]
    #    0. Goal: Find w and b that define the optimal separating hyperplane.
    #    1. Let P be an arbitrary sample point, and Q be its orthogonal projection onto the decision boundary.
    #    2. The vector connecting P and Q represents the shortest distance.
    #    3. The direction of this vector is given by the unit normal vector: w / ||w||.
    #    4. The distance margin is the projection of vector (P-Q) onto the normal vector:
    #       Margin = | (P - Q) . (w / ||w||) | = | (P.w - Q.w) | / ||w||
    #    5. Since Q lies on the decision boundary, it satisfies w.Q + b = 0, which implies w.Q = -b.
    #       Substituting this back: Margin = | P.w - (-b) | / ||w|| = | P.w + b | / ||w||
    #    6. For Support Vectors (the closest points), we enforce the constraint: y(P.w + b) = 1.
    #       Therefore, the geometric margin simplifies to: 1 / ||w||.
    #
    #    Conclusion: To MAXIMIZE the margin (1 / ||w||), we must MINIMIZE ||w||.
    #    Mathematically, this is equivalent to minimizing 1/2 * ||w||^2 (for convex optimization convenience).
    
    # ---------------------------------------------------------
    # 1. Configuration of QP Optimization Variables
    # ---------------------------------------------------------
    # The optimization variable vector 'u' concatenates weights and bias:
    # u = [w_1, w_2, ..., w_n, b] 
    
    # ---------------------------------------------------------
    # 2. Objective Function Construction (Matrix P, Vector q)
    # ---------------------------------------------------------
    # Standard QP Form: Minimize (1/2 * u.T * P * u + q.T * u)
    # Goal: Minimize ||w||^2. We do not regularize (penalize) the bias term 'b'.

    P_np = np.eye(n_features + 1)
    P_np[n_features, n_features] = 0.0 # Do not penalize bias 'b'
    P = matrix(P_np)
    
    q = matrix(np.zeros(n_features + 1)) # No linear term in objective

# ---------------------------------------------------------
    # 3. Inequality Constraints Construction (Matrix G, Vector h)
    # ---------------------------------------------------------
    # Standard QP Form: G * u <= h
    # SVM Constraint:   y_i(w.x_i + b) >= 1
    # Transformation:  -y_i(w.x_i + b) <= -1  (Multiplying by -1 to match QP form)
    #
    # Coefficients Breakdown:
    # - For w part: -y_i * x_i
    # - For b part: -y_i
    # G = -y*[x1, x2, 1], h = -1

    G = np.zeros((n_samples, n_features + 1))

    for i in range(n_samples):
        # Assign coefficients for weights (w): -y_i * x_i
        G[i, :n_features] = -1 * y[i] * x[i]
        # Assign coefficient for bias (b): -y_i
        G[i, n_features] = -1 * y[i]
    
    G = matrix(G)
    h = matrix(-1.0 * np.ones(n_samples)) # RHS is -1 vector

    # ---------------------------------------------------------
    # 4. Solve QP using cvxopt solver
    # ---------------------------------------------------------
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h)


# ---------------------------------------------------------
    # 5. Extract Optimal Parameters (w, b)
    # ---------------------------------------------------------
    # The solution vector contains [w_1, ..., w_d, b]. We slice it to separate w and b.
    w = np.array(solution['x'][:n_features]).flatten()
    b = np.array(solution['x'][n_features]).flatten()[0]

    return w, b


# 2. Soft Margin QP Solver using cvxopt
def solve_primal_soft_margin_qpsolver(x, y, C=1.0):
    n_samples, n_features = x.shape

    # ---------------------------------------------------------
    # Theoretical Background: Transition to Soft Margin
    # ---------------------------------------------------------
    # 1. Limitation of Hard Margin:
    #    The formulation requires linear separability. If the data has noise or overlap,
    #    the constraints become infeasible (no solution exists).
    #
    # 2. Introduction of Slack Variables (xi):
    #    We relax the strict constraints by introducing a non-negative slack variable 
    #    xi_i for each sample. This allows data points to violate the margin.
    #    - xi_i = 0: Correctly classified, on the correct side of the margin.
    #    - 0 < xi_i <= 1: Correctly classified, but within the margin.
    #    - xi_i > 1: Misclassified.
    #
    # 3. Optimization Variable Expansion:
    #    - Hard Margin: u = [w_1, ..., w_d, b]      (Size: features + 1)
    #    - Soft Margin: u = [w..., b, xi_1, ..., xi_n] (Size: features + 1 + samples)
    #
    # Remark on QP Formulation:
    # Unlike Gradient Descent which handles the Hinge Loss implicitly (via max(0, ...)),
    # QP solvers require explicit variables to represent the piecewise linear loss function.

    n_vars = n_features + 1 + n_samples # Total number of optimization variables

    # ---------------------------------------------------------
    # 1. Objective Function Construction (Matrix P, Vector q)
    # ---------------------------------------------------------
    # Primal Objective: Minimize 1/2 * ||w||^2 + C * sum(xi_i)
    # - Term 1 (Regularization): Maximizes the margin (minimizes complexity).
    # - Term 2 (Loss): Penalizes margin violations.
    # - Parameter C: Controls the trade-off. Large C -> Hard Margin behavior.
    
    # [Matrix P]: Quadratic Term
    # Applies only to weight vector 'w'. Bias 'b' and slacks 'xi' have 0 curvature penalty.


    P_np = np.zeros((n_vars, n_vars))
    P_np[0:n_features, 0:n_features] = np.eye(n_features)
    P = matrix(P_np)


    # [Vector q] Linear terms (only for xi)
    # ---------------------------------------------------------
    # The parameter C acts as a penalty weight for slack variables.
    #
    # [CRITICAL DISTINCTION: Primal C vs. Dual Alpha]
    # 1. Primal Form (Static Penalty):
    #    - C is a STATIC CONSTANT applied directly in the objective function.
    #    - The objective is explicit: Minimize ... + C * sum(xi).
    #    - The solver does not optimize C; it optimizes xi based on the fixed penalty C.
    #
    # 2. Dual Form (Dynamic Multipliers):
    #    - There is no 'xi'. Instead, the solver finds DYNAMIC variables 'alpha' (Lagrange multipliers).
    #    - In the Dual, C appears as a 'Box Constraint' (0 <= alpha <= C).
    #    - It limits the maximum influence (alpha) a single data point can have.
    #
    # Conclusion: 
    # The Primal form accumulates penalty explicitly for each slack variable using static C.
    # The Dual form balances constraints implicitly using dynamic alphas bounded by C.
    # ---------------------------------------------------------


    q_np = np.zeros(n_vars)
    q_np[n_features + 1:] = C 
    q = matrix(q_np)

    # ---------------------------------------------------------
    # 2. Inequality Constraints Construction (Matrix G, Vector h)
    # ---------------------------------------------------------
    # We enforce two sets of constraints for every sample i:
    #
    # 1. Margin Constraint with Slack: 
    #    y_i(w.x_i + b) >= 1 - xi_i
    #    Standard Form (<=): -y_i(w.x_i + b) - xi_i <= -1
    #
    # 2. Non-Negativity of Slack:
    #    xi_i >= 0
    #    Standard Form (<=): -xi_i <= 0
    
    G_np = np.zeros((2 * n_samples, n_vars))
    h_np = np.zeros(2 * n_samples)

    # [Constraint Set 1] Margin constraints
    for i in range(n_samples):
        # Coefficients for w: -y_i * x_i
        G_np[i, :n_features] = -1 * y[i] * x[i]
        # Coefficient for b: -y_i
        G_np[i, n_features] = -1 * y[i]
        # Coefficient for xi_i: -1 (Moved from RHS to LHS)
        G_np[i, n_features + 1 + i] = -1.0
        # RHS constant: -1
        h_np[i] = -1.0

    # [Constraint Set 2] Slack positivity constraints
    for i in range(n_samples):
        row_idx = n_samples + i 
        # Coefficient for xi_i: -1
        G_np[row_idx, n_features + 1 + i] = -1.0
        # RHS constant: 0
        h_np[row_idx] = 0.0

    G = matrix(G_np)
    h = matrix(h_np)

    # ---------------------------------------------------------
    # 3. Solve QP using cvxopt solver
    # ---------------------------------------------------------
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h)

    # ---------------------------------------------------------
    # 4. Extract w and b from solution
    # ---------------------------------------------------------
    u_opt = np.array(solution['x']).flatten()
    
    # Slice the result vector to get w and b
    w = u_opt[:n_features]
    b = u_opt[n_features]
    
    # (Optional) Extract slack variables for analysis
    # xi = u_opt[n_features+1:]
    
    return w, b




# 3. Gradient Descent Method (Manual Implementation) (update all samples at once) (mini batch can be implemented similarly)
def solve_primal_gradient(x, y, learning_rate=0.001, n_iters=1000, C=1.0):
    """
    Solves Soft Margin SVM using Batch Gradient Descent.
    
    [Optimization Strategy: Unconstrained Formulation]
    Instead of solving the Constrained QP problem directly, we convert it into 
    an unconstrained optimization problem by minimizing the Primal Objective Function:
    
      J(w, b) = (1/2)||w||^2  +  C * \sum_{i=1}^{N} max(0, 1 - y_i(w \cdot x_i + b))
      
      - Term 1: L2 Regularization (Maximizes Margin / Prevents Overfitting).
      - Term 2: Hinge Loss (Penalizes Margin Violations).
    
    [Key Concept: Hinge Loss & Sub-gradients]
    The Hinge Loss function, L(z) = max(0, 1-z), is continuous but not differentiable 
    at z=1 (the 'kink'). Therefore, strictly speaking, we calculate the 'sub-gradient'.
    
    - If z < 1 (Violation): Gradient is -y * x
    - If z > 1 (Correct & Safe): Gradient is 0
    - If z = 1 (Exactly on Margin): We choose the gradient to be -y * x (or 0) by convention.
    """


    n_samples, n_features = x.shape
    
    # Initialize weights and bias (Zero initialization is common for convex problems)
    w = np.zeros(n_features)
    b = 0.0
    
    for epoch in range(n_iters):
        # ---------------------------------------------------------
        # 1. Forward Pass: Compute Linear Activation
        # ---------------------------------------------------------
        # Calculate the decision function value (score) for all samples.
        decision = np.dot(x, w) + b

        # ---------------------------------------------------------
        # 2. Compute Functional Margin Deficit (Shortfall)
        # ---------------------------------------------------------
        # 'Shortfall' measures how much a sample fails to meet the margin requirement.
        # Condition: y_i(w.x + b) >= 1
        # Shortfall = 1 - y_i(w.x + b)
        shortfall = 1 - y * decision

        # Identify the 'Active Set' (Samples contributing to the gradient)
        # - Misclassified points (Shortfall > 1)
        # - Points inside the margin (0 < Shortfall < 1)
        # - Points exactly on the margin (Shortfall = 0, depending on implementation)
        # The mask is True if the sample contributes to the Loss (Hinge Loss > 0).
        mask = shortfall > 0
        
        # ---------------------------------------------------------
        # 3. Calculate Gradients (Vectorized Sub-gradient)
        # ---------------------------------------------------------
        # The gradient of the Objective Function J(w, b) is:
        # \nabla_w J = w + C * \sum (-y_i * x_i)  [Summed only over violating samples]
        
        # Filter labels: Keep y_i for active samples, set to 0 for safe samples.
        y_masked = y * mask
      
        # Gradient component from Hinge Loss (Data term)
        # Matrix multiplication (X.T dot Y) efficiently computes the sum over samples.
        grad_hinge_w = -C * np.dot(x.T, y_masked)
        grad_hinge_b = -C * np.sum(y_masked)
      
        # Total Gradient: Regularization Gradient + Loss Gradient
        #
        # [Crucial Implementation Detail: Gradient Scaling]
        # We divide the loss gradient by 'n_samples'.
        # This effectively optimizes the 'Mean Loss' instead of 'Sum Loss'.
        # Benefit: The learning rate becomes independent of the dataset size (N).
        # Without this, larger datasets would require incredibly small learning rates to prevent explosion.
        dw = w + (grad_hinge_w / n_samples)
        db = grad_hinge_b / n_samples
      
        # ---------------------------------------------------------
        # 4. Parameter Update Step (Descent Direction)
        # ---------------------------------------------------------
        # Move parameters in the opposite direction of the gradient.
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


# 4. Vectorized Newton's Method
def solve_primal_newton_vectorized(x, y, n_iters=20, C=1.0):
    """
    Solves Primal SVM using Newton's Method (Newton-Raphson).
    
    [Why Newton's Method?]
    Standard Gradient Descent relies only on the 1st derivative (Slope).
    Newton's Method utilizes both the 1st (Slope) and 2nd derivative (Curvature/Hessian).
    By approximating the loss function as a quadratic bowl, it can jump directly 
    towards the minimum, achieving Quadratic Convergence (extremely fast).

    [Objective Function Modification: Squared Hinge Loss]
    The standard Hinge Loss `max(0, 1-z)` is not differentiable at z=1 (the 'kink').
    Consequently, its Hessian (2nd derivative) is undefined.
    To apply Newton's Method, we adopt the **Squared Hinge Loss**: `max(0, 1-z)^2`.
    This function is differentiable everywhere and strictly convex.
    
    J(u) = (1/2)||w||^2  +  C * \sum_{i=1}^{N} max(0, 1 - y_i(u \cdot x_i))^2
    """
    n_samples, n_features = x.shape
    
    # ---------------------------------------------------------
    # 1. Variable & Data Setup for Vectorization
    # ---------------------------------------------------------
    # Combine parameters w and b into a single vector 'u' = [w_1, ..., w_d, b].
    u = np.zeros(n_features + 1)
    
    # Augment feature matrix X with a column of 1s to handle bias 'b'.
    # X_bias = [[x1, 1], [x2, 1], ...]
    # This enables computing 'w.x + b' as a single matrix operation: X_bias.dot(u)
    X_bias = np.hstack([x, np.ones((n_samples, 1))])
    
    # Create Identity Matrix for Regularization Term (1/2 * ||w||^2).
    # We must NOT regularize the bias term 'b', so the last diagonal element is 0.
    I_reg = np.eye(n_features + 1)
    I_reg[n_features, n_features] = 0
    
    for i in range(n_iters):
        # ---------------------------------------------------------
        # 2. Forward Pass & Active Set Identification
        # ---------------------------------------------------------
        # Compute scores: y_pred = u^T x
        scores = np.dot(X_bias, u)
        
        # Calculate 'Shortfall' (Functional Margin Deficit)
        # shortfall = 1 - y * score
        shortfall = 1 - y * scores
        
        # Identify Active Set: Samples that contribute to the loss (Shortfall > 0)
        # Only these samples will affect the Gradient and Hessian.
        mask = shortfall > 0
        
        # ---------------------------------------------------------
        # 3. Calculate Gradient (First Derivative)
        # ---------------------------------------------------------
        # We need the derivative of the Squared Hinge Loss w.r.t u.
        # Chain Rule: d/du (1 - y(u.x))^2 
        #           = 2 * (1 - y(u.x)) * d/du(1 - y(u.x))
        #           = 2 * shortfall * (-y * x)
        #           = -2 * y * shortfall * x
        
        # Element-wise weighting for the gradient sum
        grad_weights = -2 * C * y * shortfall * mask
        
        # Compute the weighted sum of X vectors
        # X.T dot weights performs summation: sum( weight_i * x_i )
        grad_loss = np.dot(X_bias.T, grad_weights)
        
        # Total Gradient = Regularization Gradient (u) + Loss Gradient
        gradient = np.dot(I_reg, u) + grad_loss
        
        # ---------------------------------------------------------
        # 4. Calculate Hessian (2nd Derivative)
        # ---------------------------------------------------------
        # [Conceptual Understanding: The Role of Hessian]
        # The 2nd Derivative (Hessian) provides 'curvature' information for Newton's Method.
        # It tells us exactly how the gradient changes as we move in the parameter space.
        # Unlike Gradient Descent which uses a fixed step size (learning rate), 
        # Newton's Method uses this curvature to dynamically adjust the step size,
        # taking larger steps in flat regions and smaller steps in steep regions.
        
        # [Mathematical Derivation: From Gradient to Hessian]
        # We need to differentiate the Gradient vector w.r.t 'u' again.
        #
        # 1. Recall the Gradient Element (for a single active sample i):
        #    g_i(u) = -2 * C * y_i * (1 - y_i * u.x_i) * x_i
        #
        # 2. Differentiate g_i(u) with respect to u:
        #    - The terms (-2 * C * y_i * x_i) are constant coefficients w.r.t u.
        #    - We only need to derive the term in the bracket: (1 - y_i * u.x_i).
        #    - d/du (1 - y_i * u.x_i) = -y_i * x_i
        #
        # 3. Combine terms:
        #    H_i(u) = [Constant Coeffs] * [Derivative of Bracket]
        #           = (-2 * C * y_i * x_i) * (-y_i * x_i^T)
        #           = 2 * C * (y_i)^2 * (x_i * x_i^T)
        #
        # 4. Simplify:
        #    Since labels y_i are {-1, 1}, y_i^2 is always 1.
        #    H_i(u) = 2 * C * x_i * x_i^T
        #
        # [Vectorized Implementation]
        # Instead of summing x_i * x_i^T iteratively, we use matrix multiplication:
        # Sum(x_i * x_i^T) is equivalent to X_active.T @ X_active.
        
        # Select only active samples (mask == True) because for inactive samples,
        # the loss is 0, so the curvature (Hessian) is also 0.
        X_active = X_bias[mask]
        
        # Compute X^T * X for active samples (Sum of Outer Products)
        # This single matrix multiplication replaces the explicit summation loop.
        hessian_loss = 2 * C * np.dot(X_active.T, X_active)
        
        # Total Hessian = Regularization Hessian (I) + Loss Hessian
        hessian = I_reg + hessian_loss
        
        # ---------------------------------------------------------
        # 5. Newton Update Step (Solving Linear System)
        # ---------------------------------------------------------
        # Standard Update: u_new = u_old - Learning_Rate * Gradient
        # Newton Update:   u_new = u_old - Hessian^{-1} * Gradient
        #
        # Instead of explicitly inverting the Hessian (computationally expensive O(d^3) and unstable),
        # we solve the linear system: H * delta = -gradient
        try:
            delta = np.linalg.solve(hessian, -gradient)
            u += delta
        except np.linalg.LinAlgError:
            # Fallback: If Hessian is singular (rare due to regularization), stop early.
            break
            
    # Decompose 'u' back into weight 'w' and bias 'b'
    w = u[:n_features]
    b = u[n_features]
    
    return w, b



# =============================================================================
# DUAL SOLVERS (Kernel Methods for Non-Linear Boundaries)
# =============================================================================
# [Transition Overview: From Primal to Dual]
# So far, we have implemented 4 Primal Solvers. While they vary in optimization strategy 
# (QP vs. Iterative), they share a fundamental limitation:
# -> They can only learn Linear Decision Boundaries.
#
# To capture non-linear patterns (like the 'circles' dataset), we must move to the 
# DUAL FORMULATION. The Dual form allows us to apply the **Kernel Trick**.
# 
# [Why Soft Margin Dual QP?]
# 1. Hard Margin Dual: Rarely used in practice due to extreme sensitivity to outliers.
# 2. Dual Gradient/Newton: Less common because the Dual problem is a 
#    well-structured Quadratic Programming (QP) problem, which specialized solvers 
#    (like cvxopt or libsvm) can handle very efficiently.
#
# Therefore, we will focus on implementing the **Soft Margin Dual QP Solver**.


# ---------------------------------------------------------
# Kernel Functions
# ---------------------------------------------------------
# The Kernel Function K(x, y) computes the inner product in a transformed feature space:
# K(x, y) = < phi(x), phi(y) >
# This allows us to measure similarity in high-dimensional spaces without 
# explicitly computing the transformation phi(x). (The Kernel Trick)

def linear_kernel(x1, x2):
    """
    Standard Dot Product Kernel: K(a, b) = a^T b
    
    - Use Case: When data is linearly separable or feature dimension is already very high (e.g., Text Classification).
    - Result: Equivalent to the Linear Boundary found by Primal Solvers.
    """
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=0.5):
    """
    Radial Basis Function (Gaussian) Kernel:
    K(a, b) = exp( -gamma * ||a - b||^2 )
    
    [Mathematical Interpretation]
    - Implicitly maps data into an **Infinite Dimensional Space** (via Taylor Series expansion of exp).
    - It measures similarity based on Euclidean distance:
      - If 'a' and 'b' are close: ||a-b||^2 -> 0, thus K(a, b) -> 1 (High Similarity)
      - If 'a' and 'b' are far:   ||a-b||^2 -> Large, thus K(a, b) -> 0 (Low Similarity)

    [Hyperparameter Gamma (gamma = 1 / 2*sigma^2)]
    - Gamma controls the 'width' or 'influence radius' of the Gaussian bell curve.
    - Large Gamma: Narrow peak. Only very close points are considered similar.
      -> Effect: Complex, wiggly boundaries. Risk of **Overfitting**.
    - Small Gamma: Broad peak. Distant points still have influence.
      -> Effect: Smooth, almost linear boundaries. Risk of **Underfitting**.
    """
    # Euclidean distance squared: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
    # Computed simply as scalar for single point, or vector for matrices.
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)



"""
===============================================================================
   DUAL SOFT MARGIN SVM: MATHEMATICAL DERIVATION & ALGORITHM
===============================================================================

1. PRIMAL PROBLEM FORMULATION (The Starting Point)
   -----------------------------------------------
   Goal: Find the optimal hyperplane (w, b) that maximizes the geometric margin 
   while minimizing classification errors (slack variables).

   Minimize: J(w, b, xi) = (1/2)||w||^2 + C * \sum(xi)
   Subject to:
      1. y_i (w \cdot x_i + b) >= 1 - xi_i  (Margin constraint with slack)
      2. xi_i >= 0                          (Non-negativity of slack)

2. LAGRANGIAN CONSTRUCTION
   -----------------------
   We incorporate the constraints into the objective function using Lagrange 
   Multipliers: alpha (for margin) and mu (for non-negativity).

   L(w, b, xi, alpha, mu) = 
       [ (1/2)||w||^2 + C * \sum(xi) ]                        <-- Primal Objective
       - \sum( alpha_i * [ y_i(w \cdot x_i + b) - 1 + xi ] )  <-- Margin Penalty
       - \sum( mu_i * xi )                                    <-- Slack Penalty

3. STATIONARITY CONDITIONS (KKT Conditions)
   ----------------------------------------
   To find the optimum, the gradient of the Lagrangian w.r.t. the primal variables 
   (w, b, xi) must be zero.

   A) dL / dw = 0  =>  w = \sum( alpha_i * y_i * x_i )
      [KEY INSIGHT 1]: The optimal weight vector 'w' is a linear combination of the Support Vectors.

   B) dL / db = 0  =>  \sum( alpha_i * y_i ) = 0
      [KEY INSIGHT 2]: The weighted sum of positive and negative labels must balance to zero.

   C) dL / dxi = 0 =>  C - alpha_i - mu_i = 0
      [KEY INSIGHT 3]: alpha_i = C - mu_i. 
      Since mu_i >= 0, this yields the **Box Constraint**: 0 <= alpha_i <= C.

4. DUAL PROBLEM FORMULATION (Substitution)
   ---------------------------------------
   Substituting [KEY INSIGHT 1] back into the Lagrangian allows us to eliminate w, b, and xi.
   Crucially, the dot product (x_i \cdot x_j) is replaced by the Kernel Function K(x_i, x_j).

   Maximize:  Q(alpha) = \sum(alpha_i) - (1/2) \sum \sum ( alpha_i * alpha_j * y_i * y_j * K(x_i, x_j) )
   Subject to:
      1. 0 <= alpha_i <= C       (Box Constraint)
      2. \sum(alpha_i * y_i) = 0 (Equality Constraint)
"""

# ---------------------------------------------------------
# Dual SVM Solver Implementation
# ---------------------------------------------------------
def solve_dual_soft_margin_qpsolver(X, y, C=1.0, kernel='rbf', gamma=0.5):
    n_samples, n_features = X.shape
    
    # Kernel Selection Strategy
    if kernel == 'linear':
        k_func = linear_kernel
    else:
        # Lambda function to fix gamma parameter for RBF
        k_func = lambda x1, x2: rbf_kernel(x1, x2, gamma)

    # -----------------------------------------------------
    # Step 1: Compute Gram Matrix (Kernel Matrix)
    # -----------------------------------------------------
    # The Gram Matrix K contains pairwise similarities: K_ij = K(x_i, x_j).
    # This enables the "Kernel Trick" by replacing explicit feature mapping.
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = k_func(X[i], X[j])

    # -----------------------------------------------------
    # Step 2: Construct QP Matrices (P, q)
    # -----------------------------------------------------
    # Dual Objective: MAXIMIZE Q(alpha).
    # Standard QP Solvers: MINIMIZE (1/2 x.T P x + q.T x).
    #
    # Transformation: Multiply Objective by -1.
    # New Objective: Minimize (1/2) alpha.T (Y.Y.T * K) alpha - \sum(alpha)
    
    # [Matrix P]: Quadratic Coefficient
    # Element P_ij = y_i * y_j * K(x_i, x_j)
    # Computed efficiently via Outer Product of y.
    P_np = np.outer(y, y) * K
    P = matrix(P_np)
    
    # [Vector q]: Linear Coefficient
    # q = [-1, -1, ..., -1] (Because we flipped Max to Min)
    q = matrix(-1.0 * np.ones(n_samples))


    # -----------------------------------------------------
    # Step 3: Inequality Constraints (Box Constraint)
    # -----------------------------------------------------
    # Origin: From dL/dxi = 0, we got 0 <= alpha_i <= C.
    # Meaning: 
    #   - alpha = 0: Correctly classified (Influence is zero).
    #   - 0 < alpha < C: Support Vector (On the margin).
    #   - alpha = C: Outlier or Misclassified (Max influence capped at C).
    
    # QP format: Gx <= h
    # Split 0 <= alpha <= C into:
    #   1. -alpha <= 0
    #   2.  alpha <= C
    
    G_std = -1.0 * np.eye(n_samples)   # Matrix for -alpha <= 0
    G_slack = np.eye(n_samples)        # Matrix for  alpha <= C
    
    # Stack constraints vertically
    G = matrix(np.vstack((G_std, G_slack)))
    # Stack RHS vector: [Zeros... , C, C, ...]
    h = matrix(np.hstack((np.zeros(n_samples), C * np.ones(n_samples))))

    # -----------------------------------------------------
    # Step 4: Equality Constraint
    # -----------------------------------------------------
    # Origin: From dL/db = 0, we got sum(alpha_i * y_i) = 0.
    # Meaning: The "torques" of positive and negative classes must balance out.
    # QP format: Ax = b
    
    A = matrix(y.reshape(1, -1))
    b = matrix(0.0)

    # -----------------------------------------------------
    # Step 5: Solve
    # -----------------------------------------------------
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])

    # -----------------------------------------------------
    # Step 6: Interpret Results & Support Vectors
    # -----------------------------------------------------
    """
    [MATHEMATICAL REASONING: Why do we filter alphas?]
    
    1. KKT Complementary Slackness Condition:
       The KKT conditions state that at the optimal solution, the product of the 
       Lagrange multiplier (alpha) and the constraint must be ZERO.
       
       Formula: alpha_i * [ y_i(w.x_i + b) - 1 + xi_i ] = 0
    
    2. Case A: Non-Support Vectors (Safe Points)
       - These points are correctly classified and far away from the margin.
       - Constraint: y_i(w.x_i + b) > 1 
       - Since the term in the bracket [ ... ] is NOT zero (it's positive),
         alpha_i MUST be ZERO to satisfy the equation.
       -> Conclusion: alpha_i = 0 implies "This point does not affect the model."
       
    3. Case B: Support Vectors (On the Margin or Inside)
       - These points are exactly on the boundary or violating it.
       - The term in the bracket is zero (or handled by slack).
       - Therefore, alpha_i CAN be greater than zero.
       -> Conclusion: alpha_i > 0 implies "This point supports/defines the boundary."
    """
    
    # Filter out numerical noise (e.g., 1e-10) to find true Support Vectors

    sv_threshold = 1e-5
    sv_indices = alphas > sv_threshold
    
    sv_alphas = alphas[sv_indices]
    sv_X = X[sv_indices]
    sv_y = y[sv_indices]
    
    print(f"[{kernel} kernel] Optimization Finished.")
    print(f" - Derivatives satisfied.")
    print(f" - Constraints satisfied.")
    print(f" - Found {len(sv_alphas)} Support Vectors out of {n_samples} samples.")

    # -----------------------------------------------------
    # Step 7: Calculate Bias (b) using 'Free' Support Vectors
    # -----------------------------------------------------
    """
    [MATHEMATICAL REASONING: How to find the intercept 'b'?]
    
    Since 'b' disappeared during the Dual derivation, we must recover it.
    We use the subset of Support Vectors that lie EXACTLY on the margin.
    
    Three Types of Alphas in Soft Margin:
      1. alpha = 0: Correct & Safe (Ignore)
      2. alpha = C: Outlier or Misclassified (Inside the margin, can't use for b)
      3. 0 < alpha < C: "Free" Support Vector (Exactly on the margin)
      
    For points in case #3 (0 < alpha < C):
       The constraint is strictly active: y_i(w.x_i + b) = 1
       Multiplying by y_i (since y^2=1):  w.x_i + b = y_i
       Rearranging for b:                 b = y_i - w.x_i
       
       Substituting w with Kernel trick:  b = y_i - sum( alpha_j * y_j * K(x_j, x_i) )
    """
    
    b_sum = 0
    valid_count = 0
    
    for k in range(len(sv_alphas)):
        # We only use "Free" Support Vectors (strictly on margin) for numerical stability.
        # If alpha == C, the point involves slack variable xi, making equation y(wx+b) = 1 - xi.
        # Since we don't know xi explicitly here, we avoid using these points for calculating b.
        if sv_alphas[k] < C - 1e-5:
            
            # Calculate w.x_k using the Kernel Trick
            # w.x_k = sum( alpha_j * y_j * K(x_j, x_k) )
            w_dot_x = 0
            for j in range(len(sv_alphas)):
                w_dot_x += sv_alphas[j] * sv_y[j] * k_func(sv_X[j], sv_X[k])
            
            # b = label - prediction_without_b
            b_sum += sv_y[k] - w_dot_x
            valid_count += 1
            
    # Average the 'b' values to reduce numerical error
    if valid_count > 0:
        b = b_sum / valid_count
    else:
        # Fallback: If all SVs are bound (alpha == C), use average of all SVs (Approximation)
        # This happens if data is very messy and no points sit exactly on the margin.
        b_sum = 0
        for k in range(len(sv_alphas)):
            w_dot_x = 0
            for j in range(len(sv_alphas)):
                w_dot_x += sv_alphas[j] * sv_y[j] * k_func(sv_X[j], sv_X[k])
            b_sum += sv_y[k] - w_dot_x
        b = b_sum / len(sv_alphas)

    return {
        'sv_alphas': sv_alphas,
        'sv_X': sv_X,
        'sv_y': sv_y,
        'b': b,
        'k_func': k_func
    }

# ---------------------------------------------------------
# Prediction Function (The Result of all math above)
# ---------------------------------------------------------
def predict_dual(model, X_new):
    """
    f(x) = sign( w.x + b )
         = sign( sum(alpha_i * y_i * K(x_i, x)) + b )

    Makes predictions using the learned Dual SVM model.
    
    Decision Function:
      f(x) = sign( \sum_{i \in SV} alpha_i * y_i * K(x_i, x) + b )
    
    Mechanism:
      1. Measure similarity K(x_i, x) between new point 'x' and each Support Vector 'x_i'.
      2. Weight the similarity by alpha_i * y_i (Influence * Class).
      3. Sum the weighted influences and add bias 'b'.
      4. Return sign: +1 or -1.
    """
    y_pred = []
    for x in X_new:
        decision = 0
        for i in range(len(model['sv_alphas'])):
            decision += model['sv_alphas'][i] * model['sv_y'][i] * model['k_func'](model['sv_X'][i], x)
        y_pred.append(np.sign(decision + model['b']))
    return np.array(y_pred)




# =============================================================================
# 5. MAIN EXECUTION (Demonstration & Comparison)
# =============================================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # Scenario 1: Linear Data (Linearly Separable)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" SCENARIO 1: Linear Data with Primal Soft Margin Solver")
    print("="*60)
    
    # 1. Generate Data
    x1, y1 = gen_linear_data(n_samples=100)
    
    # 2. Train Model (Primal QP)
    print("[Training] Solving Primal QP...")
    w, b = solve_primal_soft_margin_qpsolver(x1, y1, C=1.0)
    
    print(f" -> Learned Weight Vector (w): {w}")
    print(f" -> Learned Bias (b): {b:.4f}")
    
    # 3. Visualize (Using Linear Visualization Function)
    print("[Visualization] Plotting Primal Linear Boundary...")
    plt.figure(figsize=(6, 5))
    print_margin_boundary(w, b, x1, y1)

    # ---------------------------------------------------------
    # Scenario 2: Non-Linear Data (Concentric Circles)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" SCENARIO 2: Non-Linear Data with Dual Kernel SVM")
    print("="*60)
    
    # 1. Generate Data
    x2, y2 = get_nonlinear_data(n_samples=100, noise=0.1)
    
    # 2. Train Model (Dual QP with RBF Kernel)
    print("[Training] Solving Dual QP with RBF Kernel...")
    dual_model = solve_dual_soft_margin_qpsolver(x2, y2, C=1.0, kernel='rbf', gamma=0.5)
    
    # 3. Visualize (Using the NEW Non-Linear Visualization Function)
    print("[Visualization] Plotting Dual Non-Linear Boundary...")
    plt.figure(figsize=(6, 5))
    print_dual_boundary(dual_model, x2, y2)
    
    print("\n[Conclusion]")
    print("1. Primal Solver successfully found the Linear Hyperplane.")
    print("2. Dual Solver successfully captured the Non-Linear (Circular) Boundary.")