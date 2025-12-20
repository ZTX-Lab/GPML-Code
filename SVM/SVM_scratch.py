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

"""
SVM Primal Solver Implementations:

1. Linear Boundary Only: 
   It always produces a straight linear hyperplane.
   It cannot capture non-linear patterns (like circles) even with Soft Margin.
   (Soft Margin allows ignoring outliers, but the boundary itself remains linear.)

2. Direct Optimization: 
   It computes the optimal weights (w) and bias (b) directly in the primal space.

3. Kernel Difficulty: 
   It assumes we can calculate 'w'. In Kernel methods, 'w' often resides in an 
   infinite-dimensional space, making explicit calculation impossible. 
   Thus, Primal Solvers cannot easily utilize the Kernel Trick.

4. Dimensionality Sensitivity: 
   Computational complexity depends on the feature dimension (d), 
   which can be computationally expensive for high-dimensional data.

Implementations provided:
1. Hard Margin QP Solver (solve_primal_qpsolver) - Uses cvxopt
2. Soft Margin QP Solver (solve_primal_soft_margin_qpsolver) - Uses cvxopt
3. Gradient Descent Solver (solve_primal_gradient) - Iterative approach
4. Newton's Method Solver (solve_primal_newton_vectorized) - Iterative approach
"""
# 1. Hard Margin QP Solver using cvxopt
def solve_primal_qpsolver(x,y):
    n_samples, n_features = x.shape


    # ---------------------------------------------------------
    # Derivation of Hard Margin SVM
    # ---------------------------------------------------------
    # 1. Decision Boundary: w.x + b = 0
    # 2. Constraint: y(w.x + b) >= 1 for all samples.
    # 3. Geometric Margin: The distance between the boundary and nearest points.
    #    Formula: |w.x + b| / ||w||
      # Explanation:
      # 0. We want to find w and b that defines decision boundary.
      # 1. P = sample, Q = point on decision boundary
      # 2. distance vector = P - Q
      # 3. unit vector w = w / ||w||
      # 4. margin = |(P-Q) . (w/||w||)| = |(P-Q).w| / ||w||
      # 5. since Q is on decision boundary, Q satisfies w.Q + b = 0, thus Q.w = -b
      #    thus margin = |P.w + b| / ||w||
      # 6. we know that y(P.w + b) >= 1, for support vectors, y(P.w + b) = 1
      #    thus margin = 1 / ||w||
      # thus, to maximize margin, we need to minimize ||w||.
      # this is equivalent to minimizing 1/2 * ||w||^2 for easier calculation.
      # now we have objective function and constraints.
    
    # ---------------------------------------------------------
    # 1. Configuration of QP parameters
    # ---------------------------------------------------------
    # Variables: u = [w1, w2, ..., wn, b] 
    
    # ---------------------------------------------------------
    # 2. Objective Function (Matrix P, Vector q)
    # ---------------------------------------------------------
    # Standard QP: Minimize (1/2 * u.T * P * u + q.T * u)
    # We want to minimize w^2, but ignore b.


    P_np = np.eye(n_features + 1)
    P_np[n_features, n_features] = 0.0 # Do not penalize bias 'b'
    P = matrix(P_np)
    
    q = matrix(np.zeros(n_features + 1)) # No linear term in objective

    # ---------------------------------------------------------
    # 3. Inequality constraints for QP solver: G, h
    # ---------------------------------------------------------
    # Standard QP: G * u <= h
    # Our Constraint: y(w.x + b) >= 1
    # Converted:     -y(w.x + b) <= -1
    # Coefficients:  -y*x (for w), -y (for b)
    # G = -y*[x1, x2, 1], h = -1

    G = np.zeros((n_samples, n_features + 1))

    for i in range(n_samples):
      # w: -y_i * x_i
      G[i, :n_features] = -1 * y[i] * x[i]
      # b: -y_i * 1
      G[i, n_features] = -1 * y[i]
    G = matrix(G)
    h = matrix(-1.0 * np.ones(n_samples))

    # ---------------------------------------------------------
    # 4. Solve QP using cvxopt solver
    # ---------------------------------------------------------
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h)


    # ---------------------------------------------------------
    # 5. Extract w and b from solution
    # ---------------------------------------------------------
    w = np.array(solution['x'][:n_features]).flatten()
    b = np.array(solution['x'][n_features]).flatten()[0]
    return w, b


# 2. Soft Margin QP Solver using cvxopt
def solve_primal_soft_margin_qpsolver(x, y, C=1.0):
    n_samples, n_features = x.shape

    # ---------------------------------------------------------
    # Difference between Hard Margin & Soft Margin Variables
    # ---------------------------------------------------------
    # Hard Margin variables: u = [w..., b] (size: features + 1)
    # Soft Margin variables: u = [w..., b, xi...] (size: features + 1 + samples)
    #
    # We need 'slack variables (xi)' for EACH sample to handle
    # - Hard margin does not allow any violation (infeasible for noisy data).
    # - Slack variables allow samples to be inside the margin or misclassified,
    #   at the cost of a penalty in the objective function.
    
    n_vars = n_features + 1 + n_samples # Total number of optimization variables

    # ---------------------------------------------------------
    # 1. Objective Function (Matrix P, Vector q)
    # ---------------------------------------------------------
    # Objective: Minimize 1/2 * ||w||^2 + C * sum(xi)
    
    # [Matrix P] Quadratic terms (only for w)
    P_np = np.zeros((n_vars, n_vars))
    P_np[0:n_features, 0:n_features] = np.eye(n_features)
    P = matrix(P_np)


    # [Vector q] Linear terms (only for xi)
    # The parameter C acts as a penalty weight for slack variables.
    # Note: In the Primal form, C is a static constant penalty applied directly.
    # (Unlike Lagrange multipliers 'alpha', which are dynamic variables determined by the solver)
    # We can use static penalty C because primal form accumulates penalty for each slack variable directly.
    # This is different from dual form which uses lagrange multipliers to balance constraints and objective.


    q_np = np.zeros(n_vars)
    q_np[n_features + 1:] = C 
    q = matrix(q_np)

    # ---------------------------------------------------------
    # 2. Inequality Constraints (Matrix G, Vector h)
    # ---------------------------------------------------------
    # Soft Margin has two sets of constraints:
    # 1. Margin Constraint: y(w.x + b) >= 1 - xi  =>  -y(w.x + b) - xi <= -1
    # 2. Slack Positivity:  xi >= 0               =>  -xi <= 0
    
    G_np = np.zeros((2 * n_samples, n_vars))
    h_np = np.zeros(2 * n_samples)

    # [Constraint Set 1] Margin constraints
    for i in range(n_samples):
        # w: -y * x
        G_np[i, :n_features] = -1 * y[i] * x[i]
        # b: -y
        G_np[i, n_features] = -1 * y[i]
        # xi: -1 (moved to LHS)
        G_np[i, n_features + 1 + i] = -1.0
        # RHS: -1
        h_np[i] = -1.0

    # [Constraint Set 2] Slack positivity constraints
    for i in range(n_samples):
        row_idx = n_samples + i 
        # xi: -1
        G_np[row_idx, n_features + 1 + i] = -1.0
        # RHS: 0
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
    
    # If needed, you can also extract slack variables:
    # xi = u_opt[n_features+1:] 
    
    return w, b




# 3. Gradient Descent Method (Manual Implementation) (update all samples at once) (mini batch can be implemented similarly)
def solve_primal_gradient(x, y, learning_rate=0.001, n_iters=1000, C=1.0):
    """
    Solves Soft Margin SVM using Batch Gradient Descent.
    
    Objective Function (Primal):
      Loss = 1/2 * ||w||^2 + C * sum( max(0, 1 - y(w.x + b)) )
    
    - The term 'max(0, 1 - y(w.x + b))' is the Hinge Loss.
    - It penalizes samples that violate the margin boundary. (Allowing some violations for soft margin)
    - We compute the gradient of this loss function and update w and b iteratively.
    - We use static value C to control the penalty for margin violations, not use alhpa, lagrange multipliers.
    """


    n_samples, n_features = x.shape
    
    # Initialize weights
    w = np.zeros(n_features)
    b = 0.0
    
    for epoch in range(n_iters):
      # ---------------------------------------------------------
      # 1. Forward Pass
      # ---------------------------------------------------------
      # Calculate scores for all samples
      decision = np.dot(x, w) + b
      
      # ---------------------------------------------------------
      # 2. Check Constraints & Calculate Shortfall
      # ---------------------------------------------------------
      # 'shortfall' represents how much a sample violates the margin requirement (1).
      # Also known as 'functional margin deficit'.
      # shortfall = 1 - y * (w.x + b)
      shortfall = 1 - y * decision
      
      # Identify violating samples (Active Set)
      # mask is True if the sample is within the margin or misclassified (Loss > 0)
      mask = shortfall > 0
        
      # ---------------------------------------------------------
      # 3. Calculate Gradients (Vectorized)
      # ---------------------------------------------------------
      # Derivative of Objective Function:
      # dJ/dw = w + C * sum( -y_i * x_i )  [only for violators]
        
      # Filter y: keep only violating samples, set others to 0
      y_masked = y * mask 
      
      # Gradient from Hinge Loss part
      # Matrix multiplication handles the summation over samples
      grad_hinge_w = -C * np.dot(x.T, y_masked)
      grad_hinge_b = -C * np.sum(y_masked)
      
      # Combine with Regularization term (w)
      # Note: We divide the loss gradient by n_samples to prevent explosion 
      # with large datasets (optimizing Mean Loss instead of Sum Loss).
      dw = w + (grad_hinge_w / n_samples)
      db = grad_hinge_b / n_samples
      
      # ---------------------------------------------------------
      # 4. Update Weights
      # ---------------------------------------------------------
      w -= learning_rate * dw
      b -= learning_rate * db

    return w, b


# 4. Vectorized Newton's Method
def solve_primal_newton_vectorized(x, y, n_iters=20, C=1.0):
    """
    Solves Primal SVM using Newton's Method (Fully Vectorized).
    
    [Why Newton's Method?]
    Standard Gradient Descent uses only the 1st derivative (Slope).
    Newton's Method uses the 1st (Slope) AND 2nd derivative (Curvature/Hessian).
    This allows it to find the minimum much faster (fewer iterations), 
    jumping towards the bottom of the error function.

    [Objective Function: Squared Hinge Loss]
    Standard Hinge Loss 'max(0, 1-z)' is not differentiable at the kink (z=1).
    So we cannot calculate the Hessian (2nd derivative).
    Instead, we use 'Squared Hinge Loss': max(0, 1-z)^2.
    
    J(u) = 1/2 * ||w||^2 + C * sum( max(0, 1 - y(u.x))^2 )
    """
    n_samples, n_features = x.shape
    
    # ---------------------------------------------------------
    # 1. Variable & Data Setup
    # ---------------------------------------------------------
    # Combine w and b into a single variable 'u' for easier matrix calc.
    # u = [w1, w2, ..., wn, b]
    u = np.zeros(n_features + 1)
    
    # Augment X with a column of 1s to handle bias 'b' automatically.
    # X_bias = [[x1, x2, ..., 1], ...]
    # This allows computing 'w.x + b' as a single dot product 'X_bias . u'
    X_bias = np.hstack([x, np.ones((n_samples, 1))])
    
    # Create Identity matrix for Regularization term (1/2 * ||w||^2)
    # The last element is set to 0 because we do NOT regularize the bias 'b'.
    I_reg = np.eye(n_features + 1)
    I_reg[n_features, n_features] = 0
    
    for i in range(n_iters):
        # ---------------------------------------------------------
        # 2. Forward Pass & Check Constraints
        # ---------------------------------------------------------
        # Calculate scores for all samples: y_pred = w.x + b
        scores = np.dot(X_bias, u)
        
        # Calculate 'Shortfall' (Functional Margin Deficit)
        # shortfall = 1 - y * score
        # Positive shortfall means the sample is an error or within the margin.
        shortfall = 1 - y * scores
        
        # Identify Active Set (Samples that contribute to the Loss)
        # Only samples with shortfall > 0 are considered for Gradient/Hessian.
        mask = shortfall > 0
        
        # ---------------------------------------------------------
        # 3. Calculate Gradient (1st Derivative)
        # ---------------------------------------------------------
        # We need the derivative of Squared Hinge Loss: d/du (1 - y(u.x))^2
        # Chain Rule: 2 * (1 - y(u.x)) * (-y * x)
        #           = -2 * y * shortfall * x
        
        # Calculate scalar weights for the gradient
        # shape: (n_samples, )
        grad_weights = -2 * C * y * shortfall * mask
        
        # Compute the weighted sum of X vectors
        # X.T dot weights performs summation: sum( weight_i * x_i )
        grad_loss = np.dot(X_bias.T, grad_weights)
        
        # Total Gradient = Regularization Gradient (u) + Loss Gradient
        gradient = np.dot(I_reg, u) + grad_loss
        
        # ---------------------------------------------------------
        # 4. Calculate Hessian (2nd Derivative)
        # ---------------------------------------------------------
        # We need the derivative of the Gradient.
        # Gradient term was: -2 * C * y * (1 - y(u.x)) * x
        # Derive with respect to u again:
        #           = -2 * C * y * (-y * x) * x.T
        #           = 2 * C * y^2 * x * x.T
        # Since y^2 is always 1, this simplifies to: 2 * C * x * x.T
        
        # Select only active samples (mask == True)
        X_active = X_bias[mask]
        
        # Compute X^T * X for active samples (Matrix Multiplication)
        # This replaces the loop for summation of outer products.
        hessian_loss = 2 * C * np.dot(X_active.T, X_active)
        
        # Total Hessian = Regularization Hessian (I) + Loss Hessian
        hessian = I_reg + hessian_loss
        
        # ---------------------------------------------------------
        # 5. Newton Update Step
        # ---------------------------------------------------------
        # Standard Update: u_new = u_old - Learning_Rate * Gradient
        # Newton Update:   u_new = u_old - Inverse(Hessian) * Gradient
        #
        # Instead of calculating Inverse(H) explicitly (slow & unstable),
        # we solve the linear system: H * delta = -gradient
        try:
            delta = np.linalg.solve(hessian, -gradient)
            u += delta
        except np.linalg.LinAlgError:
            # Fallback if Hessian is singular (rare with Regularization)
            break
            
    # Extract w and b from the unified variable u
    w = u[:n_features]
    b = u[n_features]
    
    return w, b



# TODO 12-21
# 4.Vectorized Newton's Method with Hard Margin complete implmementation
# 5.Dual QP Solver using cvxopt
# 6.Dual QP Solver with Kernel using cvxopt



if __name__ == "__main__":
    x1, y1 = gen_linear_data(n_samples=100)
    x2, y2 = get_nonlinear_data(n_samples=100, noise=0.1)
    print_figure(x1, x2, y1, y2)

    w, b = solve_primal(x1, y1)
    print("Learned weights (w):", w)
    print("Learned bias (b):", b)
    print_margin_boundary(w, b, x1, y1)
    # Note: The above SVM primal solver will only work well for linear data (x1, y1).
    # It will not be able to find a good decision boundary for non-linear data (x2, y2).




