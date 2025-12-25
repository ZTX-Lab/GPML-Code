"""
================================================================================
[GP (Gaussian Process) Learning Note: Core Concepts Summary]
================================================================================

1. The Fundamental Dilemma & GP's Approach
   - Connecting data points with a single function (e.g., y=ax+b) risks Underfitting.
   - Using high-degree polynomials risks Overfitting.
   - GP's Solution: Instead of fixing one function, we assign a probability 
     to "every possible function."

2. Intuition: Function = Infinite-Dimensional Vector
   - A vector v=[v1, v2] has discrete indices. 
   - A function f(x) has a value for every continuous x, so it is like a vector 
     with infinite indices.
   - Therefore, a GP is like rolling "infinite dice" simultaneously.

3. Q. "What exactly is a Kernel?"
   [Comparison: SVM vs. GP]
   - Definition: A Kernel is an inner product in a high-dimensional feature space.
     Math: k(x, x') = <phi(x), phi(x')>
   
   - In SVM (Geometric View): It is used as a "Kernel Trick" to calculate 
     dot products without actually transforming data, to find a decision boundary.
   - In GP (Statistical View): It is used as a "Similarity Measure" (Covariance).
     "High inner product = The two points are similar."
     GP uses this to enforce the rule: "If inputs x and x' are close, 
     their outputs y and y' must be similar."

4. Q. "Why is RBF (Radial Basis Function) the default?"
   - Reason 1: Infinite Differentiability.
     The RBF function (bell curve) has no sharp corners no matter how many times 
     you differentiate it. It creates the smoothest possible curves.
     It reflects the assumption that "Physical laws in nature are usually smooth."
     
   - Reason 2: Universality.
     RBF corresponds to an inner product in an "infinite-dimensional" space.
     It is flexible enough to approximate any smooth function given enough data.

5. The Magic of Marginalization (Handling Infinity)
   - We don't need to compute the infinite vector.
   - We only need to compute the finite set of points we care about (Training + Test).
   - The math guarantees that the result is identical to calculating the infinite case.

6. The Workflow (Prior -> Observation -> Posterior)
   - Prior: "Any smooth curve is possible." (Mean = 0, Wide uncertainty)
   - Observation: "The function MUST pass through (1, 3)."
   - Posterior: We filter out functions that don't match the data.
     Near data -> Uncertainty collapses (Tube narrows).
     Far from data -> Uncertainty remains high (Tube widens).

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PART 1: The "Stickiness" Rule (The Kernel)
# ==========================================
def kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Computes the Squared Exponential (RBF) Kernel.
    
    [Meaning in GP]
    This acts as the 'Covariance Function'.
    - Math: It computes the inner product in an infinite-dimensional feature space.
    - Intuition: It defines how 'sticky' two points are.
      If x1, x2 are close -> Output near 1 -> y values MUST be similar.
      If x1, x2 are far -> Output near 0 -> y values are independent.
    """
    # Calculate Euclidean distance between every pair of points
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    
    # Apply Gaussian function: exp(-distance)
    return variance * np.exp(-0.5 / length_scale**2 * sqdist)

# ==========================================
# PART 2: Setup - "Discretizing the Infinite"
# ==========================================
# [NOTE: What are n_test and X_test?]
# Since a function is an infinite-dimensional vector, a computer cannot store it directly.
# We must "discretize" it by picking a finite number of points (e.g., 50) to represent it.
#
# - n_test: Think of this as the "Resolution" of our graph.
#           If we pick 5 points -> The graph looks blocky.
#           If we pick 50 points -> The graph looks like a smooth curve.
#           (We are just connecting the dots to visualize the function.)
#
# - X_test: These are the "Query Points" (the arbitrary locations where we want to know y).

n_test = 50
X_test = np.linspace(-5, 5, n_test).reshape(-1, 1) # Shape: (50, 1)

# ==========================================
# PART 3: The Prior - "Imagining before Seeing"
# ==========================================
# 1. Mean: We assume the function oscillates around 0.
mu_prior = np.zeros(X_test.shape[0])

# 2. Covariance: How are these 50 points related?
# We ask the kernel: "Who is close to whom?"

# [CAUTION: Understanding Covariance vs. Y-values]
# We ask the kernel: "Who is close to whom?" 
#
# Q: "If Covariance is 0 (No relation), does it mean y-values MUST be different?"
# A: NO! It simply means they are "Independent".
#
# - High Covariance (Close points): "Strong Constraint."
#   They are tied together. If y1 is 10, y2 is FORCED to be near 10.
#
# - Low/Zero Covariance (Far points): "No Constraint."
#   They don't care about each other. If y1 is 10, y50 can be anything (100, -5, etc.).
#   *IMPORTANT*: It is possible that y1 and y50 happen to be equal (e.g., both 10)
#   by pure coincidence. Independence means "not influenced," not "must be different."

Cov_prior = kernel(X_test, X_test)

# 3. Sampling: We draw 3 random functions.
# Because of Cov_prior (Kernel), they look like smooth curves, not noise.

# [NOTE: How does 'multivariate_normal' actually work?]
# This function is the "Engine" of GP. It turns random noise into smooth curves.
#
# <The Internal Mechanism: X = mu + L * Z>
# 
# Step 1: Generate White Noise (Z)
#   - It creates 50 independent random numbers (e.g., [1.2, -0.5, ...]).
#   - If plotted, this looks like jagged TV static noise (No correlation).
#
# Step 2: Apply the "Filter" (L) - The Cholesky Decomposition
#   - It decomposes the Covariance Matrix (Cov_prior) into a lower triangular matrix L.
#   - It multiplies L with Z. This "mixes" the independent noise.
#   - The noise at point x1 starts to influence x2, x3... creating "correlation."
#
# Step 3: Result (Smooth Function)
#   - The jagged noise is smoothed out according to the Kernel's rules.
#   - We get a continuous, wave-like vector instead of scattered dots.

samples_prior = np.random.multivariate_normal(mu_prior, Cov_prior, 3)

# ==========================================
# PART 4: The Observation - "Reality Check"
# ==========================================
# Ideally, these are the points our function MUST pass through.
X_train = np.array([[-4], [-1], [0], [2], [3]])
y_train = np.array([[-2], [1], [0], [-1], [2]])

# ==========================================
# PART 5: The Posterior - "Collapsing the Possibilities"
# ==========================================
# This relies on the property of Joint Gaussian Distributions (Conditioning).

# Construct Kernel Matrices
K = kernel(X_train, X_train)      # Train vs Train: How correlated the training points are with themselves.
K_s = kernel(X_train, X_test)     # Train vs Test: The "Bridge". Each column tells how close a test point is to the 5 training points.
K_ss = kernel(X_test, X_test)     # Test vs Test: The prior uncertainty.

# Inverse of K (with small jitter for numerical stability)
# We need K^-1 to "normalize" the information from training data.
# Added 1e-8 (Jitter) to diagonal to prevent dividing by zero (Singular Matrix).
# if we are not use de-correlation, the model may be unstable.
# e.g., two training points are extremely close to each other. and then K becomes singular.?


K_inv = np.linalg.inv(K + 1e-8 * np.eye(len(X_train)))

# --- STEP A: Calculate the New Mean (Prediction) ---
# Formula (Math): mu_* = K_*^T * K^-1 * y
# Formula (Code): mu   = K_s.T * K_inv * y_train
#
# [Deep Dive into the Formula]
# This line performs a "Weighted Average" logic in two steps:
#
# 1. The Weight Calculation: alpha = (K_inv * y_train)
#    - We don't just use y_train directly because some data points might be clustered together (redundant info).
#    - Multiplying by K_inv "normalizes" the y values, removing correlation/redundancy between them.
#    - Result: 'alpha' represents the "Pure Influence Weight" of each training point.
#
# 2. The Prediction: result = K_s.T * alpha
#    - K_s is shape (5, 50) [Train x Test].
#    - We need [Test x Train] to multiply with weights. So we transpose it -> K_s.T
#    - Meaning: "Weighted sum of training points based on how close I am to them."
#
# 3. .reshape(-1): Flattens the result from (50, 1) to (50,) for easier plotting.

mu_posterior = K_s.T.dot(K_inv).dot(y_train).reshape(-1)

# --- STEP B: Calculate the New Covariance (Uncertainty) ---
# Formula (Math): Sigma_* = K_** - K_*^T * K^-1 * K_*
# Formula (Code): Cov     = K_ss - K_s.T * K_inv * K_s
# [Q: why we subtract this term from K_ss?]
# k_ss represents our initial uncertainty about the test points (Prior).
# we dont have y, the values at test points. however we can get correlation between test points and train points (K_s).
# K_s, k_ss represent correlation between points, not the actual values.
# we can get value using correlation and known values at train points (y_train).


#
# [Deep Dive into the Formula]
# Intuition: "Posterior Uncertainty = Prior Uncertainty - Information Gained"
#
# 1. Prior: Start with K_ss (Total ignorance/uncertainty).
# 2. Info Gain: The term (K_s.T * K_inv * K_s)
#    - If K_s is HIGH (Test point is near Data): We subtract a lot.
#      Result: Variance becomes near 0. (We are confident!)
#    - If K_s is LOW (Test point is far from Data): We subtract almost nothing.
#      Result: Variance stays high. (We are still uncertain.)

# [Q: Why multiply K_s twice? (The Quadratic Form)]
# 
# 1. Dimensionality (The Loop):
#    We need the result to be (Test x Test).
#    Path: Test -> Train (K_s) -> Train (K_inv) -> Test (K_s.T)
#    This "Round Trip" converts information from training data back to the test domain.
#
# 2. Unit Consistency (Squaring):
#    Variance is a "Squared" quantity (Sigma^2).
#    K_s is just correlation (Linear).
#    To subtract from Variance, we effectively need "Correlation Squared" (K_s * K_s).
#
# 3. multivariate Gaussian distribution property: (schur complement)
# This is a standard result from the theory of multivariate Gaussian distributions.
# It ensures that the resulting covariance matrix is valid (positive semi-definite).

Cov_posterior = K_ss - K_s.T.dot(K_inv).dot(K_s)

# Sampling again from the Posterior distribution.
# 3 random functions that fit the observed data.
# They should pass near the red points (X_train, y_train).

# [Q: Why use Cov_posterior instead of Cov_prior here?]
#
# - If we use Cov_prior:
#   The functions will wander around randomly like the Prior.
#   They will IGNORE the training data points (Training data acts as a constraint).
#
# - By using Cov_posterior:
#   The covariance near the data points is almost 0.
#   This forces the random samples to pass EXACTLY through (or very close to) the red dots.
#   This creates the "Tying the knot" effect


samples_posterior = np.random.multivariate_normal(mu_posterior, Cov_posterior, 3)

# ==========================================
# PART 6: Visualization
# ==========================================
plt.figure(figsize=(14, 6))

# Plotting Prior
plt.subplot(1, 2, 1)
for i in range(3):
    plt.plot(X_test, samples_prior[i], linestyle='--')
plt.title('Prior: Smooth functions (defined by Kernel), No Data')
plt.ylim(-3, 3)

# Plotting Posterior
plt.subplot(1, 2, 2)
# 1. The "Tube" of Uncertainty
uncertainty = 1.96 * np.sqrt(np.diag(Cov_posterior)) # from gausian distribution,  1 : 68%, 1.96 : 95%, 2.58 : 99%
plt.fill_between(X_test.flatten(), 
                 mu_posterior - uncertainty, 
                 mu_posterior + uncertainty, 
                 alpha=0.2, color='gray', label='Uncertainty (95% Confidence)')

# 2. The Mean Prediction
plt.plot(X_test, mu_posterior, 'b-', lw=2, label='Mean Prediction')

# 3. The Sampled Functions
for i in range(3):
    plt.plot(X_test, samples_posterior[i], linestyle='-', alpha=0.5)

# 4. The Real Data
plt.plot(X_train, y_train, 'ro', markersize=10, label='Observed Data')

plt.title('Posterior: Functions constrained by Data')
plt.legend()
plt.ylim(-3, 3)
plt.show()