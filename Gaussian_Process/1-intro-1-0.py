"""
Intro_0.py: The Grand Unified Theory of Gaussian Processes
==============================================================================
PART 1: THEORETICAL FOUNDATIONS (Definitions & Philosophy)
==============================================================================

This section synthesizes the insights from our discussions regarding Covariance, 
Kernels, and the duality between Statistics and Function Spaces.


==============================================================================
[Final Summary: The Logical Flow of Gaussian Processes]
==============================================================================
1. The Problem: Prediction without a Function
  - We do not know the exact function f(x), but we need to predict it.
  - However, we CAN specify "conditions" or "characteristics" of the function 
    before seeing the data (Prior).

2. Defining "Shape" via Correlation
  - The best way to describe a function's character is its "Shape" (smoothness, wiggle).
  - The shape is determined by the relationship: "How does Output change when Input changes?"
  - If we can define: "Input Similarity (x) -> Output Similarity (y)",
    we effectively define the function's shape.
  - This relationship is mathematically quantified as "COVARIANCE".

3. The Mathematical Link: Covariance = Inner Product
  - Standard Covariance definition: E[ (f(x)-m)(f(x')-m) ] (Statistical Form).
  - If we expand this formula using the linear model assumption (f = w*phi),
    it transforms into a Vector Inner Product form:
    Cov(f(x), f(x')) = < phi(x), phi(x') >

4. The Discovery (Mercer's Theorem)
  - This leads to a surprising fact: "Any equation that can be expressed as 
    a Vector Inner Product can serve as a Covariance Function (Kernel)."
  - We don't need to manually design features (phi). As long as a formula (like RBF)
    satisfies the inner product condition, it is a valid Covariance.

5. Verification (Theoretical vs. Statistical)
  - If we define the Covariance as a specific Kernel (e.g., "This is RBF"),
    we are setting the "Law" of the world.
  - If we generate infinite data from this law and calculate the statistical 
    covariance (Sample Covariance) of that data, it will Mathematically Converge 
    to the Kernel value we defined.
==============================================================================


+ Additonal Info.

------------------------------------------------------------------------------
0. Fundamental, Unchanging Definition of Covariance
------------------------------------------------------------------------------

For any objects (random variables, function values, processes),
covariance is ALWAYS defined via expectation:

   Cov(A, B) = E[(A - μ_A)(B - μ_B)]

This definition is universal and never changes.
Then why we use upper equation in statiscis but use different equation in GP?

------------------------------------------------------------------------------
[Deep Dive: The Philosophy & Mathematics of Kernels]
------------------------------------------------------------------------------
Q1. If the definition of Covariance is the same as in statistics (E[(y-u)(y-u)]),
    why do we need Kernels like RBF? Can't we just use the standard formula?
A: The "Chicken and Egg" Problem.
  - Standard Statistics (Sample Covariance): Requires existing 'y' values to calculate.
    (e.g., "I have data y1 and y3, let's check their correlation.")
  - Gaussian Processes (Prediction): We don't know 'y' yet! We want to predict it.
    Using the standard formula is impossible because we lack the target values.
  - The Role of Kernel: It allows us to calculate Covariance using ONLY 'x' (location).
    It acts as a "Rule" (Assumption): "If x is close, y will be similar."

Q2. Is the RBF Kernel just an arbitrary definition, or is it mathematically 
    related to the original covariance formula? Can it be expanded back?
A: They are mathematically IDENTICAL (Mercer's Theorem).
  - The Kernel is essentially a "Compressed Zip File" of the original formula.
  - If you expand the RBF function (e.g., using Taylor Series), it reveals itself 
    as an infinite sum of basis functions (features) multiplied together:
    K(x, x') = sum( w_i * phi_i(x) * phi_i(x') )  <-- The original Covariance form!
  - We use the closed-form equation (np.exp) simply because calculating 
    an infinite sum directly is computationally impossible.


Q3. What is the difference between "Feature Covariance" and "Sample Covariance"?
  And does Mercer's Theorem say RBF equals Sample Covariance?
A: No. Mercer's Theorem links RBF to Feature Covariance, not Sample Covariance.
Feature covariance don't use 'y', but sample covariance is calculated using 'y'

  [Comparison]
  1. Feature Covariance (The Blueprint):
      - Formula: phi(x).T * Sigma_p * phi(x')  (or infinite sum of features)
      - Input: Uses theoretical features 'phi' and weights 'w'. No 'y' needed.
      - Meaning: "How the model IS DESIGNED to behave."

  2. Sample Covariance (The Measurement):
      - Formula: sum( (y - mean)... )
      - Input: Uses actual data 'y'.
      - Meaning: "How the data ACTUALLY behaves."

  3. Mercer's Theorem:
      - It proves: RBF Kernel(x, x') == Feature Covariance (Infinite sum).
      - It does NOT claim it equals Sample Covariance.
      - It allows us to compute the infinite feature sum using a simple closed-form
        equation (like np.exp) without knowing 'y'.

Q4. How is the Feature Covariance derived as "phi(x).T * Sigma_p * phi(x')"?
A: It is mathematically derived from two linear model assumptions.

  1. Assumption 1 (Linear Model): f(x) = phi(x).T * w
  2. Assumption 2 (Weight Prior): w ~ N(0, Sigma_p) -> E[w * w.T] = Sigma_p

  [Derivation Step-by-Step]
  Cov(f(x), f(x')) = E[ f(x) * f(x').T ]                 <-- Definition of Cov
                    = E[ (phi(x).T * w) * (phi(x').T * w).T ]
                    = E[ phi(x).T * w * w.T * phi(x') ]   <-- (AB).T = B.T * A.T
                    = phi(x).T * E[w * w.T] * phi(x')     <-- phi is constant, move out
                    = phi(x).T * Sigma_p * phi(x')        <-- Substitute Sigma_p

  * Conclusion: The variance of weights (Sigma_p) is "sandwiched" by features,
                creating the covariance of the function.

Q5. Definition of a Kernel (What is it?)
A Kernel k(x, x') is a function that computes the "similarity" between two inputs
in a high-dimensional feature space, without explicitly computing the features.

  - Mathematical Definition:
    k(x, x') = < phi(x), phi(x') >

  - Where:
    * phi(x): A mapping function that projects input 'x' into a feature space.
    * < , > : The Inner Product (Dot Product) operation in that space.

Q6. Why MUST it be an "Inner Product"? (The Geometric Reason)
Covariance fundamentally measures "Linear Association" or "Similarity".
In geometry and linear algebra, the standard tool to measure similarity 
(direction alignment) between two vectors is the Inner Product.

  - If two vectors point in the same direction:
    Inner Product is MAXimized -> High Covariance (Highly Correlated).
  - If two vectors are orthogonal (90 degrees):
    Inner Product is ZERO -> Zero Covariance (Uncorrelated).

* Conclusion: Since Covariance is a measure of similarity, any valid Covariance
  function must mathematically correspond to an Inner Product in some space.

Q7. Eligibility: What qualifies as a valid Kernel? (Mercer's Condition)
Not every random function can be a Kernel. To be a valid Covariance function,
it must satisfy the "Positive Semi-Definite (PSD)" condition.

  - Why PSD?
    1. Variance (k(x,x)) represents "uncertainty," so it can NEVER be negative.
    2. If a matrix is PSD, linear algebra guarantees that it can be decomposed
      into an inner product of some feature vectors (Mercer's Theorem).

  - Summary:
    Any function k(x, x') that results in a Positive Semi-Definite matrix
    is a valid Kernel, because it proves the existence of a feature map phi(x)
    such that k(x, x') = phi(x) . phi(x').
==============================================================================



==============================================================================
PART 2: THE LOGIC FLOW (From Assumption to Selection)
==============================================================================

This section reconstructs the workflow: How we move from an "infinite possibility space" 
to a "specific function" using Data and Kernels.

------------------------------------------------------------------------------
Step 1. The Setup: The Canvas & The Rule (Prior)
------------------------------------------------------------------------------
    * The Problem: We want to find f(x), but we don't know the formula.
    * The Assumption (Kernel): 
      We define the RBF Kernel. This is our "Law of Physics" (Smoothness).
    * The Prior State:
      We assume Mean = 0. We calculate Covariance on the canvas (X_test).
      We draw "Random Functions" that follow the rule but fit no data yet.

------------------------------------------------------------------------------
Step 2. The Conditioning: Injecting Reality (Data)
------------------------------------------------------------------------------
    * The actual data points (X_train, y_train) arrive.
    * We calculate matrices to bridge the ideal world and reality:
      1. K (Train-Train): Internal correlation of observed data.
      2. K_s (Train-Test): The "Bridge" / "Information Pipeline".
      3. K_ss (Test-Test): The Prior Uncertainty.
    * We compute K^-1 to "normalize" and de-correlate the training information.

------------------------------------------------------------------------------
Step 3. The Selection: Squeezing the Probability Balloon (Posterior)
------------------------------------------------------------------------------
    * We don't "choose" functions manually. The Math does the "Squeezing."
    
    1. Mean Update (The Shape): 
       mu_new = K_s.T * K^-1 * y
       -> The data pulls the mean function towards the observed points.
       
    2. Covariance Update (The Squeeze):
       Cov_new = K_ss - (Information_Gain)
       -> Near the data points, uncertainty drops to near zero.
       -> The probability distribution is "collapsed" at the constraints.
    
    * Result: Only functions that pass through the data points survive.

==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# [Part 1] Visualizing the "Rules" (Kernels)
# ==============================================================================
def rbf_kernel(x1, x2, length_scale=1.0):
    sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-0.5 * sq_dist / length_scale**2)

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

# Canvas for Kernel Visualization
X_vis = np.linspace(-3, 3, 50).reshape(-1, 1)
K_lin = linear_kernel(X_vis, X_vis)
K_rbf = rbf_kernel(X_vis, X_vis)

# ==============================================================================
# [Part 2] Visualizing the "Process" (Prior -> Posterior)
# ==============================================================================
# 1. Setup Canvas (Test Points)
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

# 2. Step 1: The Prior (Before Data)
K_prior = rbf_kernel(X_test, X_test) + 1e-8 * np.eye(len(X_test))
mu_prior = np.zeros(len(X_test))
samples_prior = np.random.multivariate_normal(mu_prior, K_prior, 3)

# 3. Step 2: Conditioning (Data Injection)
X_train = np.array([-4, -2, 0, 2, 4]).reshape(-1, 1)
y_train = np.sin(X_train).flatten() * 1.5  # Artificial data

K_train = rbf_kernel(X_train, X_train) + 1e-8 * np.eye(len(X_train))
K_s = rbf_kernel(X_train, X_test)
K_inv = np.linalg.inv(K_train)

# 4. Step 3: The Posterior (Squeezing)
# Mean: "Weighted sum of influence"
mu_post = K_s.T.dot(K_inv).dot(y_train)
# Covariance: "Prior - Info Gain"
Cov_post = K_prior - K_s.T.dot(K_inv).dot(K_s)
samples_post = np.random.multivariate_normal(mu_post, Cov_post, 3)

# ==============================================================================
# [Plotting] The Grand View
# ==============================================================================
plt.figure(figsize=(15, 10))

# Top Row: The Rules (Kernels)
plt.subplot(2, 2, 1)
plt.imshow(K_lin, extent=[-3, 3, 3, -3], cmap='viridis')
plt.title("Rule 1: Linear Kernel Covariance\n(Uncertainty grows with x)")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(K_rbf, extent=[-3, 3, 3, -3], cmap='viridis')
plt.title("Rule 2: RBF Kernel Covariance\n(Local influence only)")
plt.colorbar()

# Bottom Row: The Process (Inference)
plt.subplot(2, 2, 3)
for sample in samples_prior:
    plt.plot(X_test, sample, '--', alpha=0.7)
plt.title("Step 1: The Prior (Imagination)\nFunctions wandering freely")
plt.ylim(-3, 3)

plt.subplot(2, 2, 4)
std_dev = np.sqrt(np.diag(Cov_post))
plt.fill_between(X_test.flatten(), mu_post - 2*std_dev, mu_post + 2*std_dev, color='gray', alpha=0.2)
for sample in samples_post:
    plt.plot(X_test, sample, '-', alpha=0.8)
plt.plot(X_train, y_train, 'ro', markersize=8, label='Data (Nails)')
plt.title("Step 3: The Posterior (Reality)\nFunctions squeezed by Data")
plt.ylim(-3, 3)
plt.legend()

plt.tight_layout()
plt.show()

print("Intro_0.py created successfully.")
print("It includes Part 1 (Theory Bible) and Part 2 (Logic Flow) in the docstring.")
