"""
Intro_0.py: The Grand Unified Theory of Gaussian Processes
==============================================================================
PART 1: THEORETICAL FOUNDATIONS (Definitions & Philosophy)
==============================================================================

This section synthesizes the insights from our discussions regarding Covariance, 
Kernels, and the duality between Statistics and Function Spaces.

------------------------------------------------------------------------------
1. The Dual Nature of Covariance: "Observation" vs. "Definition"
------------------------------------------------------------------------------
    A. Classical Statistics (Sample Covariance) -> "Bottom-Up / Inductive"
       - Context: Tabular data (Features X, Y).
       - Definition: A calculated measure of linearity based on observed samples.
       - Formula: Cov(X, Y) = E[(X - mu_x)(Y - mu_y)]
       - Philosophy: "We observe the data first, then estimate the relationship."
       - Limit: Standard statistical covariance implicitly assumes a 'Linear Kernel'.
                It checks "Do these points form a line?"

    B. Gaussian Processes (Kernel Covariance) -> "Top-Down / Deductive"
       - Context: Function space (f(x) at infinite points).
       - Definition: A "Rule" or "Law" we impose on the world (The Model Assumption).
       - Philosophy: "We define the relationship rule (Kernel) first, then data is generated."
       - The Kernel Function k(x, x') IS the Covariance.
         - It dictates the "Physics" of the function space before any data is seen.

------------------------------------------------------------------------------
2. The Deep Meaning of Covariance in GP (Physical & Informational)
------------------------------------------------------------------------------
    Covariance is not just a number; it is the "Invisible Link" between points.

    A. Physical View: "Rigidity and Springs"
       - High Covariance (~1.0): "Rigid Steel Rod".
         If you lift f(x_i), f(x_j) MUST move up by the same amount. (Hard Constraint)
       - Low Covariance (~0.0): "Broken Link".
         Moving f(x_i) has zero effect on f(x_j). (Independence)

    B. Information View: "The Pipeline"
       - Covariance is the capacity of the pipe through which information flows.
       - Formula: Posterior_Cov = Prior_Cov - (Info_Transferred_via_Kernel)
       - If Cov(train, test) is high, the "Knowledge" from training data flows 
         perfectly to the test point, collapsing its uncertainty to near zero.

------------------------------------------------------------------------------
3. One Formula, Many Kernels: The Universality of the Gaussian
------------------------------------------------------------------------------
    A. The Question:
       "Can we put any kernel (Linear, RBF, Periodic) into the SAME Gaussian formula?"
    
    B. The Answer: YES.
       - The Multivariate Gaussian Formula ONLY cares that the Covariance Matrix (K) 
         is Symmetric and Positive Semi-Definite (PSD).
       - It works universally, regardless of the "Fuel" (Kernel).

    C. The Result: "Same Formula, Different Worlds"
       - While the formula is the same, the *Interpretation* changes completely based on K.
       - Linear Kernel -> Forces data to fit a Line.
       - RBF Kernel    -> Forces data to fit a Smooth Curve.

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
