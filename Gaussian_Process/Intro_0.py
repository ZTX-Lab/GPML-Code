"""
Intro_0.py: The Theoretical Bible of Gaussian Processes
==============================================================================
Summary of the Dialogue: From Classical Statistics to Function-Space View
==============================================================================

This document serves as the theoretical foundation for understanding Gaussian Processes.
It synthesizes the insights from our in-depth discussions regarding Covariance, 
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
       - Intermediate: "Elastic Rubber Band".
         Influence is transferred but decays with distance.

    B. Information View: "The Pipeline"
       - Covariance is the capacity of the pipe through which information flows.
       - Formula: Posterior_Cov = Prior_Cov - (Info_Transferred_via_Kernel)
       - If Cov(train, test) is high, the "Knowledge" from training data flows 
         perfectly to the test point, collapsing its uncertainty to near zero.

    C. Geometric View: "Inner Product"
       - In function space, Covariance is the Dot Product of two vectors.
       - k(x, x') = <phi(x), phi(x')>
       - It measures "Directional Alignment". Are these two points facing the same way?

------------------------------------------------------------------------------
3. One Formula, Many Kernels: The Universality of the Gaussian
------------------------------------------------------------------------------

    A. The Question:
       "Can we put any kernel (Linear, RBF, Periodic) into the SAME Gaussian formula?"
       "Does the math still work even if the function is non-linear?"

    B. The Answer: YES.
       - The Multivariate Gaussian Formula ONLY cares that the Covariance Matrix (K) 
         is Symmetric and Positive Semi-Definite (PSD).
       - It does not care if K came from a line, a curve, or a heartbeat pattern.
       - The Formula: P(y) ~ exp( -0.5 * y.T * K^-1 * y )
         This "Engine" works universally, regardless of the "Fuel" (Kernel).

    C. The Result: "Same Formula, Different Worlds"
       - While the formula is the same, the *Interpretation* changes completely based on K.
       - Case 1 (Linear Kernel): K^-1 forces data to fit a Line.
       - Case 2 (RBF Kernel):    K^-1 forces data to fit a Smooth Curve.
       - Case 3 (Periodic):      K^-1 forces data to fit a Repeating Wave.
       
       * Crucial Insight: 
         Model Selection (finding the best kernel) is about choosing the right 
         "Lens" to view the data, using the Log Marginal Likelihood as the score.

------------------------------------------------------------------------------
4. Specific Kernel Properties: Linear vs. RBF
------------------------------------------------------------------------------

    A. Linear Kernel: k(x, x') = x * x'
       - Input: Takes 2 variables (just like any kernel).
       - Function Shape (Mean): f(x) = w * x. (A straight line through origin).
       - Variance Shape: Var(x) = x^2. (Quadratic).
         * Paradox: The function is linear, but uncertainty grows quadratically (Parabola).
       - Connection: This matches standard Linear Regression (Bayesian interpretation).

    B. RBF Kernel: k(x, x') = exp( -|x-x'|^2 / 2l^2 )
       - Dimensionality: Infinite.
       - Function Shape: A sum of infinite Gaussian bumps -> Can model ANY continuous curve.
       - The "Kernel Trick": 
         We implicitly map data to an infinite dimensional space where it becomes "Linear",
         so we can validly use the Linear Gaussian formulas.

==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, length_scale=1.0):
    sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-0.5 * sq_dist / length_scale**2)

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def periodic_kernel(x1, x2, period=1.0, length_scale=1.0):
    dist = np.abs(x1 - x2.T) # 1D distance for simplicity
    return np.exp(-2 * np.sin(np.pi * dist / period)**2 / length_scale**2)

# --- Visualizing "Same Formula, Different Worlds" ---
# We will create Prior Covariance Matrices for different kernels to see their "Rules".

X = np.linspace(-3, 3, 60).reshape(-1, 1)

kernels = {
    "Linear Kernel\n(Rule: Linearity)": linear_kernel(X, X),
    "RBF Kernel\n(Rule: Smoothness)": rbf_kernel(X, X, length_scale=1.0),
    "Periodic Kernel\n(Rule: Repetition)": periodic_kernel(X, X, period=1.5)
}

plt.figure(figsize=(15, 5))

for i, (name, K) in enumerate(kernels.items()):
    plt.subplot(1, 3, i+1)
    plt.imshow(K, extent=[-3, 3, 3, -3], cmap='viridis')
    plt.title(name)
    plt.colorbar()
    
    # Add descriptive text to the plot
    if "Linear" in name:
        txt = "Values grow away from center.\nUncertainty is Quadratic."
    elif "RBF" in name:
        txt = "High correlation only near diagonal.\nLocal Influence."
    else:
        txt = "Checkerboard pattern.\nRemote Influence repeats."
    plt.xlabel(txt)

plt.suptitle("Visualizing the 'Physical Laws' Defined by Different Kernels", fontsize=16)
plt.tight_layout()
plt.show()

print("Intro_0.py created.")
print("The docstring contains the complete theoretical summary of our discussions.")
print("Check the plots to see how different kernels define different 'Physical Laws' for the same data space.")
