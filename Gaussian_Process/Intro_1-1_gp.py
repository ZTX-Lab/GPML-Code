"""
1-1.py: Properties of the GP Prior (Length-scale & Squashing)
============================================================
Reference: Section 1.1 of Rasmussen & Williams (GPML)

Focus:
1. Visualizing the "Prior" (Before seeing data).
2. Understanding how Hyperparameters (Length-scale) affect smoothness.
3. Understanding how we convert GP outputs to Probabilities (Squashing).

Note: No X_train or y_train is used here. We are only observing the "assumptions" of the model.
"""

import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Squared Exponential Kernel (RBF).
    Args:
        length_scale (l): Controls smoothness. 
                          Small l = Rapid variation. Large l = Slow variation.
    """
    sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 * sq_dist / length_scale**2)

def sigmoid(z):
    """Logistic Sigmoid function to squash (-inf, inf) to (0, 1)."""
    return 1 / (1 + np.exp(-z))

# ------------------------------------------------------
# Setup: The Domain (Canvas)
# ------------------------------------------------------
# We only need the test domain (the x-axis) to draw functions.
X_canvas = np.linspace(-5, 5, 100).reshape(-1, 1)
n_samples = 3  # Number of random functions to draw

# ======================================================
# Topic 1: The Effect of Length-scale on the Prior
# ======================================================
# The text mentions: "Slower variation is achieved by simply adjusting parameters..."
# We will compare a SHORT length-scale vs. a LONG length-scale.

length_scales = [0.5, 2.0]  # 0.5 = Rapid, 2.0 = Smooth

plt.figure(figsize=(12, 5))

for i, ls in enumerate(length_scales):
    # 1. Construct the Covariance Matrix (K_ss)
    # Since we have NO data, the Prior Covariance is just the Kernel itself.
    K_prior = rbf_kernel(X_canvas, X_canvas, length_scale=ls)
    
    # 2. Add Jitter for numerical stability (to ensure positive definiteness)
    K_prior += 1e-8 * np.eye(len(X_canvas))
    
    # 3. Sample from the Prior
    # Mean is assumed to be 0 vector. Covariance is K_prior.
    f_samples = np.random.multivariate_normal(
        mean=np.zeros(len(X_canvas)), 
        cov=K_prior, 
        size=n_samples
    )
    
    # 4. Plot
    plt.subplot(1, 2, i+1)
    for f in f_samples:
        plt.plot(X_canvas, f)
    
    plt.title(f"Topic 1: Prior with Length-scale l={ls}\n({'Rapid' if ls < 1 else 'Smooth'} Variation)")
    plt.xlabel('Input x')
    plt.ylabel('f(x)')
    plt.ylim(-3, 3)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# ======================================================
# Topic 2: From Regression to Classification (Squashing)
# ======================================================
# The text mentions: "Squash the prior function f... through a response function."
# We take a sample f(x) and pass it through a Sigmoid to get pi(x).

plt.figure(figsize=(10, 6))

# 1. Generate one random function f(x) from the Prior (l=1.0)
K_cls = rbf_kernel(X_canvas, X_canvas, length_scale=1.0) + 1e-8 * np.eye(len(X_canvas))
f_sample = np.random.multivariate_normal(np.zeros(len(X_canvas)), K_cls, 1).flatten()

# 2. Squash it! (Transformation to Probability)
pi_sample = sigmoid(f_sample)

# 3. Plot Comparison
plt.plot(X_canvas, f_sample, 'k--', label=r'Latent Function $f(x)$ (Range: $-\infty$ to $\infty$)')
plt.plot(X_canvas, pi_sample, 'r-', linewidth=2, label=r'Class Probability $\pi(x) = \sigma(f(x))$ (Range: 0 to 1)')

# Add guide lines
plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(0.5, color='blue', linestyle='-.', alpha=0.3, label='Decision Boundary (0.5)')

plt.title("Topic 2: Squashing Function (Prior for Classification)")
plt.xlabel("Input x")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()