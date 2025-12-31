import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



"""
================================================================================
[Section 2.1: Weight-space View - Core Concepts & Motivation]
================================================================================
Based on Rasmussen & Williams, Gaussian Processes for Machine Learning.

1. The Trade-off: Simplicity vs. Flexibility
   - Standard Linear Model (y = w^T x):
     * Virtues: Simple to implement, easy to interpret (weights w have meaning).
     * Drawback: Limited flexibility. It fails if the data relationship is non-linear 
       (e.g., trying to fit a curve with a straight line).

2. The Bridge to GP: Feature Spaces & The Kernel Trick
   - Solution: To handle non-linear data, we project inputs into a high-dimensional 
     feature space (phi(x)) and apply the linear model there.
   - The "Kernel Trick": Explicitly computing in high dimensions is expensive. 
     However, we can compute the inner products implicitly using a Kernel function k(x, x').
     * This leads to massive computational savings when feature dimension >> data points.

3. Bayesian Treatment
   - Instead of finding one "best" weight vector w, we infer a distribution over w.
   - Prior: p(w) (Our belief before seeing data).    <=> GP: K_ss (Prior Covariance)
   - Likelihood: p(y|X, w) (How well weights explain the data).  <=> GP: K^-1*y
   - Posterior: p(w|X, y) (Updated belief after seeing data).   <=> GP: K_ss - K_s.T K^-1 K_s

4. Notation & Goal
   - Data D = {(x_i, y_i)} collected in Design Matrix X and Target Vector y.
   - Goal: To infer the conditional distribution p(y|x) (Predicting targets given inputs).
   - We focus on "Discriminative Modelling" (modeling the target), not modeling 
     the input distribution p(x) itself.
================================================================================
"""


"""
================================================================================
[Section 2.1.1: Bayesian Linear Regression - Derivation & Logic]
================================================================================
Reference: Equation (2.3) to (2.4) in Rasmussen & Williams


--------------------------------------------------------------------------------
0. Distributed gaussian distribution with zero mean and variance sigma_p^2
--------------------------------------------------------------------------------
we assume that observaed values y differ from the true function value f(x) by
additive Gaussian noise

  eplison ~ N(0, sigma_p^2)

The noise assumption leads to the likelihood function p(y|X, w) being a Gaussian
with mean X^T w and covariance sigma_n^2 I
  
--------------------------------------------------------------------------------
1. The Likelihood (우도): p(y|X, w)
--------------------------------------------------------------------------------

p(y|X, w) = Π [ Gaussian(y_i | mean=x_i^T w, var=σ_n^2) ]
          = N(X^T w, σ_n^2 I) = (1 / √(2πσ_n^2)) * exp( - |y - X^T w|^2 / (2σ_n^2) )

Q: Why does the formula look like a product (Π)?
A: Because of the "i.i.d assumption" (Independent Identically Distributed).
   The probability of seeing all n data points is the product of seeing each one.

   p(y|X, w) = p(y_1) * p(y_2) * ... * p(y_n)
             = Π [ Gaussian(y_i | mean=x_i^T w, var=σ_n^2) ]

[Step-by-Step Math Derivation]
1) Individual PDF:
   p(y_i) = (1 / √(2πσ_n^2)) * exp( - (y_i - x_i^T w)^2 / (2σ_n^2) )

2) Product of n terms:
   - Constant part: (1 / √(2πσ_n^2))^n  => (2πσ_n^2)^(-n/2)
   - Exponential part: exp(A) * exp(B) = exp(A + B)
     So, Product(exp(...)) becomes exp( Sum(...) )

   Sum of squares = Σ (y_i - x_i^T w)^2  =  |y - X^T w|^2 (Euclidean Norm)

3) Final Matrix Form (Eq 2.3):
   p(y|X, w) = N(X^T w, σ_n^2 I) = (1 / √(2πσ_n^2)) * exp( - |y - X^T w|^2 / (2σ_n^2) )
   
   * Meaning: "Given weights w, the data y is expected to appear around the line X^T w, 
     scattered by noise variance σ_n^2."

--------------------------------------------------------------------------------
2. The Prior (사전 분포): p(w)
--------------------------------------------------------------------------------

Rememver, we are not looking for a single best w, but a distribution over possible w.
Then how do we define our initial belief about w?
We know nothing about w, so we use a Gaussian prior with zero mean and covariance Σ_p.

Q: What does w ~ N(0, Σ_p) mean? (Eq 2.4)
A: It represents our belief about the weights BEFORE seeing any data.

   w ~ N(0, Σ_p)

   - Mean = 0: 
     "We don't know if the relationship is positive or negative yet. 
      So we assume the weights are centered around 0."
     
   - Covariance = Σ_p (Sigma_p):
     "How large do we expect the weights to be?"
     * Large Σ_p: We allow the line to have a very steep slope (Flexible).
     * Small Σ_p: We force the weights to be small (Rigid). 
       -> This acts exactly like "L2 Regularization (Ridge Regression)"!
       -> It prevents Overfitting by penalizing large weights.

Q. is it okay to assume a Gaussian prior for w?
A. Yes, because of Mathematical Convenience (Conjugacy).
    - Gaussian Prior + Gaussian Likelihood => Gaussian Posterior

Q. but convienience is not a good reason. what if the true weight is not in covariance area?
A. True, but we often don't know the true weights anyway.
    - The Gaussian prior is the "most honest" assumption (Maximum Entropy) 
      given only mean and covariance information.
    - It allows us to perform exact Bayesian inference without numerical approximations.

Q. but the problem is "WE" define the covariance matrix Σ_p. what if we define it wrong?
A. Good point. The choice of Σ_p reflects our assumptions about the function complexity.
    - If we choose a very small Σ_p, we are assuming a simple model (strong regularization).
    - If we choose a large Σ_p, we allow more complex models (weak regularization).
    - In practice, we can tune Σ_p based on validation data or use hierarchical Bayesian methods.

Q. thus is there any problem if we use wrong covariance matrix?
A. Yes, if Σ_p is too small, we may underfit the data (too rigid).
    - If Σ_p is too large, we may overfit the data (too flexible).


           
------------------------------------------------------------------------------
[Concept Check: Addressing Common Confusions (Weight-space vs. Function-space)]
------------------------------------------------------------------------------
Q1. Is "w ~ N(0, Sigma_p)" an absolute TRUTH?
A: NO, it is a mathematical ASSUMPTION (Prior).
  - We "pre-specify" Sigma_p (hyperparameter) to control model flexibility.
  - Small Sigma_p = Rigid model / Large Sigma_p = Flexible model.

Q2. Why is Covariance written as K(x, x')? Shouldn't it be about y (or f)?
A: It IS about f, but parameterised by x.
  1. Definition of GP:
      A GP is a collection of random variables f(x).
      For any specific input x, f(x) is a random variable.

  2. Definition of Covariance Function:
      We calculate the covariance between two random variables f(x) and f(x').
      Cov(f(x), f(x')) = E[ (f(x) - m(x)) * (f(x') - m(x')) ]

  3. Notation:
      Since the value of this covariance depends entirely on the locations 
      x and x', we denote this function as k(x, x').

Q3. How does this connect to 'w' (Weight-space view)?
A: If we assume a linear model f(x) = x^T w, then the randomness of f
  comes solely from w. Thus:
  k(x, x') = E[ (x^T w)(x'^T w) ] = x^T * E[ww^T] * x' = x^T * Sigma_p * x'
------------------------------------------------------------------------------


================================================================================
"""


# ==========================================
# 1. Setup & Data Generation
# ==========================================
np.random.seed(42)

# True Function: y = -0.3 + 0.5x
true_w0 = -0.3
true_w1 = 0.5
noise_sigma = 0.2  # sigma_n (The noise standard deviation)

def generate_data(n=1):
    """
    Generates n data points from the true function with noise.
    Corresponds to the Likelihood assumption: y ~ N(X^T w, sigma_n^2 I)
    """
    X = np.random.uniform(-1, 1, n)
    Y = true_w0 + true_w1 * X + np.random.normal(0, noise_sigma, n)
    return X, Y

# ==========================================
# 2. Bayesian Inference Engine
# ==========================================
def bayesian_inference(X_data, Y_data, sigma_n, sigma_p_val=2.0):
    """
    Calculates the Posterior Distribution p(w|X, y) using Eq (2.8).
    
    Posterior ~ Likelihood x Prior
    Result is also Gaussian: N(w_bar, A^-1)
    """
    # 1. Construct Design Matrix Phi: Shape (n, 2) -> [[1, x1], [1, x2]...]
    if len(X_data) == 0:
        Phi = np.zeros((0, 2))
    else:
        Phi = np.vstack([np.ones(len(X_data)), X_data]).T
    
    # 2. Define Prior Matrix (Sigma_p)
    # w ~ N(0, sigma_p^2 I)
    Sigma_p = (sigma_p_val**2) * np.eye(2)
    Sigma_p_inv = np.linalg.inv(Sigma_p)
    
    # 3. Compute Posterior Covariance Matrix (A^-1)
    # A = (1/sigma_n^2) * Phi^T * Phi + Sigma_p^-1
    # Note: Phi^T * Phi is the "Information" from data. Sigma_p^-1 is "Information" from prior.
    A = (1/sigma_n**2) * (Phi.T @ Phi) + Sigma_p_inv
    A_inv = np.linalg.inv(A) # This is the Posterior Covariance
    
    # 4. Compute Posterior Mean (w_bar)
    # w_bar = (1/sigma_n^2) * A^-1 * Phi^T * y
    if len(Y_data) == 0:
        w_bar = np.zeros(2) # If no data, mean is 0 (Prior mean)
    else:
        w_bar = (1/sigma_n**2) * (A_inv @ Phi.T @ Y_data)
        
    return w_bar, A_inv

# ==========================================
# 3. Visualization Logic
# ==========================================
def plot_chart(ax_w, ax_d, w_mean, w_cov, X_data, Y_data, title):
    # [Left] Weight Space: Posterior distribution of w0, w1
    w0_vals = np.linspace(-1, 1, 100)
    w1_vals = np.linspace(-1, 1, 100)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    pos = np.dstack((W0, W1))
    rv = multivariate_normal(w_mean, w_cov)
    
    ax_w.contourf(W0, W1, rv.pdf(pos), cmap='Blues', levels=10)
    ax_w.scatter(true_w0, true_w1, c='red', marker='+', s=100, label='True w') 
    ax_w.set_xlabel('w0 (intercept)')
    ax_w.set_ylabel('w1 (slope)')
    ax_w.set_title(f'Weight Space {title}')
    ax_w.grid(True, alpha=0.3)

    # [Right] Data Space: Samples of lines drawn from Posterior
    # We sample 5 random (w0, w1) pairs from the posterior and plot them.
    w_samples = np.random.multivariate_normal(w_mean, w_cov, 5)
    x_range = np.linspace(-1, 1, 100)
    
    for w in w_samples:
        ax_d.plot(x_range, w[0] + w[1]*x_range, 'r-', alpha=0.5)
    
    # Plot observed data points
    if len(X_data) > 0:
        ax_d.scatter(X_data, Y_data, c='black', s=40, zorder=5)
        
    ax_d.set_ylim(-1, 1)
    ax_d.set_xlabel('x')
    ax_d.set_ylabel('y')
    ax_d.set_title(f'Data Space {title}')

# ==========================================
# 4. Execution Loop
# ==========================================
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

# Scenario 1: Prior (0 data points)
w_mean, w_cov = bayesian_inference([], [], noise_sigma)
plot_chart(axes[0,0], axes[0,1], w_mean, w_cov, [], [], "(Prior)")

# Scenario 2: 2 data points
X2, Y2 = generate_data(2)
w_mean, w_cov = bayesian_inference(X2, Y2, noise_sigma)
plot_chart(axes[1,0], axes[1,1], w_mean, w_cov, X2, Y2, "(2 observations)")

# Scenario 3: 20 data points
X20_new, Y20_new = generate_data(18)
X20 = np.concatenate([X2, X20_new])
Y20 = np.concatenate([Y2, Y20_new])
w_mean, w_cov = bayesian_inference(X20, Y20, noise_sigma)
plot_chart(axes[2,0], axes[2,1], w_mean, w_cov, X20, Y20, "(20 observations)")

plt.tight_layout()
plt.show()