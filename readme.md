# TORCH_CEL

This Python package implements a scalable joint optimization framework for **compound empirical likelihood (CEL)**. It provides a general, nonparametric, and robust inference framework that allows for outliers in each estimating function, enhancing robustness under composite null hypotheses.

## Features

- Scalable joint optimization framework for CEL problems.
- Support for composite null hypothesis testing.
- Built-in handling of outliers.
- Ready-to-use examples for **location** and **regression** testing.

## Installation

You can install directly from GitHub:

```bash
pip install git+https://github.com/Hypon-Jun/TORCH-for-CEL
```

## Requirements

- Python 3.8+
- numpy 1.21+
- scipy 1.5+

## General CEL Problem Usage

For more flexible applications beyond the provided examples, users can call the general `TORCH_CEL` solver by providing their own gradient functions and projection operators:

```python
from TORCH_CEL import TORCH_CEL

pi, delta, theta = TORCH_CEL(X, y, q,
          # Core dependency functions
          structure_constraint,
          # Pi Solver dependencies
          grad_of_pi_func,
          # Delta Optimizer dependencies
          grad_of_delta_func,
          # Theta Optimizer dependencies
          projection_Omega_func, grad_of_theta_func,
          # Learning rate functions
          learning_rate_pi_func=None,
          learning_rate_delta_func=None,
          learning_rate_theta_func=None,
          # Penalty parameter of TORCH
          varrho = 1.0,
          # Solver choices and iteration parameters
          theta_init = None,
          iterations=10000,
          iterations_pi=10000,
          iterations_delta=10000,
          iterations_theta=10000,
          theta_solver='PGD',
          delta_solver='PGD',
          pi_solver='ED')
```

- TORCH Function Parameters

  - **`q`** (int): Outlier budget, i.e., the maximum number of samples allowed to be detected as outliers.
  - **`structure_constraint`** (Callable): Function representing structural constraints of the CEL problem.
  - **`grad_of_pi_func`** (Callable): Function computing the gradient with respect to $\pi$.  
    **Interface:** `grad_of_pi_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray`
  - **`grad_of_delta_func`** (Callable): Function computing the gradient with respect to $\delta$.  
    **Interface:** `grad_of_delta_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray`
  - **`projection_Omega_func`** (Callable): Projection operator for $\theta$ to ensure it lies within the composite null.
  - **`grad_of_theta_func`** (Callable): Function computing the gradient with respect to $\theta$.  
    **Interface:** `grad_of_theta_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray`
  - **`learning_rate_pi_func`** (Callable, optional): Learning rate for $\pi$.  
    If `None`, a default line-search strategy is used.  
    **Interface:** `learning_rate_pi_func(pi, delta, lamb, varrho, X, y, theta) -> float`
  - **`learning_rate_delta_func`** (Callable, optional): Learning rate for $\delta$.  
    If `None`, a default line-search strategy is used.  
    **Interface:** `learning_rate_delta_func(pi, delta, lamb, varrho, X, y, theta) -> float`
  - **`learning_rate_theta_func`** (Callable, optional): Learning rate for $\theta$.  
    If `None`, a default line-search strategy is used.  
    **Interface:** `learning_rate_theta_func(pi, delta, lamb, varrho, X, y, theta) -> float`
  - **`varrho`** (float, optional): Penalty parameter of TORCH. Default is 1.0.
  - **`theta_init`** (np.ndarray, optional): Initialization for $\theta$ of shape `(p,)` or `(p, m)`.  
    If `None`, defaults to a zero vector or matrix.
  - **`iterations`** (int, optional): Maximum number of iterations for the main TORCH loop.
  - **`iterations_pi`** (int, optional): Maximum number of iterations for the $\pi$ subproblem.
  - **`iterations_delta`** (int, optional): Maximum number of iterations for the $\delta$ subproblem.
  - **`iterations_theta`** (int, optional): Maximum number of iterations for the $\theta$ subproblem.
  - **`theta_solver`** (str, optional): Optimization algorithm for $\theta$. Default is `'PGD'`.  
    **Supported options:**  
    - `'PGD'`: Projected Gradient Descent  
    - `'APGD'`: Accelerated Projected Gradient Descent
  - **`delta_solver`** (str, optional): Optimization algorithm for $\delta$. Default is `'PGD'`.  
    **Supported options:**  
    - `'PGD'`: Projected Gradient Descent  
    - `'Overrelaxation'`: Over-relaxation update
  - **`pi_solver`** (str, optional): Optimization algorithm for $\pi$. Default is `'ED'`.  
    **Supported options:**  
    - `'ED'`: Entropic Descent  
    - `'AED'`: Accelerated Entropic Descent

  The following are example implementations of the required functions for using TORCH in a regression setting, including:

  ```python
  ## Regression Example: TORCH Dependency Functions
  
  # Example structure constraint
  def structure_constraint(pi, delta, X, y, theta):
      return np.dot(np.dot((X.copy()).T, np.diag(pi.copy() * (1 - delta.copy()))), (np.dot(X.copy(), theta.copy()) - y.copy()))
  
  # Example projection operator
  def projection_Omega_func(theta):
      projected_beta = beta.copy()
      projected_beta[0] = 9
      return projected_beta
  
  # Example gradient function for pi
  def grad_of_pi_func(pi, delta, lamb, varrho, X, y, theta):
      return -1 / pi + (1-delta) * (np.dot(X, theta)-y) * (np.dot(X,lamb)) + varrho * np.dot(np.dot(np.diag((1-delta) * (np.dot(X, theta) - y.copy())), X), np.dot(np.dot((X).T, np.diag(pi * (1 - delta))), (np.dot(X, theta) - y)))
  
  # Example gradient function for delta
  def grad_of_delta_func(pi, delta, lamb, varrho, X, y, theta):
      return - (pi.copy()) * (np.dot(X.copy(), theta.copy())-y.copy()) * (np.dot(X.copy(), lamb.copy())) - varrho * np.dot(np.dot(np.diag((pi.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy()), np.dot(np.dot((X.copy()).T, np.diag(pi.copy() * (1 - delta.copy()))), (np.dot(X.copy(), theta.copy()) - y.copy())))
  
  # Example gradient function for theta
  def grad_of_theta_func(pi, delta, lamb, varrho, X, y, theta):
      term = X.T @ (X * pi[:, np.newaxis] * (1 - delta.copy())[:, np.newaxis]) @ lamb
      term1 = X.T @ (X * pi[:, np.newaxis] * (1 - delta.copy())[:, np.newaxis])
      term2 = np.sum(X * (X @ theta - y)[:, np.newaxis] * pi[:, np.newaxis] * (1 - delta.copy())[:, np.newaxis], axis=0)
      grad = term + varrho * np.dot(term1, term2)
      return grad
  
  # Example learning rate function for pi
  def learning_rate_pi(pi, delta, lamb, varrho, X, y, theta):
      hessian = varrho * np.dot(np.dot(np.diag((1-delta.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy()), (np.dot(np.diag((1-delta.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy())).T)
      eigenvalues = eigh(hessian, eigvals_only=True)
      max_eigenvalue = np.max(eigenvalues)
      return 1 / max_eigenvalue
  
  # Example learning rate function for theta
  def learning_rate_theta(pi, delta, lamb, varrho, X, y, theta):
      summation = np.dot(X.T, X * pi[:, np.newaxis] * (1-delta)[:, np.newaxis])
      hessian = varrho * np.dot(summation, summation.T)
      eigenvalues = eigh(hessian, eigvals_only=True)
      max_eigenvalue = np.max(eigenvalues)
      return 1 / max_eigenvalue
  
  # Example learning rate function for delta
  def learning_rate_delta(pi, delta, lamb, varrho, X, y, theta):
      hessian = varrho * np.dot(np.dot(np.diag((pi.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy()), (np.dot(np.diag((pi.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy())).T)
      eigenvalues = eigh(hessian, eigvals_only=True)
      max_eigenvalue = np.max(eigenvalues)
      return 1 / max_eigenvalue
  ```

## Quick Start Examples

## Example1: Location Test

```python
import numpy as np
from TORCH_CEL import TORCH_location

# Define the projection operator for the composite null
def projection_Omega(theta):
    projected_theta = theta.copy()
    projected_theta[0] = 1
    return projected_theta

# Settings
n_features = 100
n_samples = 150
q = int(np.floor(n_samples * 0.15))  # number of outliers

# Generate data with outliers
true_mu = np.zeros(n_features)
X = np.random.multivariate_normal(mean=true_mu, cov=np.eye(n_features) * 4, size=n_samples)
X[-q:, :] = 50  # replace last q rows with outliers

# Run CEL-based location test
pi, delta, theta = TORCH_location(X, q, varrho=1, projection_Omega=projection_Omega)

# Test statistic
statistic = 2 * np.sum(-np.log(n_samples * pi))
print("Test statistic:", statistic)
```

## Example2: Regression Test

```python
import numpy as np
import time
from TORCH_CEL import TORCH_regression

# Define the projection operator for the composite null
def projection_Omega(theta):
    projected_beta = beta.copy()
    projected_beta[0] = 9
    return projected_beta

# Settings
n_features = 100
n_samples = 250
q = int(np.floor(n_samples * 0.15))  # number of outliers
varrho = 30

# Generate Toeplitz covariance matrix
tau = 0.2
Sigma = np.array([[tau ** abs(i-j) for j in range(n_features)] for i in range(n_features)])
X = np.random.multivariate_normal(np.zeros(n_features), Sigma, n_samples)
theta_true = np.ones(n_features) * 10
noise = np.random.normal(0, 5, n_samples)
y = X @ theta_true + noise

# Introduce outliers
y[-q:] = 1000

# Run CEL-based regression test
pi, delta, theta = TORCH_regression(X, y, q, varrho, projection_Omega)

# Test statistic
statistic = 2 * np.sum(-np.log(n_samples * pi))
print("Test statistic:", statistic)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you’d like to improve TORCH_CEL.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.