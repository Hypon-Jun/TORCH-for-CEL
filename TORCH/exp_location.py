import numpy as np
from scipy.linalg import eigh
from .utils import TORCH

def structure_constraint(pi, delta, X, y, theta):
    # Step 1: Compute X - mu (n x m)
    X_minus_mu = X - theta  # Shape: (n, m)

    # Step 2: Compute the weighted summation (n x m)
    weighted_summation = (1 - delta)[:, np.newaxis] * pi[:, np.newaxis] * X_minus_mu  # Shape: (n, m)

    # Step 3: Sum over all samples (n x m) to get a vector of size (m,)
    summation = np.sum(weighted_summation, axis=0)  # Shape: (m,)
    return summation  # Shape: (n, m)

def learning_rate_pi(pi, delta, lamb, varrho, X, y, theta):
    tmp = (X - theta) * ((1 - delta)[:, np.newaxis])
    hessian = varrho * (tmp @ tmp.T)
    eigenvalues = eigh(hessian, eigvals_only=True)
    max_eigenvalue = np.max(eigenvalues)
    return 1 / max_eigenvalue

def learning_rate_theta(pi, delta, lamb, varrho, X, y, theta):
    return 1 / (varrho * np.sum(pi.copy() * (1 - delta.copy()))**2)

def learning_rate_delta(pi, delta, lamb, varrho, X, y, theta):
    tmp = (X - theta) * (pi[:, np.newaxis])
    hessian = varrho * (tmp @ tmp.T)
    eigenvalues = eigh(hessian, eigvals_only=True)
    max_eigenvalue = np.max(eigenvalues)
    return 1 / max_eigenvalue

def grad_of_pi(pi, delta, lamb, varrho, X, y, theta):
    X_minus_mu = X - theta
    weighted_summation = pi[:, np.newaxis] * (1 - delta)[:, np.newaxis] * X_minus_mu
    summation = np.sum(weighted_summation, axis=0)
    grad_part1 = -1 / pi
    dot_term = (1 - delta) * np.dot(X_minus_mu, lamb)
    weighted_dot_term = varrho * (1 - delta) * (X_minus_mu @ summation)
    grad = grad_part1 + dot_term + weighted_dot_term
    return grad

def grad_of_delta(pi, delta, lamb, varrho, X, y, theta):
    X_minus_mu = X - theta
    weighted_summation = pi[:, np.newaxis] * (1 - delta)[:, np.newaxis] * X_minus_mu
    summation = np.sum(weighted_summation, axis=0)
    grad_part1 = - pi * np.dot(X_minus_mu, lamb)
    weighted_dot_term = varrho * (- pi) * (X_minus_mu @ summation)
    grad = grad_part1 + weighted_dot_term
    return grad

def grad_of_theta(pi, delta, lamb, varrho, X, y, theta):
    X_minus_mu = X - theta
    weighted_summation = pi[:, np.newaxis] * (1 - delta)[:, np.newaxis] * X_minus_mu
    summation = np.sum(weighted_summation, axis=0)
    grad_part1 = - np.sum(pi * (1 - delta)) * lamb
    grad_part2 = - varrho * np.sum(pi * (1 - delta)) * summation
    grad = grad_part1 + grad_part2
    return grad

def TORCH_location(X, q, varrho,
                     #projection to composite null
                     projection_Omega,
                     iterations=10000,
                     theta_solver='PGD',
                     delta_solver='PGD',
                     pi_solver='ED'):
    """
        TORCH solver for location setting.

        Args:
            X (np.ndarray): Input feature matrix of shape (n, p).
            q (int): Box quantile parameter for δ, controlling the outlier budget.
            varrho (float): Penalty parameter for the Augmented Lagrangian method.
            projection_Omega (Callable): Projection operator to enforce θ within the composite null space.
            iterations (int, optional): Maximum number of iterations for the TORCH loop. Default is 10000.
            theta_solver (str, optional): Optimization algorithm for θ. Default is 'PGD'.
                Supported options:
                - 'PGD': Projected Gradient Descent
                - 'APGD': Accelerated Projected Gradient Descent
            delta_solver (str, optional): Optimization algorithm for δ. Default is 'PGD'.
                Supported options:
                - 'PGD': Projected Gradient Descent
                - 'Overrelaxation': Over-relaxation update
            pi_solver (str, optional): Optimization algorithm for π. Default is 'ED'.
                Supported options:
                - 'ED': Entropic Descent
                - 'AED': Accelerated Entropic Descent

        Returns:
            tuple: (pi, delta, theta), where
                - pi (np.ndarray): Final optimized weight vector π.
                - delta (np.ndarray): Final outlier indicator δ.
                - theta (np.ndarray): Final location parameter θ.
        """

    # 将所有的底层实现函数作为参数传入 TORCH 主函数
    return TORCH(
        X=X,
        y=None,
        q=q,
        varrho=varrho,

        # 模型特有依赖 (这里是硬编码的)
        structure_constraint=structure_constraint,
        projection_Omega_func=projection_Omega,

        # 算法特有依赖 (这里是硬编码的)
        learning_rate_pi_func=learning_rate_pi,
        grad_of_pi_func=grad_of_pi,

        learning_rate_delta_func=learning_rate_delta,
        grad_of_delta_func=grad_of_delta,

        learning_rate_theta_func=learning_rate_theta,
        grad_of_theta_func=grad_of_theta,

        # 求解器和迭代参数
        iterations=iterations,
        theta_solver=theta_solver,
        delta_solver=delta_solver,
        pi_solver=pi_solver
    )