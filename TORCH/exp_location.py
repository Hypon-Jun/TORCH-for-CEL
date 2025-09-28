import numpy as np
from scipy.linalg import eigh
from .utils import TORCH

def structure_constraint(pi, delta, X, y, theta):
    return (1 - delta)[:, np.newaxis] * pi[:, np.newaxis] * (X - theta)  # Shape: (n, m)

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
    基于 Augumented Lagrangian Method (ALM) 的 TORCH 求解器，专用于具有 L2 loss 的回归问题。
    所有梯度、学习率和投影函数均使用内部硬编码的实现。

    Args:
        X (np.ndarray): 输入特征矩阵 (n x p)。
        q (int): Delta 的 Box Quantile 约束参数。
        varrho (float): 增广拉格朗日乘子法的惩罚超参数。
        iterations (int): 最大迭代次数。
        theta_solver (str): Theta 优化算法 ('PGD' 或 'APGD')。
        delta_solver (str): Delta 优化算法 ('PGD' 或 'Overrelaxation')。
        pi_solver (str): Pi 优化算法 ('ED' 或 'AED')。

    Returns:
        tuple: 最终的 pi, delta, theta 结果。
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