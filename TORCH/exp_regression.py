import numpy as np
from scipy.linalg import eigh
from .utils import TORCH

def structure_constraint(pi, delta, X, y, theta):
    return np.dot(np.dot((X.copy()).T, np.diag(pi.copy() * (1 - delta.copy()))), (np.dot(X.copy(), theta.copy()) - y.copy()))

def learning_rate_pi(pi, delta, lamb, varrho, X, y, theta):
    hessian = varrho * np.dot(np.dot(np.diag((1-delta.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy()), (np.dot(np.diag((1-delta.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy())).T)
    eigenvalues = eigh(hessian, eigvals_only=True)
    max_eigenvalue = np.max(eigenvalues)
    return 1 / max_eigenvalue

def learning_rate_theta(pi, delta, lamb, varrho, X, y, theta):
    summation = np.dot(X.T, X * pi[:, np.newaxis] * (1-delta)[:, np.newaxis])
    hessian = varrho * np.dot(summation, summation.T)
    eigenvalues = eigh(hessian, eigvals_only=True)
    max_eigenvalue = np.max(eigenvalues)
    return 1 / max_eigenvalue

def learning_rate_delta(pi, delta, lamb, varrho, X, y, theta):
    hessian = varrho * np.dot(np.dot(np.diag((pi.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy()), (np.dot(np.diag((pi.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy())).T)
    eigenvalues = eigh(hessian, eigvals_only=True)
    max_eigenvalue = np.max(eigenvalues)
    return 1 / max_eigenvalue

def grad_of_pi(pi, delta, lamb, varrho, X, y, theta):
    return -1 / pi.copy() + (1-delta.copy()) * (np.dot(X.copy(), theta.copy())-y.copy()) * (np.dot(X.copy(),lamb.copy())) + varrho * \
           np.dot(np.dot(np.diag((1-delta.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy()), np.dot(np.dot((X.copy()).T, np.diag(pi.copy() * (1 - delta.copy()))), (np.dot(X.copy(), theta.copy()) - y.copy())))


def grad_of_delta(pi, delta, lamb, varrho, X, y, theta):
    return - (pi.copy()) * (np.dot(X.copy(), theta.copy())-y.copy()) * (np.dot(X.copy(),lamb.copy())) - varrho * \
           np.dot(np.dot(np.diag((pi.copy()) * (np.dot(X.copy(), theta.copy()) - y.copy())), X.copy()), np.dot(np.dot((X.copy()).T, np.diag(pi.copy() * (1 - delta.copy()))), (np.dot(X.copy(), theta.copy()) - y.copy())))

def grad_of_theta(pi, delta, lamb, varrho, X, y, theta):
    term = X.T @ (X * pi[:, np.newaxis] * (1 - delta.copy())[:, np.newaxis]) @ lamb
    term1 = X.T @ (X * pi[:, np.newaxis] * (1 - delta.copy())[:, np.newaxis])
    term2 = np.sum(X * (X @ theta - y)[:, np.newaxis] * pi[:, np.newaxis] * (1 - delta.copy())[:, np.newaxis], axis=0)
    grad = term + varrho * np.dot(term1, term2)
    return grad

# def projection_Omega(beta):
#     projected_beta = beta.copy()
#     projected_beta[0] = 9
#     return projected_beta


# =========================================================
# 最终封装函数: TORCH_regression
# =========================================================

def TORCH_regression(X, y, q, varrho,
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
        y (np.ndarray): 标签矩阵 (n x m)。
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
        y=y,
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