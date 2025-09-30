import numpy as np
from scipy.linalg import eigh
from .utils import TORCH_CEL

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
    return - (pi.copy()) * (np.dot(X.copy(), theta.copy())-y.copy()) * (np.dot(X.copy(), lamb.copy())) - varrho * \
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
                     theta_init = None,
                     iterations=10000,
                     theta_solver='PGD',
                     delta_solver='PGD',
                     pi_solver='ED'):
    """
        TORCH solver for regression setting.
        All gradients, learning rates, and projection functions are internally hard-coded.

        Args:
            X (np.ndarray): Input feature matrix of shape (n, p).
            y (np.ndarray): Response matrix of shape (n,).
            q (int): Box quantile parameter for δ, controlling the outlier budget.
            projection_Omega (Callable): Projection operator to enforce θ within the composite null.
            varrho (float, optional): Penalty parameter for the Augmented Lagrangian. Default is 1.0.
            theta_init (np.ndarray, optional): Initial value for θ of shape (p,).
                Defaults to a zero vector or matrix if None.
            iterations (int, optional): Maximum number of iterations for the main TORCH loop. Default is 10000.
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
                - delta (np.ndarray): Final outlier indicator vector δ.
                - theta (np.ndarray): Final parameter estimate θ.
        """

    # 将所有的底层实现函数作为参数传入 TORCH 主函数
    return TORCH_CEL(
        X=X,
        y=y,
        q=q,
        varrho=varrho,
        theta_init=theta_init,
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