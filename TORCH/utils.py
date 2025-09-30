from .acc_functions import PiOptimizer, ThetaOptimizer, DeltaOptimizer
import numpy as np


def safe_trace(a, b):
    """Compute np.dot(a, b).
    If result is scalar, return scalar.
    If result is matrix, return its trace.
    """
    val = np.dot(a, b)
    if np.ndim(val) == 0:  # scalar
        return val.item()
    else:  # array/matrix
        return np.trace(np.dot(a.T, b))

def TORCH(X, y, q,
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
          pi_solver='ED'):
    """
    TORCH main solver for CEL.

    Args:
        X (np.ndarray): Design Matrix (n x p).
        y (np.ndarray or None): Response Matrix (n x m).
        q (int): Outlier budget, i.e., the maximum number of samples allowed to be detected as outliers.

        structure_constraint (Callable): Function representing structural constraints of CEL.
        grad_of_pi_func (Callable): Function computing the gradient w.r.t. π.
            Interface: `grad_of_pi_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray`.
        grad_of_delta_func (Callable): Function computing the gradient w.r.t. δ.
            Interface: `grad_of_delta_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray`.
        projection_Omega_func (Callable): Projection operator for θ to ensure it lies
            within the composite null.
        grad_of_theta_func (Callable): Function computing the gradient w.r.t. θ.
            Interface: `grad_of_theta_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray`.

        learning_rate_pi_func (Callable, optional): Learning rate for π.
            If None, a default line-search strategy is used to find the learning rate for π.
            Interface: `learning_rate_pi_func(pi, delta, lamb, varrho, X, y, theta) -> float`.
        learning_rate_delta_func (Callable, optional): Learning rate for δ.
            If None, a default line-search strategy is used to find the learning rate for δ.
            Interface: `learning_rate_delta_func(pi, delta, lamb, varrho, X, y, theta) -> float`.
        learning_rate_theta_func (Callable, optional): Learning rate for θ.
            If None, a default line-search strategy is used to find the learning rate for θ.
            Interface: `learning_rate_theta_func(pi, delta, lamb, varrho, X, y, theta) -> float`.


        varrho (float, optional): Penalty parameter of TORCH. Default is 1.0.

        theta_init (np.ndarray, optional): Initialization for θ of shape (p,) or (p, m).
            If None, defaults to a zero vector or matrix.

        iterations (int, optional): Maximum number of iterations for the main TORCH loop.
        iterations_pi (int, optional): Maximum number of iterations for the π subproblem.
        iterations_delta (int, optional): Maximum number of iterations for the δ subproblem.
        iterations_theta (int, optional): Maximum number of iterations for the θ subproblem.
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
            - theta (np.ndarray): Final parameter estimate θ.
    """

    # --- 嵌套函数: 目标函数值 ---
    def function_value_func(pi, delta, lamb, varrho, X, y, theta):
        return -np.sum(np.log(pi)) + safe_trace(lamb, structure_constraint(pi, delta, X, y, theta)) + 0.5 * varrho * np.sum((structure_constraint(pi, delta, X, y, theta))**2)

    # --- 嵌套函数: Lambda 更新 ---
    def update_lamb_func(pi, delta, lamb, varrho, X, y, theta):
        return lamb + varrho * structure_constraint(pi, delta, X, y, theta)

    # 获取维度
    n, p = X.shape

    if y is None or y.ndim == 1:
        m = 1
    else:
        if y.shape[0] != n:
            raise ValueError("The number of samples in y must match the number of samples in X.")

        _, m = y.shape

    # 1. Initialize variables (Pi, Delta, Lambda, Theta)

    # Pi: dimension n (n x 1), initialized as uniform weights
    pi = np.ones(n) / n

    # Delta: dimension n (n x 1), initialized as zero vector
    delta = np.zeros(n)

    if theta_init is None:
        # Theta (Coef): dimension p x m, initialized as zero matrix
        # Lambda (Multiplier): dimension p x m, initialized as zero matrix
        if m == 1:
            theta = np.zeros(p)
            lamb = np.zeros(p)
        else:
            theta = np.zeros((p, m))
            lamb = np.zeros((p, m))
    else:
        theta = theta_init
        lamb = np.zeros_like(theta_init)

    # 2. Instantiate all optimizer classes (dependency injection)

    pi_optimizer = PiOptimizer(
        grad_of_pi_func=grad_of_pi_func,
        function_value_func=function_value_func,
        learning_rate_pi_func=learning_rate_pi_func
    )

    delta_optimizer = DeltaOptimizer(
        grad_of_delta_func=grad_of_delta_func,
        function_value_func=function_value_func,
        q = q,
        learning_rate_delta_func=learning_rate_delta_func
    )

    theta_optimizer = ThetaOptimizer(
        projection_Omega_func=projection_Omega_func,
        grad_of_theta_func=grad_of_theta_func,
        function_value_func=function_value_func,
        learning_rate_theta_func=learning_rate_theta_func
    )

    # 3. Main iteration loop (TORCH algorithm)
    for t in range(iterations):
        # Save previous values for convergence check
        tmp_pi = pi.copy()
        tmp_lamb = lamb.copy()
        tmp_delta = delta.copy()
        tmp_theta = theta.copy()

        # --- Step 1: Update Theta (p x m) ---
        # Use either PGD or APGD method
        if theta_solver == 'PGD':
            theta = theta_optimizer.update_theta_pgd(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                coef=theta, iterations=iterations_theta
            )
        elif theta_solver == 'APGD':
            theta = theta_optimizer.accelerated_PGD_theta(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                coef=theta, iterations=iterations_theta
            )
        else:
            raise ValueError("Invalid theta_solver. Use 'PGD' or 'APGD'.")

        # --- Step 2: Update Delta (n) ---
        # Use DeltaOptimizer with line search or accelerated overrelaxation
        if delta_solver == 'PGD':
            # 方案 1: PGD (带 Line Search 的 Proximal Gradient Descent)
            delta = delta_optimizer.update_delta_box_quantile(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                theta=theta, q=q, iterations=iterations_delta
            )
        elif delta_solver == 'Overrelaxation':
            # 方案 2: APGD (带 Overrelaxation 的 Accelerated PGD)
            delta = delta_optimizer.accelerated_delta_overrelaxation(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                coef=theta, q=q, iterations=iterations_delta
            )
        else:
            raise ValueError("Invalid delta_solver. Use 'PGD' or 'Overrelaxation'.")

        # --- Step 3: Update Pi (n) ---
        # Use Entropic Descent (ED) or Accelerated Entropic Descent (AED)
        if pi_solver == 'ED':
            pi = pi_optimizer.update_pi_mirror(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                theta=theta, iterations=iterations_pi
            )
        elif pi_solver == 'AED':
            pi = pi_optimizer.accelerated_entropic_descent(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                coef=theta, iterations=iterations_pi
            )
        else:
            raise ValueError("Invalid pi_solver. Use 'ED' or 'AED'.")
        print('stat', 2 * np.sum(-np.log(len(pi) * pi)))
        # --- Step 4: Update Lambda (p x m) ---
        lamb = update_lamb_func(
            pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y, theta=theta
        )

        # --- Step 5: Check convergence ---
        # Check if the objective function value is stable
        current_value = function_value_func(pi, delta, lamb, varrho, X, y, theta)
        previous_value = function_value_func(tmp_pi, tmp_delta, tmp_lamb, varrho, X, y, tmp_theta)

        if np.abs(current_value - previous_value) < 1e-6:
            print(f"TORCH converged at iteration {t + 1}. Function Value: {current_value:.6f}")
            break

        if t == iterations - 1:
            print(f"TORCH stopped after {iterations} iterations without convergence.")

    return pi, delta, theta


