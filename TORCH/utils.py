from .acc_functions import PiOptimizer, ThetaOptimizer, DeltaOptimizer
import numpy as np


def safe_trace(a, b):
    """Compute np.dot(a, b).
    If result is scalar, return scalar.
    If result is matrix, return its trace."""
    val = np.dot(a, b)
    if np.ndim(val) == 0:  # scalar
        return val.item()
    else:  # array/matrix
        return np.trace(val)

def TORCH(X, y, q, varrho,
          # 底层依赖函数
          structure_constraint,
          # Pi Solver 依赖
          learning_rate_pi_func, grad_of_pi_func,
          # Delta Optimizer 依赖
          learning_rate_delta_func, grad_of_delta_func,
          # Theta Optimizer 依赖
          learning_rate_theta_func, projection_Omega_func, grad_of_theta_func,
          # 求解器选择和迭代参数
          iterations=10000,
          iterations_pi=10000,
          iterations_delta=10000,
          iterations_theta=10000,
          theta_solver='PGD',
          delta_solver='PGD',
          pi_solver='ED'):
    """
    TORCH (Theta/Delta/Pi/Lambda Alternating Optimization) 求解器的主要迭代循环。

    Args:
        X (np.ndarray): Design Matrix (n x p)。
        y (np.ndarray or None): Response Matrix (n x m)。
        q (int): Outlier Budget。
        varrho (float): Penalty Parameter of TORCH。
        ..._func: 所有的底层依赖函数。
        iterations (int): 最大迭代次数。
        theta_solver (str): Theta 优化算法 ('PGD' 或 'APGD')。
        delta_solver (str): Pi 优化算法 ('PGD' 或 'Overrelaxation')。
        pi_solver (str): Pi 优化算法 ('ED' 或 'AED')。
    """

    # --- 嵌套函数: 目标函数值 ---
    def function_value_func(pi, delta, lamb, varrho, X, y, theta):
        return -np.sum(np.log(pi)) + safe_trace(lamb, structure_constraint(pi, delta, X, y, theta)) + 0.5 * varrho * np.sum((structure_constraint(pi, delta, X, y, theta))**2)

    # --- 嵌套函数: Lambda 更新 ---
    def update_lamb_func(pi, delta, lamb, varrho, X, y, theta):
        return lamb + varrho * structure_constraint(pi, delta, X, y, theta)

    # 获取维度
    n, p = X.shape  # N=样本数, P=特征数

    if y is None:
        # 如果 y 为 None (无监督场景), 将任务数 M 设置为 1
        m = 1
        # 警告: 确保所有依赖 y 的底层函数都被修改为不使用或忽略 y
    else:
        # 确保 y 的形状是 (N, M)
        if y.shape[0] != n:
            raise ValueError("y 的样本数必须与 X 的样本数匹配。")

        _, m = y.shape  # M=任务数 (M >= 1)

    # 1. 初始化变量 (Pi, Delta, Lamb, Theta)

    # Pi: 维度 N (n x 1), 初始化为均匀权重
    pi = np.ones(n) / n

    # Delta: 维度 N (n x 1), 初始化为零向量
    delta = np.zeros(n)

    # Theta (Coef): 维度 P x M, 初始化为零矩阵
    if m == 1:
        theta = np.zeros(p)
    else:
        theta = np.zeros((p, m))

    # Lambda (乘子): 维度 P x M, 初始化为零矩阵
    if m == 1:
        lamb = np.zeros(p)
    else:
        lamb = np.zeros((p, m))

    # 2. 实例化所有优化器类 (依赖注入)
    # 注意：这里假设您的 PiSolver, DeltaOptimizer, ThetaOptimizer 类已在外部定义

    pi_optimizer = PiOptimizer(
        learning_rate_pi_func=learning_rate_pi_func,
        grad_of_pi_func=grad_of_pi_func,
        function_value_func=function_value_func
    )

    delta_optimizer = DeltaOptimizer(
        learning_rate_delta_func=learning_rate_delta_func,
        grad_of_delta_func=grad_of_delta_func,
        function_value_func=function_value_func
    )

    theta_optimizer = ThetaOptimizer(
        learning_rate_theta_func=learning_rate_theta_func,
        projection_Omega_func=projection_Omega_func,
        grad_of_theta_func=grad_of_theta_func,
        function_value_func=function_value_func
    )

    # 3. 主迭代循环 (TORCH 算法)
    for t in range(iterations):
        # 保存前一步的值用于收敛检查
        tmp_pi = pi.copy()
        tmp_lamb = lamb.copy()
        tmp_delta = delta.copy()
        tmp_theta = theta.copy()

        # --- 步骤 1: 更新 Theta (P x M) ---
        # 调用 PGD 或 APGD 方法
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

        # --- 步骤 2: 更新 Delta (N) ---
        # 使用 DeltaOptimizer 的 line search 或 accelerated overrelaxation
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

        # --- 步骤 3: 更新 Pi (N) ---
        # 调用 Entropic Descent (ED) 或 Accelerated Entropic Descent (AED) 方法
        if pi_solver == 'ED':
            pi = pi_optimizer.update_pi_mirror(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                theta=theta, iterations=iterations_pi
            )
        elif pi_solver == 'AED':
            # 假设 accelerated_entropic_descent 是 APGD 的一个变体
            pi = pi_optimizer.accelerated_entropic_descent(
                pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y,
                coef=theta, iterations=iterations_pi
            )
        else:
            raise ValueError("Invalid pi_solver. Use 'ED' or 'AED'.")

        # --- 步骤 4: 更新 Lambda (P x M) ---
        # 外部传入的独立函数（通常是 ADMM/Augmented Lagrangian 的对偶更新）
        lamb = update_lamb_func(
            pi=pi, delta=delta, lamb=lamb, varrho=varrho, X=X, y=y, theta=theta
        )

        # --- 步骤 5: 检查收敛条件 ---
        # 检查目标函数值是否稳定
        current_value = function_value_func(pi, delta, lamb, varrho, X, y, theta)
        previous_value = function_value_func(tmp_pi, tmp_delta, tmp_lamb, varrho, X, y, tmp_theta)

        if np.abs(current_value - previous_value) < 1e-6:
            print(f"TORCH converged at iteration {t + 1}. Function Value: {current_value:.6f}")
            break

        if t == iterations - 1:
            print(f"TORCH stopped after {iterations} iterations without convergence.")

    return pi, delta, theta


