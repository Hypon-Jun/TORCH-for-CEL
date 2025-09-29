import numpy as np


class ThetaOptimizer:
    """
    基于投影梯度下降 (PGD) 的优化器类。
    在初始化时接收所有模型相关的核心函数（依赖注入）。
    """

    def __init__(self,
                 learning_rate_theta_func,
                 projection_Omega_func,
                 grad_of_theta_func,
                 function_value_func):

        # 将所有用户提供的函数存储为实例属性
        self.learning_rate_theta_func = learning_rate_theta_func
        self.projection_Omega_func = projection_Omega_func
        self.grad_of_theta_func = grad_of_theta_func
        self.function_value_func = function_value_func

    # --- 核心函数 1: 内嵌的 line_search_theta ---
    # 现在它使用 self. 开头调用依赖函数
    def line_search_theta(self, pi, delta, lamb, varrho, X, y, theta, initial_rho, cons=0.5):
        rho = initial_rho
        count = 0

        # 调用注入的梯度函数
        grad = self.grad_of_theta_func(pi, delta, lamb, varrho, X, y, theta)

        while count < 1000:
            count = count + 1

            # 调用注入的投影函数
            tmp_theta = self.projection_Omega_func(theta.copy() - rho * grad)


            lhs = np.sum((tmp_theta - theta) ** 2) / 2


            rhs = rho * (self.function_value_func(pi, delta, lamb, varrho, X, y, tmp_theta) -
                         self.function_value_func(pi, delta, lamb, varrho, X, y, theta) -
                         np.dot(grad, tmp_theta - theta))

            if lhs >= rhs:
                break

            rho *= cons
        return rho

    # --- 核心函数 2: update_theta_PGD (PGD 主循环) ---
    # 现在它使用 self. 开头调用依赖函数，并调用内嵌的 line_search_theta
    def update_theta_PGD(self, pi, delta, lamb, varrho, X, y, theta, iterations):
        theta_update = theta.copy()

        # 调用注入的学习率函数
        initial_rho = self.learning_rate_theta_func(pi, delta, lamb, varrho, X, y, theta) * 5

        for t in range(iterations):
            tmp_theta = theta_update.copy()

            # **调用内嵌的 line_search_theta 方法**
            rho = self.line_search_theta(pi, delta, lamb, varrho, X, y, theta_update.copy(), initial_rho=initial_rho)

            # Projected GD 步骤
            theta_update = self.projection_Omega_func(
                theta_update.copy() - rho * self.grad_of_theta_func(pi.copy(), delta.copy(), lamb.copy(), varrho,
                                                                    X.copy(), y.copy(), theta_update.copy())
            )

            # 检查收敛条件
            if abs(self.function_value_func(pi, delta, lamb, varrho, X, y, theta_update) -
                   self.function_value_func(pi, delta, lamb, varrho, X, y, tmp_theta)) < 1e-6:
                break

        return theta_update


class PiOptimizer:
    """
    用于更新变量 pi 的优化器类。
    它将所有模型相关的核心函数（梯度、值函数、学习率等）作为依赖注入。
    """

    def __init__(self,
                 learning_rate_pi_func,
                 grad_of_pi_func,
                 function_value_func):

        # 将用户提供的函数存储为实例属性
        if learning_rate_pi_func is None:
            self.learning_rate_pi_func = self._default_learning_rate_pi
        else:
            self.learning_rate_pi_func = learning_rate_pi_func
        self.grad_of_pi_func = grad_of_pi_func
        self.function_value_func = function_value_func

    def _default_learning_rate_pi(self, pi, delta, lamb, varrho, X, y, theta):
        init_learning_rate = 1.0
        return self.line_search(pi, delta, lamb, varrho, X, y, theta, init_learning_rate)
    # --- 辅助函数：熵 (Entropy) ---
    @staticmethod
    def entropy(pi):
        """计算 pi 的熵 (Entropy)，使用 np.log 防止 log(0) 问题"""
        # 使用 np.maximum(pi, 1e-10) 来提高数值稳定性，避免 log(0)
        return np.sum(pi * np.log(np.maximum(pi, 1e-10)))

    # --- 辅助函数：熵的梯度 (Gradient of Entropy) ---
    @staticmethod
    def grad_entropy(pi):
        """计算熵函数的梯度"""
        # 同样使用 np.maximum(pi, 1e-10)
        return np.log(np.maximum(pi, 1e-10)) + 1

    # --- 核心函数 1: 内嵌的 line_search ---
    def line_search(self, pi, delta, lamb, varrho, X, y, theta, initial_rho, cons=0.5):
        rho = initial_rho
        count = 0

        # 调用注入的梯度函数
        grad = self.grad_of_pi_func(pi, delta, lamb, varrho, X, y, theta)
        # 调用内嵌的熵梯度函数
        grad_of_entropy = self.grad_entropy(pi)

        while count < 1000:
            count = count + 1

            # 更新 pi
            pi_tmp = pi * np.exp(-rho * grad)
            pi_tmp /= np.sum(pi_tmp)  # 归一化

            if np.any(np.isnan(pi_tmp)) or np.any(np.isinf(pi_tmp)):
                rho *= cons
                continue

            # 左侧：熵的二次近似下降项 (基于 Proximal Gradient 导出的线搜索条件)
            lhs = self.entropy(pi_tmp) - self.entropy(pi) - np.dot(grad_of_entropy, pi_tmp - pi)

            # 右侧：目标函数值下降项
            rhs = rho * (self.function_value_func(pi_tmp, delta, lamb, varrho, X, y, theta) -
                         self.function_value_func(pi, delta, lamb, varrho, X, y, theta) -
                         np.dot(grad, pi_tmp - pi))

            if lhs >= rhs:
                break

            rho *= cons
        return rho

    # --- 核心函数 2: update_pi (主循环) ---
    def update_pi(self, pi, delta, lamb, varrho, X, y, theta, iterations):
        pi_update = pi.copy()

        # 调用注入的学习率函数
        initial_rho = self.learning_rate_pi_func(pi_update, delta, lamb, varrho, X, y, theta) * 50

        for t in range(iterations):
            pi_tmp = pi_update.copy()

            # **调用内嵌的 line_search 方法**
            rho = self.line_search(pi_update.copy(), delta.copy(), lamb.copy(), varrho, X.copy(), y.copy(),
                                   theta.copy(), initial_rho=initial_rho)

            # 更新 pi 的步骤（基于熵正则化和指数梯度下降）
            grad = self.grad_of_pi_func(pi_update.copy(), delta.copy(), lamb.copy(), varrho, X.copy(), y.copy(),
                                        theta.copy())
            pi_update_0 = pi_update.copy() * np.exp(-rho * grad)
            pi_update = pi_update_0 / np.sum(pi_update_0)  # 归一化

            # 检查收敛条件
            if abs(self.function_value_func(pi_update, delta, lamb, varrho, X, y, theta) -
                   self.function_value_func(pi_tmp, delta, lamb, varrho, X, y, theta)) < 1e-6:
                break

        return pi_update


class DeltaOptimizer:
    """
    用于更新变量 delta 的优化器类。
    在初始化时接收所有模型相关的核心函数（梯度、值函数、学习率等）作为依赖注入。
    """

    def __init__(self,
                 learning_rate_delta_func,
                 grad_of_delta_func,
                 function_value_func):

        # 将用户提供的函数存储为实例属性
        self.learning_rate_delta_func = learning_rate_delta_func
        self.grad_of_delta_func = grad_of_delta_func
        self.function_value_func = function_value_func

    # --- 辅助函数：Box Quantile Thresholding ---
    @staticmethod
    def box_quantile_thresholding(y, q):
        """
        Implements the box-quantile thresholding operator Θ_{[0,1]}^{sharp}(y; q).
        This method is static as it doesn't depend on any instance variables (self).
        """
        # Get the order statistics (sorted values in descending order)
        # Sort y in descending order (y is typically a copy, so direct mutation is okay)
        sorted_indices = np.argsort(-y)
        y_sorted = y[sorted_indices]

        y_thresholded = np.zeros_like(y_sorted)

        # Apply the thresholding rule
        # Note: We use 1-based indexing logic from the original for clarity,
        # but Python is 0-indexed. q represents the *count* of elements.
        for i in range(len(y_sorted)):
            if i < q:
                # Top q elements
                if 0 < y_sorted[i] <= 1:
                    y_thresholded[i] = y_sorted[i]
                elif y_sorted[i] > 1:
                    y_thresholded[i] = 1
                else:  # y_sorted[i] <= 0 (Should not happen often for top elements, but handled)
                    y_thresholded[i] = 0
            else:  # Case: i >= q
                y_thresholded[i] = 0

        # Reconstruct the original order
        result = np.zeros_like(y)
        # Assign values back to their original positions using the inverse sorting
        result[sorted_indices] = y_thresholded

        return result

    # --- 核心函数 1: 内嵌的 line_search_delta ---
    def line_search_delta(self, pi, delta, lamb, varrho, X, y, theta, q, initial_rho, cons=0.5):
        rho = initial_rho
        count = 0

        # 调用注入的梯度函数
        grad = self.grad_of_delta_func(pi, delta, lamb, varrho, X, y, theta)

        while count < 1000:
            count = count + 1

            # 投影梯度步骤：等价于对 f(x) + (1/2\rho)*||x-y||^2 的 Proximal Mapping
            tmp = np.maximum(delta - rho * grad, 0)
            tmp_delta = self.box_quantile_thresholding(tmp, q)

            # Armijo 准则（用于投影梯度）的左侧：二次近似项
            lhs = np.sum((tmp_delta - delta) ** 2) / 2

            # Armijo 准则的右侧：函数值下降项
            rhs = rho * (self.function_value_func(pi, tmp_delta, lamb, varrho, X, y, theta) -
                         self.function_value_func(pi, delta, lamb, varrho, X, y, theta) -
                         np.dot(grad, tmp_delta - delta))

            if lhs >= rhs:
                break

            rho *= cons
        return rho

    # --- 核心函数 2: update_delta_optimal_rho (主循环) ---
    def update_delta_optimal_rho(self, pi, delta, lamb, varrho, X, y, theta, q, iterations):
        delta_update = delta.copy()

        # 调用注入的学习率函数
        initial_rho = self.learning_rate_delta_func(pi, varrho, X, y, theta) * 5

        for t in range(iterations):
            tmp_delta = delta_update.copy()

            # **调用内嵌的 line_search_delta 方法**
            rho = self.line_search_delta(pi, delta_update.copy(), lamb, varrho, X, y, theta, q, initial_rho=initial_rho)

            # 投影梯度下降步骤 (Proximal Gradient Descent)

            # 1. 梯度下降一步，并执行 ReLU (np.maximum(..., 0)) 以满足非负约束
            tmp = np.maximum(
                delta_update.copy() - rho * self.grad_of_delta_func(pi.copy(), delta_update.copy(), lamb.copy(), varrho,
                                                                    X.copy(), y.copy(), theta.copy()), 0)

            # 2. 调用内嵌的 Box Quantile Thresholding 投影
            delta_update = self.box_quantile_thresholding(tmp, q)

            # 检查收敛条件
            if abs(self.function_value_func(pi, delta_update, lamb, varrho, X, y, theta) -
                   self.function_value_func(pi, tmp_delta, lamb, varrho, X, y, theta)) < 1e-6:
                break

        return delta_update


