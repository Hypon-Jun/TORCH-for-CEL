import numpy as np


def safe_trace(a, b):
    """Compute np.dot(a, b).
    If result is scalar, return scalar.
    If result is matrix, return its trace."""
    val = np.dot(a.T, b)
    if np.ndim(val) == 0:  # scalar
        return val.item()
    else:  # array/matrix
        return np.trace(np.dot(a.T, b))

class PiOptimizer:
    """
    Encapsulates the optimization logic for the variable π,
    including both standard and accelerated entropic regularized gradient descent.

    Dependency injection (user-provided functions):
    - learning_rate_pi_func: function providing the initial learning rate or step size.
    - grad_of_pi_func: function computing the gradient with respect to π.
    - function_value_func: function computing the objective function value.
    """

    def __init__(self,
                 grad_of_pi_func,
                 function_value_func,
                 learning_rate_pi_func = None):
        """
        Initialize the PiOptimizer with user-supplied functions.

        Args:
            grad_of_pi_func (Callable): Gradient function of π.
                Signature: grad_of_pi_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray
            function_value_func (Callable): Objective function evaluator.
                signature: function_value_func(pi, delta, lamb, varrho, X, y, theta) -> float
            learning_rate_pi_func (Callable, optional): Function to compute the learning rate for π.
                Signature: learning_rate_pi_func(pi, delta, lamb, varrho, X, y, theta) -> float
                If None, a default line-search strategy (_default_learning_rate_pi) is used.
        """
        if learning_rate_pi_func is None:
            self.learning_rate_pi_func = self._default_learning_rate_pi
        else:
            self.learning_rate_pi_func = learning_rate_pi_func
        self.grad_of_pi_func = grad_of_pi_func
        self.function_value_func = function_value_func

    def _default_learning_rate_pi(self, pi, delta, lamb, varrho, X, y, theta):
        init_learning_rate = 1.0
        return self.line_search_mirror(pi, delta, lamb, varrho, X, y, theta, init_learning_rate)

    # =======================================================
    # II. 基础辅助函数
    # =======================================================

    @staticmethod
    def mirror_function(pi):
        return np.sum(pi * np.log(np.maximum(pi, 1e-10)))

    @staticmethod
    def grad_mirror_function(pi):
        return np.log(np.maximum(pi, 1e-10)) + 1

    @staticmethod
    def entropy(pi):
        # 别名，因为熵即为常用的镜面函数
        return PiOptimizer.mirror_function(pi)

    @staticmethod
    def grad_entropy(pi):
        # 别名，因为熵的梯度即为常用的镜面函数的梯度
        return PiOptimizer.grad_mirror_function(pi)

    # =======================================================
    # III. 加速算法辅助函数 (作为类方法)
    # =======================================================

    def breg_loss(self, parameter1, parameter2, loss, grad_loss):
        return loss(parameter1) - loss(parameter2) - np.dot(grad_loss(parameter2), parameter1 - parameter2)

    def breg_mirror_function(self, parameter1, parameter2):
        # 使用类静态方法
        return self.mirror_function(parameter1) - self.mirror_function(parameter2) - np.dot(
            self.grad_mirror_function(parameter2), parameter1 - parameter2)

    def breg_psi(self, parameter1, parameter2, mu, loss, grad_loss):
        return self.breg_loss(parameter1, parameter2, loss, grad_loss) - mu * self.breg_mirror_function(parameter1,
                                                                                                        parameter2)

    def C_mirror_function(self, parameter1, parameter2, theta):
        return theta * self.mirror_function(parameter1) + (1 - theta) * self.mirror_function(
            parameter2) - self.mirror_function(theta * parameter1 + (1 - theta) * parameter2)

    def E_function(self, optimal_beta, gamma, mu, loss, grad_loss):
        return self.breg_psi(optimal_beta, gamma, mu, loss, grad_loss)

    def R_function(self, gamma, alpha, alpha_new, beta, beta_new, theta, mu, rho, loss, grad_loss):
        return (theta ** 2) * rho * self.breg_mirror_function(alpha_new, alpha) - self.breg_psi(beta_new, gamma, mu,
                                                                                                loss, grad_loss) \
               + (1 - theta) * self.breg_psi(beta, gamma, mu, loss, grad_loss) + mu * self.C_mirror_function(alpha_new,
                                                                                                             beta,
                                                                                                             theta)

    @staticmethod
    def theta_update(theta_prev, mu_prev, rho_prev, rho):
        prev = theta_prev * (rho_prev * theta_prev + mu_prev)
        return (- prev + np.sqrt(prev ** 2 + 4 * rho * prev)) / (2 * rho)

    @staticmethod
    def gamma_update(beta, alpha, theta):
        return (1 - theta) * beta.copy() + theta * alpha.copy()

    @staticmethod
    def alpha_update(gamma, alpha, theta, mu, rho, grad_function):
        grad_gamma = grad_function(gamma)

        exponent = - grad_gamma / (mu + theta * rho)
        tmp = (gamma ** (mu / (mu + theta * rho))) * (alpha ** (theta * rho / (mu + theta * rho))) * np.exp(exponent)

        return tmp / np.sum(tmp)

    @staticmethod
    def beta_update(beta, alpha_new, theta):
        return (1 - theta) * beta.copy() + theta * alpha_new.copy()

    # =======================================================
    # IV. 加速算法 Line Search 内部函数
    # =======================================================

    # 辅助函数：创建损失和梯度函数（因为它们在外部被定义，这里只是一个包装器）
    def _create_loss_wrapper(self, delta, lamb, varrho, X, y, coef):
        # 封装 self.function_value_func
        return lambda pi: self.function_value_func(pi, delta, lamb, varrho, X, y, coef)

    def _create_grad_wrapper(self, delta, lamb, varrho, X, y, coef):
        # 封装 self.grad_of_pi_func
        return lambda pi: self.grad_of_pi_func(pi, delta, lamb, varrho, X, y, coef)

    # --- Line Search for mu ---
    def line_search_mu(self, beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                       alpha, beta, loss_function, grad_function, con=3.4, search_iterations=100000):
        U_values = []
        mu_values = []

        for search_iter in range(search_iterations):
            theta_try = self.theta_update(theta, mu, rho, rho_try)
            gamma_try = self.gamma_update(beta, alpha, theta_try)
            alpha_try = self.alpha_update(gamma_try, alpha, theta_try, mu_try, rho_try, grad_function)
            beta_try = self.beta_update(beta, alpha_try, theta_try)

            E_try = self.E_function(beta_optimal_prev, gamma_try, mu_try, loss_function, grad_function)
            R_try = self.R_function(gamma_try, alpha, alpha_try, beta, beta_try, theta_try, mu_try, rho_try,
                                    loss_function, grad_function)

            RE_try = R_try + theta_try * E_try
            U_values.append(RE_try)
            mu_values.append(mu_try)

            if mu_try > rho_try or R_try < 0 or search_iter == search_iterations - 1:
                U_array = -np.array(U_values)
                min_value = np.min(U_array)

                tolerance = 1e-2
                if min_value > 0:
                    lower_bound = min_value
                    upper_bound = (1 + tolerance) * min_value
                else:
                    lower_bound = min_value
                    upper_bound = (1 - tolerance) * min_value

                indices = np.where((U_array >= lower_bound) & (U_array <= upper_bound))[0]
                max_index = np.max(indices) if len(indices) > 0 else 0

                return mu_values[max_index]

            mu_try *= con
        return mu_try

    # --- Line Search for rho ---
    def line_search_rho(self, beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                        alpha, beta, loss_function, grad_function, con=0.4, search_iterations=100000):

        best_rho = rho_try

        for search_iter in range(search_iterations):
            theta_try = self.theta_update(theta, mu, rho, rho_try)
            gamma_try = self.gamma_update(beta, alpha, theta_try)
            alpha_try = self.alpha_update(gamma_try, alpha, theta_try, mu_try, rho_try, grad_function)
            beta_try = self.beta_update(beta, alpha_try, theta_try)

            R_try = self.R_function(gamma_try, alpha, alpha_try, beta, beta_try, theta_try, mu_try, rho_try,
                                    loss_function, grad_function)

            if rho_try < mu_try or R_try < 0:
                best_rho = rho_try / con
                break
            else:
                best_rho = rho_try

            rho_try *= con
        return best_rho

    # --- Line Search Wrapper ---
    def line_search_for_acc_alg(self, beta_optimal_prev, alpha, beta, theta, mu, rho, loss_function, grad_function):
        # 1. 初始猜测
        mu_try = mu * 0.4
        rho_try = rho * 1.5

        # 2. rho fixed, search mu
        mu_try = self.line_search_mu(beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                                     alpha, beta, loss_function, grad_function, con=2.8, search_iterations=3)

        # 3. mu fixed, search rho
        rho_try = self.line_search_rho(beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                                       alpha, beta, loss_function, grad_function, con=0.2, search_iterations=3)

        return mu_try, rho_try

    # =======================================================
    # V. PGD 主循环 (非加速算法 - 兼容旧版)
    # =======================================================
    # 整合原来的 line_search 和 update_pi 逻辑
    def line_search_mirror(self, pi, delta, lamb, varrho, X, y, theta, initial_learning_rate, cons=0.5):
        learning_rate = initial_learning_rate
        count = 0

        grad = self.grad_of_pi_func(pi, delta, lamb, varrho, X, y, theta)
        grad_of_entropy = self.grad_entropy(pi)

        while count < 1000:
            count = count + 1
            pi_tmp = pi * np.exp(-learning_rate * grad)
            pi_tmp /= np.sum(pi_tmp)

            if np.any(np.isnan(pi_tmp)) or np.any(np.isinf(pi_tmp)):
                learning_rate *= cons
                continue

            lhs = self.entropy(pi_tmp) - self.entropy(pi) - np.dot(grad_of_entropy, pi_tmp - pi)

            rhs = learning_rate * (self.function_value_func(pi_tmp, delta, lamb, varrho, X, y, theta) -
                         self.function_value_func(pi, delta, lamb, varrho, X, y, theta) -
                         np.dot(grad, pi_tmp - pi))

            if lhs >= rhs:
                break
            learning_rate *= cons
        return learning_rate

    def update_pi_mirror(self, pi, delta, lamb, varrho, X, y, theta, iterations=10000):
        pi_update = pi.copy()
        initial_learning_rate = self.learning_rate_pi_func(pi_update, delta, lamb, varrho, X, y, theta) * 5

        for t in range(iterations):
            pi_tmp = pi_update.copy()

            # 调用内嵌的 line_search_mirror 方法
            rho = self.line_search_mirror(pi_update, delta, lamb, varrho, X, y,
                                       theta, initial_learning_rate=initial_learning_rate)

            grad = self.grad_of_pi_func(pi_update, delta, lamb, varrho, X, y,
                                        theta)
            pi_update_0 = pi_update.copy() * np.exp(-rho * grad)
            pi_update = pi_update_0 / np.sum(pi_update_0)  # 归一化

            if abs(self.function_value_func(pi_update, delta, lamb, varrho, X, y, theta) -
                   self.function_value_func(pi_tmp, delta, lamb, varrho, X, y, theta)) < 1e-6:
                break

        return pi_update

    # =======================================================
    # VI. 主加速算法
    # =======================================================
    def accelerated_entropic_descent(self, pi, delta, lamb, varrho, X, y, coef, iterations=10000):

        # 封装损失和梯度函数，以便在加速算法中传递
        loss = self._create_loss_wrapper(delta, lamb, varrho, X, y, coef)
        grad_loss = self._create_grad_wrapper(delta, lamb, varrho, X, y, coef)

        # 2. 初始化变量 (严格遵循原代码逻辑)
        alpha = pi.copy()
        beta = pi.copy()
        theta = 1.0  # theta0
        mu = 1.0  # mu0

        # 初始 rho 设定
        inverse_rho = self.learning_rate_pi_func(pi, delta, lamb, varrho, X, y, coef)
        rho = 1.0 / (inverse_rho * 5)  # rho0

        for iter in range(iterations):
            # 记录前一步的值用于收敛检查和更新
            alpha_tmp = alpha.copy()
            # beta_tmp = beta.copy()
            # gamma_tmp = self.gamma_update(beta, alpha, theta)

            theta_tmp = theta
            mu_tmp = mu
            rho_tmp = rho

            # --- 步骤 1: 更新变量 ---
            gamma = self.gamma_update(beta, alpha, theta)
            alpha = self.alpha_update(gamma, alpha, theta, mu, rho, grad_loss)
            beta = self.beta_update(beta, alpha, theta)

            # --- 步骤 2: 检查收敛 ---
            if np.abs(loss(alpha) - loss(alpha_tmp)) < 1e-6:
               break

            # --- 步骤 3: Line Search 更新参数 ---
            beta_optimal_prev = alpha.copy()
            # 传入当前的 theta, mu, rho (即 tmp_后缀变量的值)
            mu, rho = self.line_search_for_acc_alg(beta_optimal_prev, alpha, beta, theta, mu, rho, loss, grad_loss)

            # --- 步骤 4: 更新 theta ---
            # 使用前一步的值 (theta_tmp, mu_tmp, rho_tmp) 和线搜索后的新 rho
            theta = self.theta_update(theta_tmp, mu_tmp, rho_tmp, rho)



        return alpha





class ThetaOptimizer:
    """
    Encapsulates the optimization logic for the variable θ,
    including both standard (PGD) and accelerated (APGD) algorithms.

    Dependency injection (user-provided functions):
    - learning_rate_theta_func: function providing the initial learning rate or step size for θ.
    - projection_Omega_func: projection operator to ensure θ remains in the composite null.
    - grad_of_theta_func: function computing the gradient with respect to θ.
    - function_value_func: function computing the objective function value.
    """

    def __init__(self,
                 projection_Omega_func,
                 grad_of_theta_func,
                 function_value_func,
                 learning_rate_theta_func=None):
        """
            Initialize the ThetaOptimizer with user-supplied functions.

            Args:
                projection_Omega_func (Callable): Projection operator to ensure θ stays within
                    the feasible set Ω. Signature: projection_Omega_func(theta) -> np.ndarray
                grad_of_theta_func (Callable): Function computing the gradient w.r.t. θ.
                    Signature: grad_of_theta_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray
                function_value_func (Callable): Function computing the objective value.
                    Signature: function_value_func(pi, delta, lamb, varrho, X, y, theta) -> float
                learning_rate_theta_func (Callable, optional): Function to compute the learning rate
                    for θ updates. Signature: learning_rate_theta_func(pi, delta, lamb, varrho, X, y, theta) -> float
                    If None, a default line-search strategy (_default_learning_rate_theta) is used.

            Notes:
                This class uses dependency injection to allow flexible projection
                implementation provided by the user.
        """

        # 存储用户提供的核心依赖
        if learning_rate_theta_func is None:
            self.learning_rate_theta_func = self._default_learning_rate_theta
        else:
            self.learning_rate_theta_func = learning_rate_theta_func



        self.projection_Omega_func = projection_Omega_func
        self.grad_of_theta_func = grad_of_theta_func
        self.function_value_func = function_value_func

    def _default_learning_rate_theta(self, pi, delta, lamb, varrho, X, y, theta):
        init_learning_rate = 1.0
        return self.line_search_theta(pi, delta, lamb, varrho, X, y, theta, init_learning_rate)

    # =======================================================
    # II. 基础辅助函数 (L2 镜面函数)
    # =======================================================

    @staticmethod
    def l2(pi):
        return np.sum(pi ** 2) / 2


    # =======================================================
    # III. 加速算法辅助函数 (APGD)
    # =======================================================

    # --- APGD Bregman/Quadratic Approximation Functions ---

    def breg_loss(self, parameter1, parameter2, loss, grad_loss):
        return loss(parameter1) - loss(parameter2) - safe_trace(grad_loss(parameter2), parameter1 - parameter2)

    def breg_l2(self, parameter1, parameter2):
        return np.sum((parameter1 - parameter2) ** 2) / 2

    def breg_psi(self, parameter1, parameter2, mu, loss, grad_loss):
        return self.breg_loss(parameter1, parameter2, loss, grad_loss) - mu * self.breg_l2(
            parameter1, parameter2)

    def C_2(self, parameter1, parameter2, theta):
        return theta * self.l2(parameter1) + (1 - theta) * self.l2(parameter2) - self.l2(
            theta * parameter1 + (1 - theta) * parameter2)

    def E_function(self, optimal_beta, gamma, mu, loss, grad_loss):
        return self.breg_psi(optimal_beta, gamma, mu, loss, grad_loss)

    def R_function(self, gamma, alpha, alpha_new, beta, beta_new, theta, mu, rho, loss, grad_loss):
        return (theta ** 2) * rho * self.breg_l2(alpha_new, alpha) - self.breg_psi(beta_new, gamma, mu,
                                                                                   loss, grad_loss) \
               + (1 - theta) * self.breg_psi(beta, gamma, mu, loss, grad_loss) + mu * self.C_2(alpha_new,
                                                                                               beta, theta)

    # --- APGD Update Steps ---

    @staticmethod
    def theta_update(theta_prev, mu_prev, rho_prev, rho):
        prev = theta_prev * (rho_prev * theta_prev + mu_prev)
        return (- prev + np.sqrt(np.maximum(prev ** 2 + 4 * rho * prev, 0))) / (2 * rho)

    @staticmethod
    def gamma_update(beta, alpha, theta):
        return (1 - theta) * beta.copy() + theta * alpha.copy()

    def alpha_update(self, gamma, alpha, theta, mu, rho, grad_function):
        return self.projection_Omega_func(
            (mu * gamma.copy() + theta * rho * alpha.copy() - grad_function(gamma.copy())) / (mu + theta * rho)
        )

    @staticmethod
    def beta_update(beta, alpha_new, theta):
        return (1 - theta) * beta.copy() + theta * alpha_new.copy()

    # =======================================================
    # IV. PGD (非加速算法) 逻辑
    # =======================================================

    def line_search_theta(self, pi, delta, lamb, varrho, X, y, theta, initial_learning_rate, cons=0.5):
        learning_rate = initial_learning_rate
        count = 0

        # 调用注入的梯度函数
        grad = self.grad_of_theta_func(pi, delta, lamb, varrho, X, y, theta)

        while count < 1000:
            count = count + 1

            # 调用注入的投影函数
            tmp_theta = self.projection_Omega_func(theta.copy() - learning_rate * grad)

            lhs = np.sum((tmp_theta - theta) ** 2) / 2

            rhs = learning_rate * (self.function_value_func(pi, delta, lamb, varrho, X, y, tmp_theta) -
                         self.function_value_func(pi, delta, lamb, varrho, X, y, theta) -
                         safe_trace(grad, tmp_theta - theta))

            if lhs >= rhs:
                break

            learning_rate *= cons
        return learning_rate

    def update_theta_pgd(self, pi, delta, lamb, varrho, X, y, coef, iterations=10000):
        theta_update = coef.copy()

        initial_learning_rate = self.learning_rate_theta_func(pi, delta, lamb, varrho, X, y, coef) * 5

        for t in range(iterations):
            tmp_theta = theta_update.copy()

            # 调用内嵌的 line_search_theta 方法
            learning_rate = self.line_search_theta(pi, delta, lamb, varrho, X, y, theta_update.copy(), initial_learning_rate=initial_learning_rate)

            # Projected GD 步骤
            grad = self.grad_of_theta_func(pi, delta, lamb, varrho, X, y, theta_update)
            theta_update = self.projection_Omega_func(theta_update - learning_rate * grad)

            # 检查收敛条件
            if abs(self.function_value_func(pi, delta, lamb, varrho, X, y, theta_update) -
                   self.function_value_func(pi, delta, lamb, varrho, X, y, tmp_theta)) < 1e-6:
                break

        return theta_update

    # =======================================================
    # V. 加速算法 Line Search 完整实现
    # =======================================================

    # 辅助函数：创建损失和梯度函数（因为它们在外部被定义，这里只是一个包装器）
    def _create_loss_wrapper(self, pi, delta, lamb, varrho, X, y):
        # 封装 self.function_value_func
        return lambda coef: self.function_value_func(pi, delta, lamb, varrho, X, y, coef)

    def _create_grad_wrapper(self, pi, delta, lamb, varrho, X, y):
        # 封装 self.grad_of_theta_func
        return lambda coef: self.grad_of_theta_func(pi, delta, lamb, varrho, X, y, coef)

    # --- Line Search for mu (APGD) ---
    def line_search_mu(self, beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                       alpha, beta, loss_function, grad_function, con=3.4,
                       search_iterations=3):  # Note: search_iterations set to 3 as per original logic
        U_values = []
        mu_values = []

        for search_iter in range(search_iterations):
            theta_try = self.theta_update(theta, mu, rho, rho_try)
            gamma_try = self.gamma_update(beta, alpha, theta_try)
            alpha_try = self.alpha_update(gamma_try, alpha, theta_try, mu_try, rho_try, grad_function)
            beta_try = self.beta_update(beta, alpha_try, theta_try)

            E_try = self.E_function(beta_optimal_prev, gamma_try, mu_try, loss_function, grad_function)
            R_try = self.R_function(gamma_try, alpha, alpha_try, beta, beta_try, theta_try, mu_try, rho_try,
                                    loss_function, grad_function)

            RE_try = R_try + theta_try * E_try
            U_values.append(RE_try)
            mu_values.append(mu_try)

            if mu_try > rho_try or R_try < 0 or search_iter == search_iterations - 1:
                U_array = -np.array(U_values)
                min_value = np.min(U_array)

                # 寻找在 [min_value, (1 + tolerance) * min_value] 范围内的最大索引
                tolerance = 1e-2
                if min_value > 0:
                    lower_bound = min_value
                    upper_bound = (1 + tolerance) * min_value
                else:
                    lower_bound = min_value
                    upper_bound = (1 - tolerance) * min_value

                indices = np.where((U_array >= lower_bound) & (U_array <= upper_bound))[0]
                max_index = np.max(indices) if len(indices) > 0 else 0

                return mu_values[max_index]

            mu_try *= con
        return mu_try

    # --- Line Search for rho (APGD) ---
    def line_search_rho(self, beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                        alpha, beta, loss_function, grad_function, con=0.4,
                        search_iterations=2):  # Note: search_iterations set to 2 as per original logic

        best_rho = rho_try

        for search_iter in range(search_iterations):
            theta_try = self.theta_update(theta, mu, rho, rho_try)
            gamma_try = self.gamma_update(beta, alpha, theta_try)
            alpha_try = self.alpha_update(gamma_try, alpha, theta_try, mu_try, rho_try, grad_function)
            beta_try = self.beta_update(beta, alpha_try, theta_try)

            R_try = self.R_function(gamma_try, alpha, alpha_try, beta, beta_try, theta_try, mu_try, rho_try,
                                    loss_function, grad_function)

            if rho_try < mu_try or R_try < 0:
                best_rho = rho_try / con
                break
            else:
                best_rho = rho_try

            rho_try *= con
        return best_rho

    # --- Line Search Wrapper (APGD) ---
    def line_search_acc_alg(self, beta_optimal_prev, alpha, beta, theta, mu, rho, loss_function, grad_function):
        # 1. 初始猜测 (严格遵循原代码)
        mu_try = mu * 0.4
        rho_try = rho * 2

        # 2. rho fixed, search mu
        mu_try = self.line_search_mu(beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                                     alpha, beta, loss_function, grad_function, con=3.4, search_iterations=3)

        # 3. mu fixed, search rho
        rho_try = self.line_search_rho(beta_optimal_prev, theta, mu, rho, rho_try, mu_try,
                                       alpha, beta, loss_function, grad_function, con=0.4, search_iterations=2)

        return mu_try, rho_try

    # =======================================================
    # VI. 主加速算法 (APGD)
    # =======================================================
    def accelerated_PGD_theta(self, pi, delta, lamb, varrho, X, y, coef, iterations=10000):

        # 1. 内部柯里化（包装）损失和梯度函数
        loss = self._create_loss_wrapper(pi, delta, lamb, varrho, X, y)
        grad_loss = self._create_grad_wrapper(pi, delta, lamb, varrho, X, y)

        # 2. 初始化变量 (严格遵循原代码逻辑)
        alpha = coef.copy()
        beta = coef.copy()
        theta = 1.0

        lr = self.learning_rate_theta_func(pi, delta, lamb, varrho, X, y, coef)
        mu = 0.0
        rho = 1.0 / lr

        for iter in range(iterations):
            # 记录前一步的值
            alpha_tmp = alpha.copy()
            theta_tmp = theta
            mu_tmp = mu
            rho_tmp = rho

            # --- 1. 更新变量 ---
            gamma = self.gamma_update(beta, alpha, theta)
            alpha = self.alpha_update(gamma, alpha, theta, mu, rho, grad_loss)
            beta = self.beta_update(beta, alpha, theta)

            # --- 2. 检查收敛 ---
            if np.abs(loss(alpha) - loss(alpha_tmp)) < 1e-3:
                return alpha

            # --- 3. Line Search/参数更新 ---
            beta_optimal_prev = alpha.copy()

            if iter == 26:
                mu = 1e-4

            if iter > 26:
                mu, rho = self.line_search_acc_alg(beta_optimal_prev, alpha, beta, theta, mu, rho, loss, grad_loss)
            elif iter < 25:
                mu = 0.0


            # --- 4. 更新 theta ---
            theta = self.theta_update(theta_tmp, mu_tmp, rho_tmp, rho)



        return alpha



class DeltaOptimizer:
    """
    Encapsulates the optimization logic for the variable δ.

    Dependency injection (user-provided functions):
    - learning_rate_delta_func: function providing the initial learning rate or step size for δ.
    - grad_of_delta_func: function computing the gradient with respect to δ.
    - function_value_func: function computing the objective function value.
    - q: outlier budget, controlling the maximum number of allowed outliers.
    """

    def __init__(self,
                 grad_of_delta_func,
                 function_value_func,
                 q,
                 learning_rate_delta_func):

        """
                Initialize the DeltaOptimizer with user-provided functions.

                Args:
                    grad_of_delta_func (Callable): Function computing the gradient w.r.t. δ.
                        Signature: grad_of_delta_func(pi, delta, lamb, varrho, X, y, theta) -> np.ndarray
                    function_value_func (Callable): Function computing the objective value.
                        Signature: function_value_func(pi, delta, lamb, varrho, X, y, theta) -> float
                    q (int): Outlier budget, used in δ updates to control the number of allowed outliers.
                    learning_rate_delta_func (Callable, optional): Function to compute the learning rate
                        for δ updates. Signature: learning_rate_delta_func(pi, delta, lamb, varrho, X, y, theta) -> float
                        If None, a default learning rate strategy (_default_learning_rate_delta) is used.


        """

        # 将用户提供的函数存储为实例属性
        if learning_rate_delta_func is None:
            self.learning_rate_delta_func = self._default_learning_rate_delta
        else:
            self.learning_rate_delta_func = learning_rate_delta_func
        self.grad_of_delta_func = grad_of_delta_func
        self.function_value_func = function_value_func
        self.q = q

    def _default_learning_rate_delta(self, pi, delta, lamb, varrho, X, y, theta):
        init_learning_rate = 1.0
        return self.line_search_delta(pi, delta, lamb, varrho, X, y, theta, init_learning_rate, self.q)

    # --- 辅助函数：Box Quantile Thresholding ---
    @staticmethod
    def box_quantile_thresholding(y, q):

        sorted_indices = np.argsort(-y)
        y_sorted = y[sorted_indices]

        y_thresholded = np.zeros_like(y_sorted)

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
    def line_search_delta(self, pi, delta, lamb, varrho, X, y, theta, q, initial_learning_rate, cons=0.5):
        learning_rate = initial_learning_rate
        count = 0

        # 调用注入的梯度函数
        grad = self.grad_of_delta_func(pi, delta, lamb, varrho, X, y, theta)

        while count < 1000:
            count = count + 1


            tmp = np.maximum(delta - learning_rate * grad, 0)
            tmp_delta = self.box_quantile_thresholding(tmp, q)


            lhs = np.sum((tmp_delta - delta) ** 2) / 2


            rhs = learning_rate * (self.function_value_func(pi, tmp_delta, lamb, varrho, X, y, theta) -
                         self.function_value_func(pi, delta, lamb, varrho, X, y, theta) -
                         np.dot(grad, tmp_delta - delta))

            if lhs >= rhs:
                break

            learning_rate *= cons
        return learning_rate

    # --- 核心函数 2: update_delta_optimal_rho (主循环) ---
    def update_delta_box_quantile(self, pi, delta, lamb, varrho, X, y, theta, q, iterations=10000):
        delta_update = delta.copy()

        # 调用注入的学习率函数
        initial_learning_rate = self.learning_rate_delta_func(pi, delta, lamb, varrho, X, y, theta) * 5

        for t in range(iterations):
            tmp_delta = delta_update.copy()


            learning_rate = self.line_search_delta(pi, delta_update.copy(), lamb, varrho, X, y, theta, q, initial_learning_rate=initial_learning_rate)


            tmp = np.maximum(
                delta_update.copy() - learning_rate * self.grad_of_delta_func(pi, delta_update, lamb, varrho,
                                                                    X, y, theta), 0)

            # 2. 调用内嵌的 Box Quantile Thresholding 投影
            delta_update = self.box_quantile_thresholding(tmp, q)

            # 检查收敛条件
            if abs(self.function_value_func(pi, delta_update, lamb, varrho, X, y, theta) -
                   self.function_value_func(pi, tmp_delta, lamb, varrho, X, y, theta)) < 1e-6:
                break

        return delta_update

    # =======================================================
    # III. 辅助函数：创建损失和梯度函数（内部柯里化包装）
    # =======================================================

    def _create_loss_wrapper(self, pi, lamb, varrho, X, y, coef):
        # 封装 self.function_value_func
        return lambda delta: self.function_value_func(pi, delta, lamb, varrho, X, y, coef)

    def _create_grad_wrapper(self, pi, lamb, varrho, X, y, coef):
        # 封装 self.grad_of_delta_func
        return lambda delta: self.grad_of_delta_func(pi, delta, lamb, varrho, X, y, coef)

    # =======================================================
    # IV. 超松弛加速 PGD (Overrelaxation APGD)
    # =======================================================

    def accelerated_delta_overrelaxation(self, pi, delta, lamb, varrho, X, y, coef, q, w=1.5, iterations=10000):


        # 1. 内部柯里化（包装）损失和梯度函数
        loss = self._create_loss_wrapper(pi, lamb, varrho, X, y, coef)
        grad_loss = self._create_grad_wrapper(pi, lamb, varrho, X, y, coef)

        # 2. 初始化变量
        learning_rate = self.learning_rate_delta_func(pi, delta, lamb, varrho, X, y, coef)

        xi = delta.copy()

        for iter in range(iterations):
            delta_tmp = delta.copy()

            # --- 步骤 1: 超松弛梯度步骤 (Overrelaxed Gradient Step) ---
            # 计算当前 delta 处的梯度
            current_grad = grad_loss(delta)

            # 梯度下降一步 (delta - learning_rate * grad)
            grad_step = delta - learning_rate * current_grad

            # 应用超松弛更新 xi = (1 - w) * xi_prev + w * (grad_step)
            xi = (1 - w) * xi + w * grad_step

            # --- 步骤 2: 投影步骤 (Projection) ---
            # delta = Prox_Omega(xi)
            delta = self.box_quantile_thresholding(xi, q)

            # --- 步骤 3: 检查收敛条件 ---
            if np.abs(loss(delta_tmp) - loss(delta)) < 1e-6:
                break

        return delta


