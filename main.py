from TORCH import TORCH, PiOptimizer, TORCH_regression, TORCH_location
import numpy as np
import time

# def projection_Omega(mu):
#     # 创建一个和 beta 同形状的零数组
#     projected_mu = mu.copy()
#     # 将最后一个维度的值复制到 projected_beta 的相应位置
#     projected_mu[0] = 1
#     #projected_mu[:len(projected_mu) // 2] = -1
#     return projected_mu
#
#
# for n_features in range(100, 101, 5):
#     iterations = 10000  # 迭代次数
#     num_experiments = 20 # 重复实验次数
#
#     # 用于记录不同样本量下的平均时间
#     median_times_per_sample_size = []
#
#     print(f"Running for n_features={n_features}")
#     optimal_means = []
#     # 样本量范围从 20 到 200，步长为 10
#     for n_samples in np.array([150]):
#         print(f"Running for n_samples={n_samples}")
#
#         for exp in range(num_experiments):
#             print(f"Experiment {exp + 1}/{num_experiments} for n_features={n_features} n_samples={n_samples}")
#             np.random.seed(exp)  # 固定随机种子
#
#             # 数据生成
#             true_mu = np.zeros(n_features)  # 真实均值
#             X = np.random.multivariate_normal(mean=true_mu, cov=np.eye(n_features) * 4, size=n_samples)
#             q = int(np.floor(n_samples * 0.15))
#
#             X[-q:, :] = 50
#
#
#             varrho = 1
#
#             pi, delta, theta = TORCH_location(X, q, varrho, projection_Omega, pi_solver='AED')
#
#             print('stat',  2 * np.sum(-np.log(n_samples * pi)))


def projection_Omega(beta):
    # 创建一个和 beta 同形状的零数组
    projected_beta = beta.copy()
    # 将最后一个维度的值复制到 projected_beta 的相应位置
    projected_beta[0] = 9
    #projected_beta[:-10] = 10
    #min_index = np.argmin(projected_beta)
    #projected_beta[min_index] = 0
    return projected_beta

for n_features in range(100, 101, 5):
    iterations = 10000  # 迭代次数
    num_experiments = 20  # 重复实验次数

    # 用于记录不同样本量下的平均时间
    median_times_per_sample_size = []

    print(f"Running for n_features={n_features}")



    for n_samples in range(250, 251, 10):
        print(f"Running for n_samples={n_samples}")


        # 用于记录时间和结果
        times = []
        times_exclude_pi = []
        times_delta = []
        times_pi = []

        pi_results = []
        delta_results = []
        beta_results = []
        log_likelihoods = []
        iteration_betas = []
        iteration_deltas = []

        for exp in range(num_experiments):
            print(f"Experiment {exp + 1}/{num_experiments} for n_features={n_features} n_samples={n_samples}")
            # 初始化随机种子
            np.random.seed(exp)

            # 初始化数据
            tau = 0.2
            Sigma = np.zeros((n_features, n_features))
            for i in range(n_features):
                for j in range(n_features):
                    Sigma[i, j] = tau ** abs(i - j)
            X = np.random.multivariate_normal(np.zeros(n_features), Sigma, n_samples)
            beta = np.ones(n_features) * 10
            noise = np.random.normal(0, 4, n_samples)
            y = np.dot(X, beta) + noise
            q = int(np.floor(n_samples * 0.15))
            q = 0
            #y[-q:] = 1000

            varrho = 30
            start_time = time.time()
            pi, delta, theta = TORCH_regression(X, y, q, varrho, projection_Omega, pi_solver='AED', theta_solver='APGD')
            end_time = time.time()
            print('stat',  2 * np.sum(-np.log(n_samples * pi)))
            print('time', end_time - start_time)