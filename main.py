from TORCH import TORCH, PiOptimizer, TORCH_regression, TORCH_location
import numpy as np


def projection_Omega(mu):
    # 创建一个和 beta 同形状的零数组
    projected_mu = mu.copy()
    # 将最后一个维度的值复制到 projected_beta 的相应位置
    projected_mu[0] = 1
    #projected_mu[:len(projected_mu) // 2] = -1
    return projected_mu


for n_features in range(100, 101, 5):
    iterations = 10000  # 迭代次数
    num_experiments = 20 # 重复实验次数

    # 用于记录不同样本量下的平均时间
    median_times_per_sample_size = []

    print(f"Running for n_features={n_features}")
    optimal_means = []
    # 样本量范围从 20 到 200，步长为 10
    for n_samples in np.array([150]):
        print(f"Running for n_samples={n_samples}")

        for exp in range(num_experiments):
            print(f"Experiment {exp + 1}/{num_experiments} for n_features={n_features} n_samples={n_samples}")
            np.random.seed(exp)  # 固定随机种子

            # 数据生成
            true_mu = np.zeros(n_features)  # 真实均值
            X = np.random.multivariate_normal(mean=true_mu, cov=np.eye(n_features) * 4, size=n_samples)
            q = int(np.floor(n_samples * 0.15))





            varrho = 1

            pi, delta, theta = TORCH_location(X, q,varrho, projection_Omega)

            print('stat',  2 * np.sum(-np.log(n_samples * pi)))