# TABLED BECAUSE NON-SENSISICAL METHOD

# import numpy as np
# from scipy.stats import multivariate_normal
# import matplotlib.pyplot as plt

# # Problem Data
# grid_size = 10
# mu_true = np.array([5, 5])
# Sigma_true = np.eye(2)
# N = 1  # number of drones
# M = 1  # number of satellites
# T = 10  # number of time steps
# P_true = multivariate_normal(mu_true, Sigma_true).pdf

# loc = np.ones((T, N + M, 2))
# meas = np.zeros((T, N + M))
# # x is formatted as first N rows are drone locations, last M rows are satellite locations
# mu = np.ones((T, 2))
# Sigma = Sigma_true
# w = 0

# # function for generating sigma points for the UKF (lambda and n are tuning parameter)
# def SigPoints(mu, sig, lam, n):
#     sigmas = np.array([mu])
#     for i in range(sig.shape[0]):
#         cho_sqrt = np.linalg.cholesky((lam+n)@sig)
#         sigmas = np.append(sigmas, [mu + cho_sqrt[i]])
#         sigmas = np.append(sigmas, [mu - cho_sqrt[i]])
#     return sigmas

# # function for determining which sigma point is closest to an agent's location
