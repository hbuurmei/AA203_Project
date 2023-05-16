import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# A simple heuristic to generate a trajectory
# Specifically, moving towards the mean of the true underlying Gaussian distribution

# Problem data
grid_size = 10
mu_true = np.array([5, 5])
Sigma_true = np.array([[4, -1], [-1, 4]])
N = 1  # number of drones
M = 1  # number of satellites
T = 10  # number of time steps
P_true = multivariate_normal(mu_true, Sigma_true).pdf

loc = np.ones((T, N + M, 2))
meas = np.zeros((T, N + M))
# x is formatted as first N rows are drone locations, last M rows are satellite locations
mu = np.ones((T, 2))
Sigma = np.zeros((T, 2, 2))
w = 0

# Simulation loop
for t in range(1, T):
    # Generate drone locations and measurements
    for i in range(N):
        dir = mu[t-1, :] - loc[t-1, i, :]
        if abs(dir[1]) > abs(dir[0]):
            step = np.array([0, np.sign(dir[1])])
        else:
            step = np.array([np.sign(dir[0]), 0])
        loc[t, i, :] = loc[t-1, i, :] + step
        meas[t, i] = P_true(loc[t, i, :])
    # Generate satellite locations and measurements
    for j in range(M):
        loc[t, N + j, :] = np.random.uniform(0, grid_size, 2)
        meas[t, N + j] = P_true(loc[t, N + j, :])
    new_w = np.sum(meas[t, :])
    mu[t, :] = (w * mu[t-1, :] + np.sum(np.multiply(meas[t, :, np.newaxis], loc[t, :, :]), axis=0)) / (w + new_w)
    Sigma[t, :, :] = (w * Sigma[t-1, :, :] + np.sum([meas[t, i] * np.outer(loc[t, 0, :] - mu[t, :], loc[t, 0, :] - mu[t, :]) for k in range(N + M)], axis=0)) / (w + new_w)
    if not np.all(np.linalg.eigvals(Sigma[t, :, :]) > 0):
        Sigma[t, :, :] = np.eye(2)
        print(f'Sigma{t} is not symmetric positive definite')
    w += new_w

# Plot the true distribution
x1 = np.arange(0, grid_size, 0.1)
x2 = np.arange(0, grid_size, 0.1)
x1, x2 = np.meshgrid(x1, x2)
z = np.zeros_like(x1)
for i in range(len(x1)):
    for j in range(len(x2)):
        z[i, j] = P_true([x1[i, j], x2[i, j]]) * 10000

fig, ax = plt.subplots(layout='constrained')
contour_plot = ax.contourf(x1, x2, z, 20, cmap='hot_r')
# plt.colorbar()
ax.set_xlabel(r'$x_1 [km]$')
ax.set_ylabel(r'$x_2$ [km]')
cbar = fig.colorbar(contour_plot)
cbar.ax.set_ylabel('Temperature [K]')
# Plot the trajectories of the drones
for t in range(T):
    for i in range(N):
        if t == 0:
            plt.scatter(loc[t, i, 0], loc[t, i, 1], c='b', marker='o')
        else:
            plt.scatter(loc[t, i, 0], loc[t, i, 1], c='b', marker='x')
            plt.annotate('t=' + str(t + 1), (loc[t, i, 0] + 0.2, loc[t, i, 1] - 0.2), color='b')
plt.show()

# Plot the error between the true mean and the estimated mean over time
error = np.zeros(T)
for t in range(T):
    error[t] = np.linalg.norm(mu[t, :] - mu_true)

fig, ax = plt.subplots(layout='constrained')
ax.plot(np.arange(1, T + 1), error)
ax.set_xlabel('Time [hr]')
ax.set_ylabel(r'$\| \hat{\mu} - \bar{\mu} \|~[km]$')
plt.show()

# Plot the KL divergence between the true distribution and the estimated distribution over time
# KL = np.zeros(T)
# for t in range(1, T):
#     for x1 in np.arange(0, grid_size, 0.1):
#         for x2 in np.arange(0, grid_size, 0.1):
#             p_est = multivariate_normal(mu[t, :], Sigma[t, :, :]).pdf([x1, x2])
#             p_true = P_true([x1, x2])
#             KL[t] += p_est * np.log(p_est / p_true)
# plt.plot(np.arange(1, T), KL[1:])
# plt.show()
