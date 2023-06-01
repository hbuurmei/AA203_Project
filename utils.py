import numpy as np
import matplotlib.pyplot as plt
from env_weighted import WildFireEnv
from scipy.stats import multivariate_normal
from tqdm import tqdm

# Plot the KL divergence between the true distribution and the estimated distribution over time
def plotKL(env: WildFireEnv):
    envDistrib = multivariate_normal(env.mean, env.cov)
    KL = np.zeros(env.step_count)
    for t in tqdm(range(1,env.step_count)):
        for x1 in np.arange(0, env.width, 0.1):
            for x2 in np.arange(0, env.height, 0.1):
                est_prob = multivariate_normal(env.mu_approx_history[t], env.cov_approx_history[t,:,:]).pdf([x1, x2])
                true_prob = envDistrib.pdf([x1, x2])
                KL[t] += est_prob * np.log(est_prob / true_prob)
    fig,ax = plt.subplots(layout='constrained')
    ax.plot(np.arange(1, env.step_count), KL[1:])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('KL Divergence Magnitude')
    
    return fig, ax