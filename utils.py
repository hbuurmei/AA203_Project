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
        KL[t] += env.get_divergence(env.mu_approx_history[t,:], env.cov_approx_history[t,:])
    fig,ax = plt.subplots(layout='constrained')
    ax.plot(np.arange(1, env.step_count), KL[1:])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('KL Divergence Magnitude')
    ax.set_title('KL Divergence Between True and Estimated Distribution Over Time')
    return fig, ax