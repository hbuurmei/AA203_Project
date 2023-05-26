import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

'''
The environment includes the wildfire with the agents in it.
The state of the agent is defined as an array [x, y, mu, sigma] of size N_agents + N_sats + 1 + 2 by 2,
where x and y are the (N_agents + N_sats) locations, mu and sigma are the mean and covariance of the estimated distribution.
'''
    
class WildFireEnv:

    def __init__(self, width: int, height: int, init_state, action_range: int, p_move: float, max_temp, N_agents: int, N_sats=0, max_steps=100):
        self.width = width
        self.height = height
        self.step_count = 0
        self.mean = np.array([width/2, height/2])
        self.cov = np.array([[width/2, 0], [0, height/2]])
        self.temperature_dist = multivariate_normal(self.mean, self.cov)
        self.init_state = init_state
        self.state = init_state
        self.action_range = action_range
        self.p_move = p_move  # penalty factor for moving
        self.max_temp = max_temp
        self.N_agents = N_agents
        self.N_sats = N_sats
        self.max_steps = max_steps
        self.done = False
    
    def reset(self):
        self.state = self.init_state
        self.step_count = 0
        self.done = False

    def reposition(self, locations):
        '''Reposition the agents in the environment.'''
        self.state[:self.N_agents] = locations
        updated_mu, updated_sigma = self.fit_distribution(locations)
        self.state[-3] = updated_mu
        self.state[-2:] = updated_sigma

    def print_state(self):
        '''Print the current state of the environment.'''
        print('-'*40)
        print('Agent locations:     ', self.state[:self.N_agents])
        print('Satellite locations: ', self.state[self.N_agents:(self.N_agents + self.N_sats)])
        print('Mean:                ', self.state[-3])
        print('Covariance:          ', self.state[-2:].flatten())
        print('-'*40)
    
    def check_single_inbound(self, location):
        '''Check if a single location is within the bounds of the environment.'''
        return np.all(location >= 0) and np.all(location < np.array([self.width, self.height]))

    def check_all_inbound(self, state):
        '''Check if the state is within the bounds of the environment.'''
        return np.all(state[:self.N_agents + self.N_sats] >= 0) and np.all(state[:self.N_agents + self.N_sats] < np.array([self.width, self.height]))
    
    def get_temperatures(self, state):
        '''Get the temperatures at all locations.'''
        temperatures = np.array([self.temperature_dist.pdf(state[loc_idx]) for loc_idx in range(self.N_agents + self.N_sats)])
        temperatures = temperatures * self.max_temp / self.temperature_dist.pdf(self.mean)
        return temperatures

    def get_divergence(self, state):
        '''Evaluate the distance from the true distribution given a certain estimated distribution using the KL divergence analytical formula.'''
        k = 2 # dimension
        mu = state[-3]  # estimated mean
        sigma = state[-2:, :]  # estimated covariance
        D_kl = 1/2 * (np.log(np.linalg.det(self.cov)/np.linalg.det(sigma)) - k + (mu - self.mean).T @ np.linalg.inv(self.cov) @ (mu - self.mean) + np.trace(np.linalg.inv(self.cov) @ sigma))
        return D_kl
    
    def fit_distribution(self, new_locations):
        '''Update distribution given new locations of the agents and satellites.'''
        temperatures = self.get_temperatures(new_locations)
        temperatures_sum = np.sum(temperatures)
        fitted_mu = np.sum(temperatures * new_locations, axis=0) / temperatures_sum
        new_mu = (self.state[-3] + fitted_mu) / 2  # update mean using a weighted average
        new_sigma = self.state[-2:, :]  # do not update covariance for now    
        return new_mu, new_sigma

    def move_cost(self, new_state):
        '''Calculate the (approximate) cost of moving from current state to new state, simply being the euclidian distance between the two states'''
        move_cost = 0
        for loc_idx in range(self.N_agents + self.N_sats):
            move_cost += np.linalg.norm(new_state[loc_idx] - self.state[loc_idx])
        return move_cost

    def get_reward(self, new_state):
        '''Get the reward given a certain estimated distribution and the new state.'''
        return -self.get_divergence(new_state) - self.p_move * self.move_cost(new_state)

    def act(self, action: int):
        '''Take an action in the environment. The action is a a single integer which is a linear index of the new state relative to the current state.'''
        action = np.unravel_index(action, (self.action_range, self.action_range))  # convert linear index to 2D relative state
        action = action - np.array([self.action_range//2, self.action_range//2]) 
        new_locations = self.state[:self.N_agents] + action  # we can only move the agents
        # Check if new locations are inbound, otherwise keep the old location
        for loc_idx in range(self.N_agents):
            if not self.check_single_inbound(new_locations[loc_idx]):
                new_locations[loc_idx] = self.state[loc_idx]
        new_mu, new_sigma = self.fit_distribution(new_locations)
        return np.vstack((new_locations, new_mu, new_sigma))
    
    def flatten_state(self, state):
        return state.flatten()
    
    def step(self, action: int):
        '''Take a step in the environment given an action.'''
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        else:
            self.done = False
        new_state = self.act(action)
        reward = self.get_reward(new_state)
        self.state = new_state
        return self.flatten_state(new_state), reward, self.done
    
    def render(self):
        mu_pred = self.state[-3]
        sigma_pred = self.state[-2:]
        pred_dist = multivariate_normal(mu_pred, sigma_pred).pdf
        true_dist = self.temperature_dist.pdf

        x_pred, y_pred, z_pred = self.plotVal(pred_dist)
        x_true, y_true, z_true = self.plotVal(true_dist)

        fig,ax = plt.subplots(layout='constrained')
        pred_contour = ax.contourf(x_pred, y_pred, z_pred, 50, cmap='Blues')
        true_contour = ax.contourf(x_true, y_true, z_true, 50, cmap='Reds', alpha = 0.3)
        ax.set_title('Predicted Distribution Relative to True Distribution')
        ax.set_xlabel(r'$x_1$ [km]')
        ax.set_ylabel(r'$x_2$ [km]')
        ax.set_xlim([-self.width,self.width])
        ax.set_ylim([-self.height,self.height])
        cbar1 = fig.colorbar(pred_contour)
        cbar2 = fig.colorbar(true_contour)
        cbar1.set_label('Predicted Temps')
        cbar2.set_label('True Temps')

        plt.show()


    def plotVal(self, distrib):
        k = 0.1 # adjusts coarseness of the plot
        x, y = np.meshgrid(np.arange(-self.width,self.width, k), np.arange(-self.height,self.height, k))
        xy = np.vstack((x.flatten(), y.flatten())).T
        z = distrib(xy).reshape(x.shape)

        return x,y,z
        

