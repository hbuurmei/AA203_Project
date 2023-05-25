import numpy as np
from scipy.stats import multivariate_normal

'''
The environment includes the gridworld, the wildfire and the agents.
The state of the agent is defined as an array [x, y, mu, sigma] of size N_agents + N_sats + 1 + 2 by 2,
where x and y are the (N_agents + N_sats) locations, mu and sigma are the mean and covariance of the estimated distribution.
'''

class GridWorld:

    def __init__(self, width, height, max_steps=100):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.step = 0

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def check_inbound(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height
    

class WildFireEnv(GridWorld):

    def __init__(self, width, height, max_temperature, init_state, action_range, p_move, N_agents, N_sats=0):
        super().__init__(width, height)
        self.max_temperature = max_temperature
        self.mean = np.array([width/2, height/2])
        self.cov = np.array([[width/2, 0], [0, height/2]])
        self.temperature = multivariate_normal(self.mean, self.cov)
        self.state = init_state
        self.action_range = action_range
        self.p_move = p_move  # penalty factor for moving
        self.N_agents = N_agents
        self.N_sats = N_sats
    
    def reset(self, init_pos):
        self.state[0] = init_pos
        self.step = 0

    def get_max_temperature(self):
        return self.max_temperature
    
    def get_temperatures(self, state):
        '''Get the temperatures at all locations.'''
        temperatures = np.zeros(self.N_agents + self.N_sats)
        for loc_idx in range(self.N_agents + self.N_sats):
            temperatures[loc_idx] = self.temperature.pdf(state[loc_idx])
        return self.temperature.pdf(state)

    def get_divergence(self, state):
        '''Evaluate the distance from the true distribution given a certain estimated distribution using the KL divergence analytical formula.'''
        k = 2 # dimension
        mu = state[-3]  # estimated mean
        sigma = state[-2:, :]  # estimated covariance
        D_kl = 1/2 * (np.log(np.linalg.det(self.cov)/np.linalg.det(sigma)) - k + (mu - self.mean).T @ np.linalg.inv(self.cov) @ (mu - self.mean) + np.trace(np.linalg.inv(self.cov) @ sigma))
        return D_kl
    
    def fit_distribution(self, new_locations):
        '''Update distribution given a new locations of the agents and satellites.'''
        temperatures = self.get_temperatures(new_locations)
        temperatures_sum = np.sum(temperatures)
        fitted_mu = np.sum(temperatures * new_locations, axis=0) / temperatures_sum
        new_mu = 0.5 * self.mu + 0.5 * fitted_mu  # update mean using a weighted average
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

    def act(self, action):
        '''Take an action in the environment. The action is a a single integer which is a linear index of the new state relative to the current state.'''
        action = np.unravel_index(action, (self.action_range, self.action_range))  # convert linear index to 2D relative state
        action = action - np.array([self.action_range//2, self.action_range//2]) 
        new_locations = self.state[:(self.N_agents + self.N_sats)] + action
        new_mu, new_sigma = self.fit_distribution(new_locations)
        return np.concatenate((new_locations, new_mu, new_sigma), axis=0)
    
    def flatten_state(self, state):
        return state.flatten()
    
    def step(self, action):
        '''Take a step in the environment given an action.'''
        self.step += 1
        if self.step >= self.max_steps:
            done = True
        else:
            done = False
            new_state = self.act(action)
            reward = self.get_reward(new_state)
            self.state = new_state
        return new_state, reward, done
