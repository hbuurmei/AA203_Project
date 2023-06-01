import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from svgpath2mpl import parse_path
import imageio


'''
The environment includes the wildfire with the agents in it.
The state of the agent is defined as an array [x, y, mu, sigma] of size N_agents + N_sats + 1 + 2 by 2,
where x and y are the (N_agents + N_sats) locations, mu and sigma are the mean and covariance of the estimated distribution.
'''
    
class WildFireEnv:

    def __init__(self, width: int, height: int, init_state, action_range: int, p_move: float, max_temp, N_agents: int, N_sats=0, max_steps=100, tol=1e-8):
        self.width = width
        self.height = height
        self.step_count = 0
        self.mean = np.array([width/2, height/2])
        # self.cov = np.array([[width/2, 0], [0, height/2]])
        self.cov = np.array([[4, 2], [2, 4]])
        self.temperature_dist = multivariate_normal(self.mean, self.cov)
        self.init_state = init_state
        self.state = init_state
        self.action_range = action_range
        self.p_move = p_move  # penalty factor for moving
        self.max_temp = max_temp
        self.N_agents = N_agents
        self.N_sats = N_sats
        self.max_steps = max_steps
        self.tol = tol
        self.done = False
        self.pos_history = np.array([[]])
        self.meas_history = np.array([[]])
        self.mu_approx_history = np.array([[]])
        self.cov_approx_history = np.array([[]])
        self.agent_history = np.array([[]])
        
    def reset(self):
        # self.state = self.init_state
        self.state[:self.N_agents] = np.random.randint(0, self.width, size=(self.N_agents, 2))
        self.pos_history = np.array([[]])
        self.meas_history = np.array([[]])
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

    def get_divergence(self, mu, sigma):
        '''Evaluate the distance from the true distribution given a certain estimated distribution using the KL divergence analytical formula.'''
        k = 2 # dimension
        D_kl = 1/2 * (np.log(np.linalg.det(self.cov)/np.linalg.det(sigma)) - k + (mu - self.mean).T @ np.linalg.inv(self.cov) @ (mu - self.mean) + np.trace(np.linalg.inv(self.cov) @ sigma))
        return D_kl
    
    def fit_distribution(self):
        '''Update distribution given new locations of the agents and satellites.'''
        temps = self.meas_history
        pos = self.pos_history
        new_mu = np.sum(temps.reshape(-1,1) * pos, axis=0) / np.sum(temps)
        new_sigma = np.zeros((2, 2))
        for i in range(temps.size):
            new_sigma += ((temps[i] * np.outer(np.array([pos[i,0],pos[i,1]]) - new_mu, np.array([pos[i,0],pos[i,1]]) - new_mu)) / np.sum(temps))
        if np.linalg.det(new_sigma) < 1e-8:
            new_sigma = np.eye(2)
        return new_mu, new_sigma

    def move_cost(self, new_state):
        '''Calculate the (approximate) cost of moving from current state to new state, simply being the euclidian distance between the two states'''
        move_cost = 0
        for loc_idx in range(self.N_agents + self.N_sats):
            move_cost += np.linalg.norm(new_state[loc_idx] - self.state[loc_idx])
        return move_cost

    def get_reward(self, new_state):
        '''Get the reward given a certain estimated distribution and the new state.'''
        new_mu = new_state[-3]  # estimated mean
        new_sigma = new_state[-2:, :]  # estimated covariance
        return -self.get_divergence(new_mu, new_sigma) - self.p_move * self.move_cost(new_state)

    def update_history(self, locations):
        # append current estimates to history arrays for plotting later
        self.mu_approx_history = np.append(self.mu_approx_history, [self.state[-3]]).reshape(-1, 2)
        self.cov_approx_history = np.append(self.cov_approx_history, self.state[-2:]).reshape(-1, 2, 2)
        self.agent_history = np.append(self.agent_history, self.state[:self.N_agents]).reshape(-1, self.N_agents, 2)
        self.pos_history = np.append(self.pos_history, locations).reshape(-1, 2)
        temperatures = self.get_temperatures(locations)
        self.meas_history = np.append(self.meas_history, [temperatures])

    def act(self, action: int):
        '''Take an action in the environment. The action is a a single integer which is a linear index of the new state relative to the current state.'''
        # action = np.unravel_index(action, (self.action_range, self.action_range))  # convert linear index to 2D relative state
        # action = action - np.array([self.action_range//2, self.action_range//2]) 
        # new_locations = self.state[:self.N_agents] + action  # we can only move the agents
        action_agents = np.unravel_index(action, (self.action_range**2,) * self.N_agents)
        action = np.zeros((self.N_agents, 2))
        for agent_idx in range(self.N_agents):
            action[agent_idx, :] = np.unravel_index(action_agents[agent_idx], (self.action_range,) * 2) - np.array([self.action_range//2, self.action_range//2])  # subtract half the action range to get relative states
        new_locations = self.state[:self.N_agents] + action  # add relative states to current locations
        # Check if new locations are inbound, otherwise keep the old location
        for loc_idx in range(self.N_agents):
            if not self.check_single_inbound(new_locations[loc_idx]):
                new_locations[loc_idx] = self.state[loc_idx]
        # Append new_locations to history
        self.update_history(new_locations)
        new_mu, new_sigma = self.fit_distribution()
        # Set done to true if divergence between new and old distribution is small enough
        if self.get_divergence(new_mu, new_sigma) < self.tol:
            self.done = True
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
        window = 5 # window buffer size for plotting
        mu_pred = self.state[-3]
        sigma_pred = self.state[-2:]
        pred_dist = multivariate_normal(mu_pred, sigma_pred).pdf
        true_dist = self.temperature_dist.pdf

        x_pred, y_pred, z_pred = self.plotVal(pred_dist)
        x_true, y_true, z_true = self.plotVal(true_dist)

        fig,ax = plt.subplots(layout='constrained')
        pred_contour = ax.contourf(x_pred, y_pred, z_pred, 30, cmap='Blues')
        true_contour = ax.contourf(x_true, y_true, z_true, 30, cmap='Reds', alpha = 0.3)
        ax.set_title('Predicted Distribution Relative to True Distribution')
        ax.set_xlabel(r'$x_1$ [km]')
        ax.set_ylabel(r'$x_2$ [km]')
        ax.set_xlim([-window,self.width + window])
        ax.set_ylim([-window,self.height + window])
        cbar1 = fig.colorbar(pred_contour)
        cbar2 = fig.colorbar(true_contour)
        cbar1.set_label('Predicted Temps')
        cbar2.set_label('True Temps')

        drone = parse_path("M915 5009 c-218 -18 -426 -125 -567 -293 -65 -78 -147 -235 -173 -331 -26 -95 -31 -273 -10 -380 32 -171 116 -327 241 -446 123 -116 243 -182 401 -220 73 -17 111 -20 220 -17 73 2 137 8 143 14 6 6 -19 38 -78 96 l-87 86 -80 4 c-250 12 -506 240 -570 506 -35 144 -6 320 77 460 42 72 158 188 230 230 194 114 435 123 636 22 189 -95 322 -282 351 -494 6 -44 8 -101 5 -129 l-7 -49 79 -79 c43 -43 81 -79 85 -79 16 0 34 136 34 260 -1 190 -39 318 -141 471 -171 258 -463 394 -789 368z M4030 5014 c-14 -2 -52 -9 -85 -15 -205 -37 -414 -179 -534 -364 -106 -161 -147 -313 -138 -509 5 -117 21 -216 35 -216 4 0 43 33 87 74 l80 73 1 99 c0 122 15 185 65 292 28 57 60 104 105 152 81 86 152 133 257 172 105 40 262 49 372 24 212 -50 388 -211 462 -423 24 -69 28 -93 28 -203 0 -105 -4 -136 -23 -195 -86 -258 -331 -455 -569 -459 l-73 -1 -85 -85 c-55 -55 -81 -88 -75 -94 15 -15 239 -20 315 -7 328 57 599 309 686 636 30 112 30 298 0 410 -98 370 -411 623 -790 639 -53 3 -107 3 -121 0z M984 4241 c-89 -40 -137 -144 -114 -243 12 -51 22 -62 284 -323 347 -346 439 -461 509 -636 57 -141 62 -182 62 -469 -1 -243 -3 -272 -23 -344 -42 -150 -141 -334 -275 -511 -81 -106 -307 -336 -430 -438 -106 -87 -124 -112 -133 -179 -17 -123 83 -238 206 -238 81 0 90 7 475 389 403 400 475 459 635 520 152 59 191 66 365 66 159 0 161 -1 270 -38 173 -60 339 -157 533 -313 161 -129 381 -353 536 -548 85 -106 219 -115 312 -22 59 58 76 127 53 211 -9 33 -56 85 -258 290 -136 138 -275 282 -308 322 -132 154 -212 298 -255 454 -22 79 -23 102 -23 369 0 331 8 378 95 555 99 200 282 414 586 681 158 139 184 174 184 247 0 118 -96 217 -210 217 -81 0 -90 -8 -460 -374 -371 -368 -474 -456 -615 -526 -138 -68 -217 -85 -390 -85 -126 0 -163 4 -230 23 -100 28 -294 123 -405 197 -200 134 -451 365 -656 601 -122 142 -155 164 -237 164 -23 0 -60 -9 -83 -19z m1681 -1276 c167 -44 278 -167 308 -341 30 -173 -63 -359 -218 -435 -262 -129 -559 19 -608 305 -50 296 225 547 518 471z M2509 2811 c-118 -24 -202 -128 -202 -251 0 -191 188 -309 364 -228 90 41 142 124 142 228 0 162 -145 282 -304 251z M885 1799 c-343 -50 -624 -306 -707 -647 -29 -119 -29 -294 1 -407 81 -306 310 -534 616 -616 96 -25 287 -31 380 -10 311 68 554 296 646 607 30 102 37 290 14 400 -10 46 -20 84 -24 84 -3 0 -43 -34 -88 -75 l-82 -76 5 -77 c20 -274 -169 -543 -453 -643 -82 -29 -255 -37 -346 -15 -330 78 -546 406 -488 741 49 280 330 535 590 535 l66 0 89 89 c48 49 84 91 78 95 -15 9 -142 26 -192 25 -25 -1 -72 -5 -105 -10z M3985 1794 c-11 -3 -26 -7 -34 -10 -10 -3 14 -34 74 -94 l90 -90 65 0 c197 0 430 -160 530 -364 59 -119 74 -190 67 -316 -9 -178 -76 -322 -208 -443 -119 -111 -263 -167 -429 -167 -181 0 -318 58 -451 190 -141 139 -200 272 -202 457 l-2 103 -75 75 c-41 41 -78 75 -81 75 -4 0 -15 -33 -25 -72 -28 -109 -26 -307 4 -411 84 -294 313 -519 607 -598 113 -30 307 -30 420 0 292 78 517 292 604 574 119 385 -65 814 -429 1001 -139 72 -204 88 -365 92 -77 1 -149 1 -160 -2z")
        drone.vertices -= drone.vertices.mean(axis=0)
        ax.scatter(self.state[:self.N_agents,0], self.state[:self.N_agents,1], c='black', marker=drone, s = 250, label='Agent Locations')
        # plot location history of each agent as a line with markers
        for i in range(self.N_agents):
            ax.plot(self.agent_history[:,i,0], self.agent_history[:,i,1], marker='o', markersize=3, label='Agent {} Trajectory'.format(i))
        ax.legend()

        fig.savefig('./renderings/step_{}.png'.format(self.step_count))
        plt.close(fig)



    def plotVal(self, distrib):
        k = 0.05 # adjusts coarseness of the plot
        window = 5 # window buffer size for plotting
        x, y = np.meshgrid(np.arange(-window,self.width + window, k), np.arange(-window,self.height + window, k))
        xy = np.vstack((x.flatten(), y.flatten())).T
        z = distrib(xy).reshape(x.shape)

        return x,y,z
        

