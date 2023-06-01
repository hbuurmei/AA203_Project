import torch
import numpy as np
from env_weighted import WildFireEnv
from dqn import DQNAgent
import imageio
import utils as ut

# Create an environment
width, height = 10, 10
N_agents, N_sats = 1, 0
action_range = 5
p_move = 0  # weight of penalty of moving w.r.t. penalty of incorrect distribution estimate
max_temp = 400  # maximum temperature of the fire, i.e. at the mean
init_positions = np.column_stack((np.zeros(N_agents + N_sats), np.arange(N_agents + N_sats)))
init_mu = np.zeros((1, 2))
init_sigma = np.array([[5, 0], [0, 5]])
init_state = np.vstack((init_positions, init_mu, init_sigma))

env = WildFireEnv(width = width, height = height, init_state = init_state, action_range = action_range, max_temp = max_temp, N_agents = N_agents, N_sats = N_sats, p_move = 0.1)
env.step(0)

# Training DQN
state_dim = env.flatten_state(init_state).shape[0]
action_dim = env.action_range**2
agent = DQNAgent(env, state_dim, action_dim)

TRAIN = True
if TRAIN:
    reward_hist = agent.train(num_episodes=3000)

    # Save the model
    torch.save(agent.model.state_dict(), 'models/dqn_single_weight.pt')

# Test DQN
agent.model.load_state_dict(torch.load('models/dqn_single_weight.pt'))
env.reset()
# env.reposition(np.array([[1, 9]]))


while not env.done:
    print("Simulation step: ", env.step_count)
    action = agent.act(env.flatten_state(env.state))
    env.print_state()
    env.step(action)
    env.render()

fig, ax = ut.plotKL(env)
fig.savefig('./renderings/KL_single_weight.png')

frames = []
for t in range(1,env.step_count+1):
    image = imageio.v2.imread(f'./renderings/step_{t}.png')
    frames.append(image)
imageio.mimsave('./renderings/distributions.gif', # output gif
            frames,          # array of input frames
            duration = 500,         # optional: frames per second
            loop = 1)        # optional: loop enabled - 1 for True; 0 for False
