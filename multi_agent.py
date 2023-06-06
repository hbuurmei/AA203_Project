import torch
import argparse
import numpy as np
# from environment import WildFireEnv
from env_weighted_old import WildFireEnv
from dqn import DQNAgent
import imageio
import utils as ut

# Take in and parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_agents', type=int, default=1, help='Number of agents')
parser.add_argument('--num_sats', type=int, default=0, help='Number of satellites')
parser.add_argument('--action_range', type=int, default=5, help='Action range')
parser.add_argument('--p_move', type=float, default=0., help='Movement cost')
parser.add_argument('--env_size', type=int, default=10, help='Size of the environment')
parser.add_argument('--rand_reset', type=bool, default=False, help='Randomly reset agent positions during training')
parser.add_argument('--episode_count', type=int, default=5000, help='Number of episodes to train for')

args = parser.parse_args()

# Print to console
print('---------------Running with the following parameters:---------------')
print('Number of agents: ', args.num_agents)
print('Number of satellites: ', args.num_sats)
print('Action range: ', args.action_range)
print('Movement cost: ', args.p_move)
print('Environment size: ', args.env_size)
print('Randomly reset agent positions during training: ', args.rand_reset)
print('Number of episodes to train for: ', args.episode_count)
print('-----------------------------------------------------------------------')

# Create an environment
width, height = args.env_size, args.env_size
N_agents, N_sats = args.num_agents, args.num_sats
action_range = args.action_range
p_move = args.p_move  # weight of penalty of moving w.r.t. penalty of incorrect distribution estimate
max_temp = 400  # maximum temperature of the fire, i.e. at the mean
init_positions = np.column_stack((np.zeros(N_agents + N_sats), np.arange(N_agents + N_sats)))
init_mu = np.ones((1, 2))
init_sigma = np.array([[width/2, 0], [0, height/2]])
init_state = np.vstack((init_positions, init_mu, init_sigma))

env = WildFireEnv(width = width, height = height, init_state = init_state, 
                  action_range = action_range, p_move = p_move, max_temp = max_temp, 
                  N_agents = N_agents, N_sats = N_sats, rand_reset = args.rand_reset)

# Training DQN
state_dim = env.flatten_state(init_state).shape[0]
action_dim = (env.action_range**2)**N_agents
agent = DQNAgent(env, state_dim, action_dim)

TRAIN = False
if TRAIN:
    agent.train(num_episodes=args.episode_count)
    # fig.savefig('./renderings/training_single_weight_rewards_plt_30x30.png')
    # fig.savefig('./renderings/training_test_2agents_0p.png')
    # Save the model
    torch.save(agent.model.state_dict(), 'models/dqn_training_test_1agents_0p_1000.pt')

# Test DQN
agent.model.load_state_dict(torch.load('models/dqn_training_test_1agents_0p_1000.pt'))
env.reset()
# env.reposition(np.array([[1, 9], [2, 8], [3, 7]]))
# env.reposition(np.array([[10, 3], [15, 8]]))
# env.reposition(np.array([[8, 3]]))

while not env.done:
    print("Simulation step: ", env.step_count)
    action = agent.act(env.flatten_state(env.state))
    env.print_state()
    env.step(action)
    env.render()

# fig, ax = ut.plotKL(env)
# fig.savefig('./renderings/KL_1agents_0p.png')

# Render final state
GIF = False
if GIF:
    frames = []
    for t in range(1,env.step_count+1):
        image = imageio.v2.imread(f'./renderings/step_{t}.png')
        frames.append(image)
    imageio.mimsave('./renderings/distributions_test_1agents_0p_5000.gif', # output gif
                frames,          # array of input frames
                duration = 500,         # optional: frames per second
                loop = 1)        # optional: loop enabled - 1 for True; 0 for False
