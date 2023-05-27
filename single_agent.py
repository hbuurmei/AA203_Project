import torch
import numpy as np
from environment import WildFireEnv
from dqn import DQNAgent

# Create an environment
width, height = 10, 10
N_agents, N_sats = 1, 0
action_range = 5
p_move = 1  # weight of penalty of moving w.r.t. penalty of incorrect distribution estimate
max_temp = 400  # maximum temperature of the fire, i.e. at the mean
init_positions = np.column_stack((np.zeros(N_agents + N_sats), np.arange(N_agents + N_sats)))
init_mu = np.zeros((1, 2))
init_sigma = np.array([[width/2, 0], [0, height/2]])
init_state = np.vstack((init_positions, init_mu, init_sigma))

env = WildFireEnv(width, height, init_state, action_range, p_move, max_temp, N_agents, N_sats)

# Training DQN
state_dim = env.flatten_state(init_state).shape[0]
action_dim = env.action_range**2
agent = DQNAgent(env, state_dim, action_dim)

TRAIN = False
if TRAIN:
    agent.train(num_episodes=1000)

    # Save the model
    torch.save(agent.model.state_dict(), 'models/dqn_single.pt')

# Test DQN
agent.model.load_state_dict(torch.load('models/dqn_single.pt'))
env.reset()
env.reposition(np.array([[1, 9]]))

while not env.done:
    print("Simulation step: ", env.step_count)
    action = agent.act(env.flatten_state(env.state))
    env.print_state()
    env.step(action)
    # env.render()
