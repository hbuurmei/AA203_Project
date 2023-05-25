import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from environment import WildFireEnv

# Create a DQN Network class
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Create a Replay Buffer class
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    

# Create a DQN Agent class
class DQNAgent(object):
    def __init__(self, env: WildFireEnv, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99, epsilon=0.9, buffer_capacity=10000, batch_size=64):
        self.env = env 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_capacity)

        self.model = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        '''Function to choose the best action given the states. Use the epsilon-greedy policy'''
        # Choose the best action
        if np.random.rand() <= self.epsilon:
            # Choose the best action
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_value = self.model.forward(state)
                action = torch.argmax(q_value).item()
        else:
            # Choose a random action
            action = np.random.choice(self.action_dim)

        return action
    
    def update(self, batch):
        '''Function to update the network parameters using the loss function'''
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get the Q values
        q_values = self.model.forward(states)

        next_q_values = self.target_model.forward(next_states)

        # Get the Q value of the action taken
        q_values = torch.gather(q_values, actions.unsqueeze(1)).squeeze(1)

        # Get the Q value of the next state
        next_q_value = torch.max(next_q_values, 1)[0]

        # Compute the target
        target = rewards + self.gamma * next_q_value * (1 - dones)
        target = target.detach()

        # Compute the loss
        loss = self.loss_fn(q_values, target)

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_target(self):
        '''Function to update the target network'''
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train(self, init_pos, num_episodes=1000, batch_size=32, reward_threshold = -0.5):
        for epi in range(num_episodes):
            state = self.env.reset(init_pos)
            done  = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                self.buffer.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if len(self.buffer) > batch_size:
                    batch = self.buffer.sample(batch_size)
                    loss = self.update(batch)
            
                total_reward += reward
                
                if reward > reward_threshold:
                    done = True

                state = next_state

            self.update_target()

            print("Episode: {}, total reward: {}".format(epi, total_reward))




