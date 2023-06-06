import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv

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
        self.reward_history = np.array([])
        self.double_q = True
 
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
    
    def act(self, state):
        '''Function to choose the best action given the states greedily with known q-values.'''
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.model.forward(state)
            action = torch.argmax(q_value).item()
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
        q_values = torch.gather(q_values, 1,  actions.unsqueeze(1)).squeeze(1)

        # Get the Q value of the next state
        # next_q_value = torch.max(next_q_values, 1)[0]
        if self.double_q:
            next_actions = self.model.forward(next_states).max(1)[1]
            next_q_value = torch.gather(next_q_values, 1, next_actions.unsqueeze(1)).squeeze(1)

        else:
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
    
    def trainingPlot(self):
        fig,ax = plt.subplots(layout='constrained')    
        ax.plot(self.reward_history)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward History')
        return fig,ax

    def train(self, num_episodes=1000, batch_size=32, reward_threshold = -0.1):
        target_update_frequency = 10
        for epi in range(num_episodes):
            
            self.env.reset()
            state = self.env.flatten_state(self.env.state)
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

            if epi % target_update_frequency == 0:
                self.update_target()

            if epi % 20 == 0:
                print("Episode: {}, total reward: {}".format(epi, total_reward))

            self.reward_history = np.append(self.reward_history, total_reward)

        print("Training completed.")
        return self.trainingPlot()

    def log_training(self, save_path):
        with open(save_path, 'w') as file:
            file.write('episode, reward\n')
            for epi, reward in enumerate(self.reward_history):
                file.write(f'{epi}, {reward}\n')
        file.close()


