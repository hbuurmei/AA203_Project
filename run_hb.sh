#!/bin/bash

# Script for running experiments on multi_agent.py

# Grid size and action ranges
# python multi_agent.py --env_size 8
# python multi_agent.py --action_range 3
# python multi_agent.py --env_size 12
# python multi_agent.py --action_range 5
# python multi_agent.py --env_size 16
# python multi_agent.py --action_range 7
# python multi_agent.py --env_size 20
# python multi_agent.py --action_range 9
# python multi_agent.py --env_size 24
# python multi_agent.py --env_size 28
# python multi_agent.py --env_size 32

# Movement costs
# python multi_agent.py --p_move 0.
# python multi_agent.py --p_move 0.01
# python multi_agent.py --p_move 0.025
# python multi_agent.py --p_move 0.05
# python multi_agent.py --p_move 0.1
# python multi_agent.py --p_move 0.25
# python multi_agent.py --p_move 0.5

# Number of agents
# python multi_agent.py --num_agents 2 --episode_count 20000
# python multi_agent.py --num_agents 3 --episode_count 30000
# python multi_agent.py --num_agents 4 --episode_count 40000
# python multi_agent.py --num_agents 5 --episode_count 50000

# Random restarts
python multi_agent.py --num_agents 1 --rand_reset --episode_count 20000
python multi_agent.py --num_agents 2 --rand_reset --episode_count 30000
# python multi_agent.py --num_agents 3 --rand_reset --episode_count 30000
