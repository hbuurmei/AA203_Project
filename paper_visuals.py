import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_reward(input_names, output_name, hyperparameters):
    # Plot the reward for each episode

    # List to store dataframes
    dfs = []

    # Read all csv files into pandas DataFrames
    for file_name in input_names:
        df = pd.read_csv(f'./results/training_log/{file_name}.csv')
        dfs.append(df)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Set the style of the plot
    sns.set_style("ticks")

    # Set the linewidth of the axes lines
    # for axis in ['top','bottom','left','right']:
        # ax.spines[axis].set_linewidth(2)

    # Define color palette
    colors = sns.color_palette('coolwarm', len(dfs))  # 'viridis' is one of the color palettes

    # Plot the reward for each DataFrame
    for i, df in enumerate(reversed(dfs)):
        sns.lineplot(x='episode', y=' reward', data=df, label=hyperparameters[len(dfs) - 1 - i], ax=ax, color=colors[i])

    # Add labels and remove top and right spines
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    sns.despine()

    # Add a legend
    plt.legend()

    # Save the figure
    plt.savefig(f'./results/{output_name}.svg', dpi=300, format='svg')
    plt.savefig(f'./results/{output_name}.png', dpi=300)


## Plotting the reward for each episode, with different hyperparameters

# Environment size
exp_name = 'env_size_experiment'
env_sizes = range(8, 32 + 1, 4)
env_size_files = [f'N_agents_1_p_move_0.05_rand_reset_None_episode_count_10000_env_size_{size}_action_range_5' for size in env_sizes]
env_size_hyperparameters = [fr'{size}$\times${size} grid' for size in env_sizes]
plot_reward(env_size_files, exp_name, env_size_hyperparameters)

# Action range
exp_name = 'action_range_experiment'
action_ranges = range(3, 9 + 1, 2)
action_range_files = [f'N_agents_1_p_move_0.05_rand_reset_None_episode_count_10000_env_size_10_action_range_{range}' for range in action_ranges]
action_range_hyperparameters = [fr'{range}$\times${range} range' for range in action_ranges]
plot_reward(action_range_files, exp_name, action_range_hyperparameters)

# Movement cost
exp_name = 'movement_cost_experiment'
# movement_costs = [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
movement_costs = [0.0, 0.05, 0.1, 0.25, 0.5]
movement_cost_files = [f'N_agents_1_p_move_{cost}_rand_reset_None_episode_count_10000_env_size_10_action_range_5' for cost in movement_costs]
movement_cost_hyperparameters = [fr'$p_{{move}} = {cost}$' for cost in movement_costs]
plot_reward(movement_cost_files, exp_name, movement_cost_hyperparameters)

# Number of agents
exp_name = 'N_agents_experiment'
N_agents = range(1, 3 + 1)
N_agents_files = [f'N_agents_{N}_p_move_0.05_rand_reset_None_episode_count_{N * 10000}_env_size_10_action_range_5' for N in N_agents]
N_agents_hyperparameters = [fr'{N} agent' + ('s' if N > 1 else '') for N in N_agents]
plot_reward(N_agents_files, exp_name, N_agents_hyperparameters)

