from taxi_agent import Agent
from taxi_monitor import interact
import gym

# Hyperparameters
nA=6                            # Number of actions available to the agent
epsilon=1                       # Initial value of epsilon
eps_decay=0.9999                # The amount that epsilon is multiplied by at each episode
eps_min=0                       # The minimum value at which point epsilon will no longer decay
alpha=0.10                      # Learning Rate
gamma=0.9                       # Discount parameter
td_control="expectedsarsa"      # Control method for updating the Q table - ("sarsa" / "sarsamax" / "expectedsarsa")

# Create our environment and train our agent
env = gym.make('Taxi-v3')
agent = Agent(nA, epsilon, eps_decay, eps_min, alpha, gamma, td_control)
avg_rewards, best_avg_reward = interact(env, agent)