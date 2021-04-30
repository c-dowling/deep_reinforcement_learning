import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA, epsilon, eps_decay, eps_min, alpha, gamma, td_control):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - eps_start: the initial value for epsilon
        - eps_decay: the amount that epsilon is multiplied by at each episode
        - eps_min: the minimum value at which point epsilon will no longer decay
        - alpha: the learning rate
        - gamma: the discount parameter
        - td_control: the control method for updating the q table ("sarsa" | "sarsamax")
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.alpha = alpha
        self.gamma = gamma
        self.td_control = td_control
        print(f"epsilon - {self.epsilon}, eps_decay - {self.eps_decay}, eps_min - {self.eps_min}, alpha - {self.alpha}, gamma - {self.gamma}, td control method = {self.td_control}")
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = self.epsilon
        epsilon = np.max((epsilon*self.eps_decay, self.eps_min))
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA)
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        self.epsilon = epsilon
        return action
        

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            next_action = self.select_action(state=next_state)
            if self.td_control == "expectedsarsa":
                self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + (self.gamma*self.Q[next_state][self.select_action(state=next_state)]) - self.Q[state][action])
            if self.td_control == "sarsa":
                self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + (self.gamma*self.Q[next_state][next_action]) - self.Q[state][action])
            if self.td_control == "sarsamax":
                self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + (self.gamma*self.Q[next_state][np.argmax(self.Q[next_state])]) - self.Q[state][action])
        if done:     
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + 0 - self.Q[state][action])        
