# Taxi - TD Methods

### Introduction
This project is part of the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) by [Udacity](https://www.udacity.com/). This code demonstrates a simple 
implementation of a Deep Q Network. In this exercise an agent learns how to navigate a virtual environment, collecting 
and dropping off a virtual passenger at a target location.

### Environment
A full description of the environment can be found [here](https://gym.openai.com/envs/Taxi-v3/).

### Requirements
<ul>
<li>gym - v0.18.0</li>
<li>numpy - v1.19.5</li>
</ul>

### Instructions
The workspace contains three files:

`taxi_agent.py`: Contains the instructions for the reinforcement learning agent.<br/>
`taxi_monitor.py`: Contains the interact function, which tests how well the agent learns from interaction with the environment.<br/>
`taxi_main.py`: Contains the hyperparameters of the agent. Run this file in the terminal to check the performance of the agent.

To run the agent:
1. Install dependables
2. Open `taxi_main.py`
3. Edit the hyperparameters (optional) 
4. Run `taxi_main.py`
   
This will call the agent (specified in `agent.py`) to interact with the environment for 20,000 episodes. 
The results of the interactions are recorded in `monitor.py`, which returns two variables:<br/>

`avg_rewards`: A deque containing the average rewards.<br/>
`best_avg_reward`: The largest value in the avg_rewards deque.

### Licence
#### MIT License

Copyright (c) 2021 Colm Dowling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.