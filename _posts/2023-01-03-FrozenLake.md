---
layout: single
title:  "Frozen Lake using Q-Learning"
---

# Libraries to import


```python
!pip install gym
!pip install gym[toy_text]
```


```python
import gym

import numpy as np
import random
```

# Creating an environment


```python
env = gym.make("FrozenLake-v1", is_slippery = False, render_mode="human")
```


```python
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

qtable = np.zeros((state_space_size, action_space_size))
```

# Training Variables



```python
total_episodes = 10000
max_step = 100 # Prevents infinite loop

alpha = 0.2 # Learning rate
gamma = 0.001 # value of future reward

epsilon = 1 # For ε-Greedy policy
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.001
```

# Training with Q-Learning

* By uncommenting two lines at the top, training process could be visually seen


```python
#env.reset()
#env.render()

rewards = []

for episode in range(total_episodes):
    env.reset()
    state = 0
    step = 0
    terminated = False
    episode_reward = 0
    
    for step in range(max_steps):
        if(random.uniform(0,1) > epsilon):
            action = np.argmax(qtable[state,:]) # Exploit: Pick an action that maximises Q from the qtable
        else:
            action = env.action_space.sample() # Exploration: Picks a random action
        
        new_state, reward, terminated, truncated, info = env.step(action) # Return values of .step() function
        
        # Q-Learning equation
        qtable[state,action] = (1 - learning_rate) * qtable[state,action] + learning_rate * (reward + gamma * np.max(qtable[new_state,:]) - qtable[state, action])
        
        state = new_state
        
        if(terminated):
            break
            
        episode_reward += reward
        
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate * (episode)) # Exponentially decrease ε to become Greedy policy
    rewards.append(episode_reward)
        
env.close()
```

# Testing the agent

* Number of tests could be modifies through changing 'test_episode' variable
* However, since the agent is set to choose an optimal Q (Greedy policy), the action that the agent will take would be the same for every single test


```python
env = gym.make("FrozenLake-v1", is_slippery = False, render_mode="human")
env.reset()
env.render()

ave_reward = 0
test_episodes = 1 # Number of episodes to change

for episode in range(test_episodes):
    env.reset()
    state = 0
    terminated = False
    
    print("Episode:", episode + 1)
    
    for step in range(max_steps):
        action = np.argmax(qtable[state,:]) # Exploit: Pick an action that maximises Q from the qtable
        new_state, reward, terminated, truncated, info =  env.step(action)
        
        ave_reward += reward
        
        if terminated:
            print("Number of steps:",step)
            print("Reward:", reward)
            break
            
        state = new_state

print("Average reward:", ave_reward/test_episodes)

env.close()
```
