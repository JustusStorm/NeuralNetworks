# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 0. Importing Libraries

# %%
# !pip install tensorflow===2.3.0
# !pip install gym 
# !pip install keras
# !pip install keras-r12

# %% [markdown]
# # 1. Test Random Environment with Open Al Gym 

# %%
import gym 
import random 


# %%
# loading in environment we have available 
env = gym.make('CartPole-v0')
# num of states we have available
states = env.observation_space.shape[0]
# looks at the actions we have available in this environment 
actions = env.action_space.n


# %%
# ouputs a count of the actions we can use with the agent
actions


# %%
episodes = 10 
for episode in range(0, episodes+1):
    state = env.reset()
    done = False 
    score = 0 

    while not done: 
        env.render()
        action = random.choice([0, 1])
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))


# We are wanting this output to be as close to 200 as possible 

# %% [markdown]
# # 2. Create a Deep Learning Model 

# %%
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.optimizers import Adam


# %%
# method to build model

def build_model(states, actions): 
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model 


# %%
# build model 
model = build_model(states, actions)


# %%
model.summary()

# %% [markdown]
# # 3. Build Agent with Keras-RL 

# %%
from rl.agents import DQNAgent 
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory 


# %%
# make method to build an agent 
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

    return dqn


# %%
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


# %%
scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

# %% [markdown]
# # 4. Reloading Agent From Memory

# %%
dqn.save_weights('dqn_weights.h5f', overwrite=True)


# %%
del model 
del dqn 
del env


# %%
env = gym.make('CartPole-v0')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics = ['mae'])


# %%
dqn.load_weights('dqn_weights.h5f')


# %%
_ = dqn.test(env, nb_episodes=5, visualize=True)


