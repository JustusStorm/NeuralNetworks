# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Reinforcement Models with Futures Market Trading

# %%


# %% [markdown]
# ### 4 main things to know for reinforcement models: 
# 1. (A)gent = the model itself or the thing you are training(think of this like a puppy, a non potty trained puppy)
# 2. (R)eward = you gotta give the damn thing candy or something when it does well 
# 3. (E)nvironment = the place, yard, cage, underground cellar, etc. that your puppy (agent) lives and plays in
# 4. (A)ction = things that your Agent puppy can do 
# 
# 
# 
# Did you notice it spells out AREA? Kinda like Area 51, which makes sense - were about to try to prove something 99% of people think is impossible
# %% [markdown]
# ---
# %% [markdown]
# # 1. Import the Data
# %% [markdown]
# By the way, if you want to watch one of the videos I'm referencing, [click here](https://www.youtube.com/watch?v=D9sU1hLT0QY&list=PLPmc5znn8pmLU409mOLfbFEQOpUVfC0MB&index=19)
# 
# 
# <iframe src="https://giphy.com/embed/l4JyTTj946267CUtq" width="480" height="227" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/producthunt-rogue-one-intro-meow-l4JyTTj946267CUtq">via GIPHY</a></p>
# 
# __In the beginning...__
# __In a galaxy far, far away..__
# 
# We gather the data we need for this miraculous machine software mechanism magic. Data must be clean and organized in order for your model to be accurate at whatever you are _hoping_ to accomplish.

# %%
import gym 
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 


# %%
# Ye good ole' import
filepath = 'Data/MNQU21_Micro-e-min-Nasdaq_5min_3months.csv'
df = pd.read_csv(filepath)

df.drop(['Time'], axis=1, inplace = True)
#df.drop(['OI'], axis=1, inplace = True)
df.drop(['Up'], axis=1, inplace = True)
df.drop(['Down'], axis=1, inplace = True)

#df = df[:100]

#df.drop(['Time', 'OI', 'Up','Down'], axis=1, inplace=True) # I'm adding this here because it's quick and easy. This should really be below in the part 2


# %%
# lets see what we have here     (0.0)
print(df)
df.head()

# %% [markdown]
# ---
# %% [markdown]
# # 2. Getting our data ready for a Gym environment
# %% [markdown]
# ### Change data type of Date column to datetime
# 
# The index of the data must be set to the date and/or time column in order for the data to be accepted
# 
# 
# _Note: I think this is what I need to do in one of the other projects in order for it to start working correctly_

# %%
# change the Date column into DateTime datatypes 
df.dtypes
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes

# %% [markdown]
# ### Make sure you have at least these 4 columns when using gym_anytrading...
# 
# The following columns are required for importing data in Gym: 
# * Open 
# * Close 
# * High 
# * Low 
# 
# 

# %%
df.set_index('Date', inplace=True)
df.head

# %% [markdown]
# ---
# %% [markdown]
# # 3. Create our Environment 
# %% [markdown]
# __stocks-vo__ is the name of the pre built environment that we are using. I believe there is another pre built environment similar to this one named __forex-v0__ maybe? I will need to do some research on that
# 
# * df = our dataframe
# * frame_bound =(x, y) where: 
#     * x = where the agent should start in the data provided
#     * y = total number of rows of data of the dataframe that is being used
# * window_size = how many rows will be observed before making output

# %%
# create the underground cellar for the puppy (cough)... (cough)... I mean agent

env = gym.make('stocks-v0', df=df, frame_bound=(20, 100), window_size=20)


# %%
# shows the list of prices in the environment 
env.prices


# %%
# I'm not sure what the first column is, but the second column is the difference between values
# You can see this because

env.signal_features

# %% [markdown]
# ---
# %% [markdown]
# # 4. Build our Environment, and throw this puppy in that cellar and see what it does!
# %% [markdown]
# This is not a true test of our model, we are just thowing our agent in the environment and telling it to do random things to make sure everything is connected properly.

# %%
state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done: 
        print('info', info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

# %% [markdown]
# # 5. Build training environment and train the agent

# %%
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(20,12000), window_size=20)
env = DummyVecEnv([env_maker])


# %%
model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=2000)


# %%
env = gym.make('stocks-v0', df=df, frame_bound=(12000,12200), window_size=20)
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print('info', info)
        break


# %%
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


