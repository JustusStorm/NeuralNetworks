import gym 
import json
import datetime as dt 
import pandas as pd 

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from CustomEnv_Justus import MarketTradingEnv


df = pd.read_csv('././Data/AAPL.csv')
df = df.sort_values('Date')
#df = df.drop(['Time'], inplace=True, axis=1)
print(df)

# The algorithms require a vertorized environment to run 
env = DummyVecEnv([lambda: MarketTradingEnv(
    df, 
    MAX_ACCOUNT_BALANCE=1000000,
    MAX_SHARE_PRICE=1000000, 
    MAX_NUM_SHARES=1000000000, 
    INITIAL_ACCOUNT_BALANCE=20000, 
    MAX_STEPS=120)
    ])

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=120)


obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()