import gym 
import json
import random as rd
import datetime as dt 
import numpy as np

from gym import spaces 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO



class MarketTradingEnv(gym.Env): 
    """A futures/stock trading environment for OpenAI gym"""
    metadata = {
        'render.modes': ['human']        
        }

    # MAX_ACCOUNT_BALANCE = null
    # MAX_SHARE_PRICE = null
    # MAX_NUM_SHARES = null
    # INITIAL_ACCOUNT_BALANCE = null
    # MAX_STEPS = null

    def __init__(self, df, MAX_ACCOUNT_BALANCE: 1000000, MAX_SHARE_PRICE: 1000000, MAX_NUM_SHARES: 1, INITIAL_ACCOUNT_BALANCE: 20000, MAX_STEPS: 120):
        super(MarketTradingEnv, self).__init__()
        self.df = df
        self.MAX_ACCOUNT_BALANCE = MAX_ACCOUNT_BALANCE
        self.MAX_SHARE_PRICE = MAX_SHARE_PRICE
        self.MAX_NUM_SHARES = MAX_NUM_SHARES
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.MAX_STEPS = MAX_STEPS # This is 120 days, or 4 months

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x% , Sell x%, Hold, etc. 
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16
        )

        # Prices contain OCHL values for the last 5 prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,6), dtype=np.float16
        )

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / self.MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / self.MAX_NUM_SHARES,
        ])
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / self.MAX_ACCOUNT_BALANCE,
            self.max_net_worth / self.MAX_ACCOUNT_BALANCE,
            self.shares_held / self.MAX_NUM_SHARES,
            self.cost_basis /self. MAX_SHARE_PRICE,
            self.total_shares_sold / self.MAX_NUM_SHARES,
            self.total_sales_value / (self.MAX_NUM_SHARES * self.MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def reset(self):
        # Reset the state of the environment to an initial state 
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0 
        self.cost_basis = 0 
        self.total_shares_sold = 0 
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame 
        self.current_step = rd.randint(0,len(self.df.loc[:,'Open'].values - 6))

        return self._next_observation()


    def step(self, action):
        # Execute one time step within the environment 
        self.take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values)-6:
            self.current_step == 0

        # I really like the thought of using this below. This will cause the agent to like to have a bigger balance as it get later into the data
        delay_modifier = (self.current_step/self.MAX_STEPS)

        reward = self.balance* delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def take_action(self, action):
        # Set the current price to random price bewteen open and close of time step 
        current_price = rd.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close']
        )

        action_type = action[0]
        amount = action[1]

        if action_type < 1: 
            # Buy amount % of balance in shares 
            total_possible = self.balance / current_price
            shares_bought = total_possible * amount
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held * current_price)
            
            self.shares_held += shares_bought

        elif action_type < 2: 
            # Sell amount % of shares held 
            shares_sold = self.shares_held * amount 
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        
        self.net_worth = self.balance + self.shares_held * current_price


        if self.net_worth> self.max_net_worth: 
            self.max_net_worth = self.net_worth

        if self.shares_held == 0: 
            self.cost_basis = 0


    
    def render(self, mode='human', close=False):
        # Render env to screen 
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')


        






