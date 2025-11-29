# ad_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AdBiddingEnv(gym.Env):
    """
    Environment for ad bidding using tabular data (one row per auction/impression).
    Observation vector: [ctr, cvr, spend_today_norm, revenue_today_norm, current_bid]
    Action: discrete bid level 0..9 mapped to [0.0, 1.0]
    Reward: revenue_today - spend_today  (proxy for ROAS objective)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, episode_length=150):
        super().__init__()
        self.data = df.reset_index(drop=True)
        self.episode_length = episode_length
        self.current_step = 0

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)

        # running totals for episode
        self.total_spend = 0.0
        self.total_revenue = 0.0
        self.current_bid = 0.5

    def _get_obs(self, row):
        # row has ctr, cvr, impressions, cost, revenue
        ctr = float(row.get("ctr", 0.0))
        cvr = float(row.get("cvr", 0.0))
        # normalize spend/revenue per-row by some reasonable scale:
        spend_norm = float(row.get("cost", 0.0)) / 10.0  # scale down
        revenue_norm = float(row.get("revenue", 0.0)) / 100.0
        bid = float(self.current_bid)
        obs = np.array([ctr, cvr, spend_norm, revenue_norm, bid], dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_spend = 0.0
        self.total_revenue = 0.0
        self.current_bid = 0.5
        row = self.data.iloc[self.current_step]
        return self._get_obs(row), {}

    def step(self, action):
        bid = action / float(self.action_space.n - 1)  # map 0..9 to 0.0..1.0
        self.current_bid = bid

        if self.current_step >= len(self.data):
            return self._get_obs(self.data.iloc[-1]), 0.0, True, False, {}

        row = self.data.iloc[self.current_step]
        ctr = float(row.get("ctr", 0.0))
        cvr = float(row.get("cvr", 0.0))
        impressions = float(row.get("impressions", 1.0))
        base_cost = float(row.get("cost", 1.0))
        base_revenue = float(row.get("revenue", 0.0))

        # Simulate spend & revenue dependence on bid:
        spend_today = base_cost * (0.5 + bid)  # higher bid increases spend
        # revenue increases with CTR * CVR * bid * impressions
        revenue_today = base_revenue * (0.8 + 0.4 * bid)

        self.total_spend += spend_today
        self.total_revenue += revenue_today

        reward = revenue_today - spend_today

        self.current_step += 1
        done = self.current_step >= min(self.episode_length, len(self.data))

        obs = self._get_obs(row)
        info = {}
        return obs, float(reward), done, False, info

    def render(self):
        print(f"Step {self.current_step} | Bid {self.current_bid:.2f} | Spend {self.total_spend:.2f} | Revenue {self.total_revenue:.2f}")
