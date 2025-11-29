# train_dqn.py
import os
import pickle
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from ad_env import AdBiddingEnv

MODEL_FILE = "dqn_ad_bidding_model.zip"
DATA_FILE = "ad_data.csv"

def train_model(total_timesteps=40000, episode_length=150, sample_rows=None):
    print("Starting DQN training...")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run prepare_avazu.py first to create it.")

    df = pd.read_csv(DATA_FILE)
    if sample_rows:
        df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)

    env = DummyVecEnv([lambda: AdBiddingEnv(df, episode_length=episode_length)])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.98,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_FILE)
    print("Saved model to", MODEL_FILE)
    return MODEL_FILE

if __name__ == "__main__":
    train_model(total_timesteps=20000, episode_length=150, sample_rows=50000)
