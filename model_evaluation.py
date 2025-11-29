# model_evaluation.py
import os
import pickle
import pandas as pd
from stable_baselines3 import DQN
import numpy as np
from ad_env import AdBiddingEnv

MODEL_FILE = "dqn_ad_bidding_model.zip"
DATA_FILE = "ad_data.csv"

def evaluate_model(model_path=MODEL_FILE, data_path=DATA_FILE, episode_length=150, sample_rows=None):
    print("Evaluating model...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Ensure training saved the model.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found. Run prepare_avazu first.")

    df = pd.read_csv(data_path)
    if sample_rows:
        df = df.sample(n=sample_rows, random_state=1).reset_index(drop=True)

    env = AdBiddingEnv(df, episode_length=episode_length)
    model = DQN.load(model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    # run until done
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += float(reward)

    total_spend = float(env.total_spend)
    total_revenue = float(env.total_revenue)
    roas = (total_revenue / total_spend) if total_spend > 0 else 0.0

    print(f"Evaluation Complete | ROAS: {roas:.2f} | Spend: {total_spend:.2f} | Revenue: {total_revenue:.2f}")
    return roas, total_spend, total_revenue

if __name__ == "__main__":
    evaluate_model()
