import numpy as np
import pandas as pd
import pickle
import os


def make_synthetic_ad_data(num_days=30, num_audiences=8, hours_per_day=24, save_path="synthetic_ad_data.pkl"):
    """
    Generate realistic synthetic ad campaign data.
    Each row = (day, hour, audience_id, ctr, cvr, impressions, cost, revenue)
    """

    np.random.seed(42)

    rows = []
    for day in range(num_days):
        for hour in range(hours_per_day):
            for audience in range(num_audiences):
                base_ctr = np.random.uniform(0.02, 0.12)
                base_cvr = np.random.uniform(0.01, 0.08)

                # Variation by hour and audience
                hour_factor = np.sin(hour / 24 * np.pi) + 1.2
                aud_factor = 0.8 + (audience / num_audiences) * 0.6

                ctr = base_ctr * hour_factor * aud_factor
                cvr = base_cvr * hour_factor * aud_factor

                impressions = np.random.randint(500, 5000)
                cost = np.random.uniform(0.2, 1.5) * impressions / 100
                revenue = impressions * ctr * cvr * np.random.uniform(30, 80)

                rows.append([
                    day, hour, audience, ctr, cvr, impressions, cost, revenue
                ])

    df = pd.DataFrame(rows, columns=[
        "day", "hour", "audience_id", "ctr", "cvr", "impressions", "cost", "revenue"
    ])

    # Save to pickle for reuse
    with open(save_path, "wb") as f:
        pickle.dump(df, f)

    print(f" Synthetic dataset generated: {df.shape[0]} rows saved to {save_path}")
    return df


if __name__ == "__main__":
    make_synthetic_ad_data(num_days=30, num_audiences=8, hours_per_day=24)
