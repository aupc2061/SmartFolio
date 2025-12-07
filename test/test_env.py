import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from argparse import Namespace

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.portfolio_env import StockPortfolioEnv

def setup_dummy_env(risk_score, num_stocks=10, max_cap=None):
    """Creates a lightweight environment for testing logic."""
    steps = 5
    args = Namespace(
        risk_score=risk_score,
        input_dim=6,
        lookback=30
    )
    
    # Create Risk Profile with Constraints
    risk_profile = {'risk_score': risk_score}
    if max_cap:
        risk_profile['max_weight'] = max_cap
    
    env = StockPortfolioEnv(
        args=args,
        ts_features=torch.randn(steps, num_stocks, 30, 6),
        returns=torch.randn(steps, num_stocks),
        ind_yn=False, pos_yn=False, neg_yn=False,
        mode="test",
        risk_profile=risk_profile # Pass the profile!
    )
    return env

def test_risk_engine():
    print("--- ⚙️ Testing Environment Risk Engine (The Filter) ---")
    
    num_stocks = 5
    raw_actions = np.random.rand(num_stocks)
    raw_actions[0] += 10.0 
    
    print(f"1. AI Output (Raw Logits): {raw_actions}")
    
    # TEST 1: AGGRESSIVE USER (Risk 0.9, No Cap)
    print("\n2. Processing for Aggressive User (Risk=0.9)...")
    env_agg = setup_dummy_env(risk_score=0.9, num_stocks=num_stocks)
    
    temp_agg = env_agg.action_temperature
    print(f"   -> Temperature: {temp_agg:.2f}")
    
    exp_agg = np.exp((raw_actions - np.max(raw_actions)) / temp_agg)
    weights_agg = exp_agg / exp_agg.sum()
    
    # Apply Constraints (Aggressive usually has none or high cap)
    weights_agg = env_agg._apply_risk_constraints(weights_agg)
    print(f"   -> Final Weight (Stock 0): {weights_agg[0]*100:.1f}%")

    # TEST 2: CONSERVATIVE USER (Risk 0.1, Cap 20%)
    print("\n3. Processing for Conservative User (Risk=0.1, Cap=20%)...")
    # FIX: We give the conservative user a Hard Cap of 0.20
    env_con = setup_dummy_env(risk_score=0.1, num_stocks=num_stocks, max_cap=0.20)
    
    temp_con = env_con.action_temperature
    print(f"   -> Temperature: {temp_con:.2f}")
    
    exp_con = np.exp((raw_actions - np.max(raw_actions)) / temp_con)
    weights_con = exp_con / exp_con.sum()
    print(f"   -> Pre-Constraint Weight: {weights_con[0]*100:.1f}% (Too High)")
    
    # Apply Hard Constraints (The Real Safety Net)
    weights_con = env_con._apply_risk_constraints(weights_con)
    print(f"   -> Final Weight (Stock 0): {weights_con[0]*100:.1f}% (Capped)")

    # VERIFICATION
    if weights_agg[0] > 0.9 and weights_con[0] <= 0.21: # Allow small float error
        print("\n PASS: The Risk Engine correctly filtered the actions.")
        print("   Aggressive: Allowed Concentration.")
        print("   Conservative: Enforced Diversification via Hard Caps.")
    else:
        print("\n FAIL: Risk Engine logic failed.")

    # 4. Visualization
    plt.figure(figsize=(8, 5))
    indices = np.arange(num_stocks)
    width = 0.35
    
    plt.bar(indices - width/2, weights_agg, width, label=f'Aggressive (Risk 0.9)', color='red', alpha=0.7)
    plt.bar(indices + width/2, weights_con, width, label=f'Conservative (Risk 0.1)', color='blue', alpha=0.7)
    
    plt.title("The Safety Net: Hard Caps vs. AI Confidence")
    plt.xlabel("Stock Index")
    plt.ylabel("Final Portfolio Weight")
    plt.xticks(indices, [f"Stock {i}" for i in indices])
    plt.legend()
    plt.axhline(0.2, color='blue', linestyle='--', alpha=0.5, label='Conservative Cap (20%)')
    plt.tight_layout()
    plt.savefig("test/env_risk_engine.png")
    print("Chart saved to: test/env_risk_engine.png")

if __name__ == "__main__":
    test_risk_engine()
