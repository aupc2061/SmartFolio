"""
Kelly Criterion + Goal-Conditioned Expert with Continuous Weights
Best approach for portfolio optimization IRL
FIXED: Handles both 2D and 3D feature arrays with proper dimension handling
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy.optimize import minimize
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class KellyExpertStrategy:
    """Kelly Criterion and variants for theoretically optimal portfolio allocation"""
    
    def __init__(self, variant='fractional', fraction=0.5, confidence=0.95, 
                 max_drawdown=0.20):
        self.variant = variant
        self.fraction = fraction
        self.confidence = confidence
        self.max_drawdown = max_drawdown
    
    def fractional_kelly(self, returns, covariance):
        """Fractional Kelly: f* = fraction × Σ^{-1} μ"""
        try:
            cov_reg = covariance + 1e-6 * np.eye(len(covariance))
            cov_inv = np.linalg.inv(cov_reg)
            kelly_weights = self.fraction * np.dot(cov_inv, returns)
            kelly_weights = np.maximum(kelly_weights, 0)
            if kelly_weights.sum() > 0:
                kelly_weights = kelly_weights / kelly_weights.sum()
            else:
                kelly_weights = np.ones(len(returns)) / len(returns)
            return kelly_weights
        except:
            return np.ones(len(returns)) / len(returns)
    
    def robust_kelly(self, returns, covariance):
        """Robust Kelly: Accounts for estimation uncertainty"""
        n = len(returns)
        returns_std = np.sqrt(np.diag(covariance) / max(n, 10))
        z_score = norm.ppf((1 + self.confidence) / 2)
        worst_case_returns = returns - z_score * returns_std
        return self.fractional_kelly(worst_case_returns, covariance)
    
    def kelly_with_drawdown_control(self, returns, covariance):
        """Kelly with maximum drawdown constraint"""
        kelly_weights = self.fractional_kelly(returns, covariance)
        portfolio_vol = np.sqrt(np.dot(kelly_weights, 
                                       np.dot(covariance, kelly_weights)))
        estimated_dd = 2 * portfolio_vol
        if estimated_dd > self.max_drawdown and estimated_dd > 0:
            scale = self.max_drawdown / estimated_dd
            kelly_weights = scale * kelly_weights
            kelly_weights = kelly_weights / kelly_weights.sum()
        return kelly_weights
    
    def constrained_kelly(self, returns, covariance, max_weight=0.30):
        """Kelly with position size constraints"""
        n = len(returns)
        
        def objective(w):
            portfolio_return = np.dot(w, returns)
            portfolio_variance = np.dot(w, np.dot(covariance, w))
            log_wealth = portfolio_return - 0.5 * portfolio_variance
            return -log_wealth
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, max_weight) for _ in range(n)]
        w0 = np.ones(n) / n
        
        result = minimize(objective, w0, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
    
    def optimize(self, returns, covariance, constraints=None):
        """Main optimization method"""
        if constraints is None:
            constraints = {}
        max_weight = constraints.get('max_weight', 0.30)
        
        if self.variant == 'fractional':
            weights = self.fractional_kelly(returns, covariance)
        elif self.variant == 'robust':
            weights = self.robust_kelly(returns, covariance)
        elif self.variant == 'drawdown':
            weights = self.kelly_with_drawdown_control(returns, covariance)
        elif self.variant == 'constrained':
            weights = self.constrained_kelly(returns, covariance, max_weight)
        else:
            weights = self.fractional_kelly(returns, covariance)
        
        return weights


class GoalConditionedExpert:
    """Goal-conditioned expert that generates diverse optimal behaviors"""
    
    def __init__(self, randomize_goals=True):
        self.randomize_goals = randomize_goals
        
        self.goal_strategies = {
            'high_growth': {
                'kelly_variant': 'fractional',
                'kelly_fraction': 0.75,
                'risk_aversion': 0.5,
                'description': 'Maximize long-term growth'
            },
            'balanced': {
                'kelly_variant': 'fractional',
                'kelly_fraction': 0.50,
                'risk_aversion': 2.0,
                'description': 'Balance risk and return'
            },
            'conservative': {
                'kelly_variant': 'robust',
                'kelly_fraction': 0.25,
                'risk_aversion': 4.0,
                'description': 'Minimize risk'
            },
            'drawdown_control': {
                'kelly_variant': 'drawdown',
                'kelly_fraction': 0.50,
                'max_drawdown': 0.15,
                'description': 'Control maximum drawdown'
            },
            'constrained': {
                'kelly_variant': 'constrained',
                'max_weight': 0.20,
                'description': 'Diversified with constraints'
            },
            'aggressive': {
                'kelly_variant': 'fractional',
                'kelly_fraction': 1.0,
                'risk_aversion': 0.25,
                'description': 'Aggressive growth'
            }
        }
    
    def encode_goal(self, goal):
        """One-hot encode goal for state augmentation"""
        goals = list(self.goal_strategies.keys())
        goal_vec = np.zeros(len(goals))
        goal_vec[goals.index(goal)] = 1.0
        return goal_vec
    
    def get_expert_weights(self, returns, covariance, goal, constraints=None):
        """Get continuous expert weights for a specific goal"""
        if goal not in self.goal_strategies:
            goal = 'balanced'
        
        strategy = self.goal_strategies[goal]
        
        kelly = KellyExpertStrategy(
            variant=strategy['kelly_variant'],
            fraction=strategy.get('kelly_fraction', 0.5),
            max_drawdown=strategy.get('max_drawdown', 0.20)
        )
        
        weights = kelly.optimize(returns, covariance, constraints)
        return weights
    
    def generate_expert_action(self, returns, correlation_matrix, 
                              industry_matrix=None, goal=None, constraints=None):
        """
        Generate expert action with continuous weights
        Returns: (continuous_weights, binary_actions, goal)
        """
        if constraints is None:
            constraints = {
                'max_weight': 0.30,
                'max_positions': int(len(returns) * 0.1)
            }
        
        # Convert correlation to covariance
        volatility = np.ones(len(returns)) * 0.20
        covariance = np.outer(volatility, volatility) * correlation_matrix
        
        # Select goal
        if goal is None and self.randomize_goals:
            goal = np.random.choice(list(self.goal_strategies.keys()))
        elif goal is None:
            goal = 'balanced'
        
        # Get continuous weights
        continuous_weights = self.get_expert_weights(returns, covariance, goal, constraints)
        
        # Convert to binary actions (top-k selection)
        max_positions = constraints.get('max_positions', int(len(returns) * 0.1))
        binary_actions = np.zeros(len(returns), dtype=int)
        top_k_indices = np.argsort(-continuous_weights)[:max_positions]
        binary_actions[top_k_indices] = 1
        
        return continuous_weights, binary_actions, goal
    
    def generate_trajectories(self, dataset, num_trajectories=100, args=None):
        """Generate goal-conditioned expert trajectories with continuous weights"""
        trajectories = []
        
        for _ in range(num_trajectories):
            idx = np.random.randint(0, len(dataset))
            data = dataset[idx]
            
            features = data['features'].numpy()
            returns = data['labels'].numpy()
            correlation_matrix = data['corr'].numpy()
            
            # Get industry matrices
            ind_matrix = data.get('industry_matrix')
            if ind_matrix is not None:
                ind_matrix = ind_matrix.numpy()
            else:
                # Fallback to creating from data if available
                n_stocks = len(returns) if returns.ndim == 1 else returns.shape[-1]
                ind_matrix = np.eye(n_stocks)
            
            pos_matrix = data.get('pos_matrix')
            if pos_matrix is not None:
                pos_matrix = pos_matrix.numpy()
            else:
                pos_matrix = np.zeros_like(ind_matrix)
            
            neg_matrix = data.get('neg_matrix')
            if neg_matrix is not None:
                neg_matrix = neg_matrix.numpy()
            else:
                neg_matrix = np.zeros_like(ind_matrix)
            
            # Normalize returns to 1D (n_stocks,)
            if returns.ndim == 2:
                returns = returns[-1, :]
            elif returns.ndim == 0:
                returns = np.array([returns])
            
            n_stocks = len(returns)
            
            # Normalize features to 2D (n_stocks, n_features)
            if features.ndim == 3:
                features = features[-1, :, :]
            elif features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Ensure consistency
            if features.shape[0] != n_stocks:
                n_stocks = min(features.shape[0], n_stocks)
                features = features[:n_stocks, :]
                returns = returns[:n_stocks]
                if correlation_matrix.shape[0] > n_stocks:
                    correlation_matrix = correlation_matrix[:n_stocks, :n_stocks]
                if ind_matrix.shape[0] > n_stocks:
                    ind_matrix = ind_matrix[:n_stocks, :n_stocks]
                if pos_matrix.shape[0] > n_stocks:
                    pos_matrix = pos_matrix[:n_stocks, :n_stocks]
                if neg_matrix.shape[0] > n_stocks:
                    neg_matrix = neg_matrix[:n_stocks, :n_stocks]
            
            goal = np.random.choice(list(self.goal_strategies.keys()))
            
            constraints = {
                'max_weight': 0.30,
                'max_positions': int(n_stocks * 0.1)
            }
            
            continuous_weights, binary_actions, _ = self.generate_expert_action(
                returns, correlation_matrix, ind_matrix, goal, constraints
            )
            
            # Create flattened state matching the old format
            # Format: [ind_matrix (flattened), pos_matrix (flattened), 
            #          neg_matrix (flattened), features (flattened), goal_vector (broadcast)]
            state_parts = []
            
            # Add matrices based on args flags (if provided)
            if args is not None:
                if args.ind_yn:
                    state_parts.append(ind_matrix.flatten())
                else:
                    state_parts.append(np.zeros(ind_matrix.size))
                if args.pos_yn:
                    state_parts.append(pos_matrix.flatten())
                else:
                    state_parts.append(np.zeros(pos_matrix.size))
                if args.neg_yn:
                    state_parts.append(neg_matrix.flatten())
                else:
                    state_parts.append(np.zeros(neg_matrix.size))
            else:
                # Default: include all matrices
                state_parts.append(ind_matrix.flatten())
                state_parts.append(pos_matrix.flatten())
                state_parts.append(neg_matrix.flatten())
            
            # Add features
            state_parts.append(features.flatten())
            
            # Add goal vector (replicate for each stock's feature set)
            goal_vector = self.encode_goal(goal)
            # Broadcast goal to each stock
            goal_broadcast = np.tile(goal_vector, n_stocks)
            state_parts.append(goal_broadcast)
            
            # Concatenate all parts into single 1D state vector
            state = np.concatenate(state_parts)
            
            trajectories.append({
                'state': state,
                'continuous_weights': continuous_weights,
                'binary_actions': binary_actions,
                'goal': goal,
                'goal_vector': goal_vector
            })
        
        return trajectories


class HybridKellyGoalExpert:
    """Combines Kelly variants with goal conditioning for maximum diversity"""
    
    def __init__(self):
        self.goal_expert = GoalConditionedExpert(randomize_goals=True)
        self.kelly_variants = ['fractional', 'robust', 'drawdown', 'constrained']
        self.fractions = [0.25, 0.50, 0.75, 1.0]
    
    def generate_trajectories(self, dataset, num_trajectories=100, args=None):
        """Generate highly diverse Kelly + Goal trajectories with continuous weights"""
        trajectories = []
        
        # Phase 1: Goal-conditioned (40%)
        n_goal = int(num_trajectories * 0.4)
        goal_trajs = self.goal_expert.generate_trajectories(dataset, n_goal, args)
        trajectories.extend(goal_trajs)
        
        # Phase 2: Kelly variants (30%)
        n_kelly = int(num_trajectories * 0.3)
        for _ in range(n_kelly):
            idx = np.random.randint(0, len(dataset))
            data = dataset[idx]
            
            features = data['features'].numpy()
            returns = data['labels'].numpy()
            correlation_matrix = data['corr'].numpy()
            
            # Get industry matrices
            ind_matrix = data.get('industry_matrix')
            if ind_matrix is not None:
                ind_matrix = ind_matrix.numpy()
            else:
                n_stocks = len(returns) if returns.ndim == 1 else returns.shape[-1]
                ind_matrix = np.eye(n_stocks)
            
            pos_matrix = data.get('pos_matrix')
            if pos_matrix is not None:
                pos_matrix = pos_matrix.numpy()
            else:
                pos_matrix = np.zeros_like(ind_matrix)
            
            neg_matrix = data.get('neg_matrix')
            if neg_matrix is not None:
                neg_matrix = neg_matrix.numpy()
            else:
                neg_matrix = np.zeros_like(ind_matrix)
            
            # Normalize returns to 1D
            if returns.ndim == 2:
                returns = returns[-1, :]
            elif returns.ndim == 0:
                returns = np.array([returns])
            
            n_stocks = len(returns)
            
            # Normalize features to 2D
            if features.ndim == 3:
                features = features[-1, :, :]
            elif features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Ensure consistency
            if features.shape[0] != n_stocks:
                n_stocks = min(features.shape[0], n_stocks)
                features = features[:n_stocks, :]
                returns = returns[:n_stocks]
                if correlation_matrix.shape[0] > n_stocks:
                    correlation_matrix = correlation_matrix[:n_stocks, :n_stocks]
                if ind_matrix.shape[0] > n_stocks:
                    ind_matrix = ind_matrix[:n_stocks, :n_stocks]
                if pos_matrix.shape[0] > n_stocks:
                    pos_matrix = pos_matrix[:n_stocks, :n_stocks]
                if neg_matrix.shape[0] > n_stocks:
                    neg_matrix = neg_matrix[:n_stocks, :n_stocks]
            
            variant = np.random.choice(self.kelly_variants)
            fraction = np.random.choice(self.fractions)
            
            kelly = KellyExpertStrategy(variant=variant, fraction=fraction)
            
            volatility = np.ones(len(returns)) * 0.20
            covariance = np.outer(volatility, volatility) * correlation_matrix
            
            continuous_weights = kelly.optimize(returns, covariance)
            
            k = int(len(returns) * 0.1)
            binary_actions = np.zeros(len(returns), dtype=int)
            binary_actions[np.argsort(-continuous_weights)[:k]] = 1
            
            # Create flattened state matching the old format
            state_parts = []
            
            if args is not None:
                if args.ind_yn:
                    state_parts.append(ind_matrix.flatten())
                else:
                    state_parts.append(np.zeros(ind_matrix.size))
                if args.pos_yn:
                    state_parts.append(pos_matrix.flatten())
                else:
                    state_parts.append(np.zeros(pos_matrix.size))
                if args.neg_yn:
                    state_parts.append(neg_matrix.flatten())
                else:
                    state_parts.append(np.zeros(neg_matrix.size))
            else:
                state_parts.append(ind_matrix.flatten())
                state_parts.append(pos_matrix.flatten())
                state_parts.append(neg_matrix.flatten())
            
            state_parts.append(features.flatten())
            
            # No goal for pure Kelly variants
            state = np.concatenate(state_parts)
            
            trajectories.append({
                'state': state,
                'continuous_weights': continuous_weights,
                'binary_actions': binary_actions,
                'variant': variant,
                'fraction': fraction
            })
        
        # Phase 3: Combination (30%)
        n_combo = num_trajectories - len(trajectories)
        combo_trajs = self.goal_expert.generate_trajectories(dataset, n_combo, args)
        trajectories.extend(combo_trajs)
        
        return trajectories


def generate_expert_trajectories_kelly_goal(args, dataset, num_trajectories=100,
                                            method='hybrid'):
    """
    Generate expert trajectories using Kelly + Goal-Conditioned approach
    Returns trajectories with continuous weights and binary actions
    Matches the flattened state format of the original code
    """
    if method == 'goal':
        expert = GoalConditionedExpert(randomize_goals=True)
    elif method == 'kelly':
        expert = HybridKellyGoalExpert()
    else:
        expert = HybridKellyGoalExpert()
    
    # Pass args to handle ind_yn, pos_yn, neg_yn flags
    raw_trajectories = expert.generate_trajectories(dataset, num_trajectories, args)
    
    # Convert to format with both continuous and binary
    # Now state is already flattened 1D vector
    trajectories = []
    for traj in raw_trajectories:
        state = traj['state']  # Already flattened
        continuous_weights = traj['continuous_weights']
        binary_actions = traj['binary_actions']
        trajectories.append((state, binary_actions))  # Match old format: (state, action)
    
    return trajectories


def generate_expert_trajectories(args, dataset, num_trajectories=100):
    """
    Wrapper function that matches the old interface exactly
    Uses Kelly + Goal-Conditioned expert internally
    """
    return generate_expert_trajectories_kelly_goal(args, dataset, num_trajectories, method='hybrid')


def evaluate_expert_performance(trajectories, dataset, risk_free_rate=0.02):
    """
    Evaluate performance metrics (Sharpe ratio, returns, volatility, drawdown)
    from hybrid expert trajectories.
    """
    portfolio_returns = []
    all_weights = []
    all_dates = []

    for i, (state, actions) in enumerate(trajectories):
        data = dataset[i % len(dataset)]
        returns = data["labels"].numpy()
        weights = actions / (np.sum(actions) + 1e-8)

        if returns.ndim == 1:
            daily_portfolio_return = np.dot(returns, weights)
            portfolio_returns.append(daily_portfolio_return)
        elif returns.ndim == 2:
            daily_portfolio_returns = np.dot(returns, weights)
            portfolio_returns.extend(daily_portfolio_returns.tolist())
        else:
            raise ValueError(f"Unexpected returns shape: {returns.shape}")
        
        all_weights.append(weights)
        all_dates.append(getattr(data, "date", f"t{i}"))

    portfolio_returns = np.array(portfolio_returns)
    portfolio_returns = np.nan_to_num(portfolio_returns, nan=0.0)

    mean_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    sharpe_ratio = (mean_return - risk_free_rate / 252) / (volatility + 1e-8)
    cumulative_return = np.prod(1 + portfolio_returns) - 1

    cum_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdowns)

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = np.sqrt(np.mean(np.square(downside_returns))) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return - risk_free_rate / 252) / (downside_vol + 1e-8)

    benchmark_returns = np.zeros_like(portfolio_returns)
    tracking_error = np.std(portfolio_returns - benchmark_returns)
    information_ratio = (mean_return - np.mean(benchmark_returns)) / (tracking_error + 1e-8)

    metrics = {
        "Mean Daily Return": mean_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Information Ratio": information_ratio,
        "Cumulative Return": cumulative_return,
        "Max Drawdown": max_drawdown,
    }

    print("\n===== Expert Performance Metrics =====")
    for k, v in metrics.items():
        print(f"{k:<25}: {v:.4f}")

    return metrics, portfolio_returns


def save_expert_trajectories(trajectories, save_path):
    """Save expert trajectories to file"""
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)


def load_expert_trajectories(load_path):
    """Load expert trajectories from file"""
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories