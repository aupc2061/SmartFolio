import numpy as np
import pandas as pd
import pickle
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import risk_models, expected_returns, EfficientFrontier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings


class HybridExpertStrategy:
    """
    Hybrid expert strategy combining multiple approaches:
    1. Black-Litterman optimization (theory-driven)
    2. Momentum & Mean Reversion signals (statistical)
    3. Machine Learning ranking (data-driven)
    4. Risk parity considerations (risk-balanced)
    5. Industry diversification constraints (practical)
    """
    
    def __init__(
        self,
        top_k=0.1,
        max_industry_ratio=0.3,
        bl_weight=0.3,
        momentum_weight=0.25,
        ml_weight=0.25,
        risk_weight=0.2,
        tau=0.05,
        view_confidence=0.5,
        rho=0.7,
    ):
        self.top_k = top_k
        self.max_industry_ratio = max_industry_ratio
        self.bl_weight = bl_weight
        self.momentum_weight = momentum_weight
        self.ml_weight = ml_weight
        self.risk_weight = risk_weight
        self.tau = tau
        self.view_confidence = view_confidence
        self.rho = rho
        
    def _ensure_positive_definite(self, cov, num_stocks):
        """Ensure covariance matrix is positive definite."""
        min_eigenval = np.min(np.linalg.eigvals(cov))
        if min_eigenval < 1e-8:
            cov += (abs(min_eigenval) + 1e-6) * np.eye(num_stocks)
        return cov
    
    def _build_covariance_matrix(self, returns, correlation_matrix, num_stocks):
        """Build robust covariance matrix."""
        if returns.empty or returns.shape[0] < 2 or np.all(np.isnan(returns.values)):
            base_var = 0.01
            cov = self.rho * correlation_matrix * base_var + (1 - self.rho) * np.eye(num_stocks) * base_var
        else:
            cov = np.cov(returns.values.T)
            if not np.all(np.isfinite(cov)):
                base_var = np.nanmean(np.var(returns.values))
                if np.isnan(base_var) or base_var == 0:
                    base_var = 0.01
                cov = self.rho * correlation_matrix * base_var + (1 - self.rho) * np.eye(num_stocks) * base_var
        
        return self._ensure_positive_definite(cov, num_stocks)
    
    def _black_litterman_score(self, returns, correlation_matrix, num_stocks):
        """Generate Black-Litterman optimized scores."""
        try:
            cov = self._build_covariance_matrix(returns, correlation_matrix, num_stocks)
            mean_returns = np.nan_to_num(np.nanmean(returns.values, axis=0), nan=0.0)
            
            # Define views
            view_count = max(1, int(num_stocks * self.top_k))
            top_indices = np.argsort(-mean_returns)[:view_count]
            P = np.zeros((view_count, num_stocks))
            for i, idx in enumerate(top_indices):
                P[i, idx] = 1
            Q = mean_returns[top_indices].reshape(-1, 1)
            Q = np.nan_to_num(Q, nan=0.0)
            omega = np.eye(view_count) * (1 - self.view_confidence + 1e-6)
            
            bl = BlackLittermanModel(
                cov_matrix=cov,
                pi=mean_returns,
                P=P,
                Q=Q,
                tau=self.tau,
                omega=omega,
            )
            
            bl_returns = bl.bl_returns()
            bl_returns = np.nan_to_num(bl_returns, nan=0.0, posinf=1e6, neginf=-1e6)
            
            ef = EfficientFrontier(bl_returns, cov)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= 0.15)
            
            try:
                raw_weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
                weights = np.array(list(cleaned_weights.values()))
            except:
                ef = EfficientFrontier(bl_returns, cov, solver='SCS', solver_options={'max_iters': 10000})
                try:
                    raw_weights = ef.max_sharpe()
                    cleaned_weights = ef.clean_weights()
                    weights = np.array(list(cleaned_weights.values()))
                except:
                    weights = np.maximum(bl_returns, 0)
                    weights = weights / (np.sum(weights) + 1e-8)
            
            return weights
            
        except Exception as e:
            warnings.warn(f"Black-Litterman failed: {e}")
            mean_returns = np.nan_to_num(np.nanmean(returns.values, axis=0), nan=0.0)
            weights = np.maximum(mean_returns, 0)
            return weights / (np.sum(weights) + 1e-8)
    
    def _momentum_score(self, returns, features):
        """Calculate momentum and trend-following scores."""
        scores = np.zeros(returns.shape[1])
        
        for i in range(returns.shape[1]):
            stock_returns = returns.values[:, i]
            stock_returns = np.nan_to_num(stock_returns, nan=0.0)
            
            if len(stock_returns) > 0:
                # Short-term momentum (last 20%)
                short_period = max(1, len(stock_returns) // 5)
                short_momentum = np.mean(stock_returns[-short_period:])
                
                # Long-term momentum (entire period)
                long_momentum = np.mean(stock_returns)
                
                # Trend strength (using linear regression slope)
                if len(stock_returns) > 2:
                    x = np.arange(len(stock_returns))
                    trend = np.polyfit(x, stock_returns, 1)[0]
                else:
                    trend = 0
                
                # Volatility-adjusted momentum
                volatility = np.std(stock_returns) + 1e-8
                
                # Combined score
                scores[i] = (0.4 * short_momentum + 0.3 * long_momentum + 0.3 * trend) / volatility
        
        # Normalize scores
        scores = np.nan_to_num(scores, nan=0.0)
        if np.std(scores) > 0:
            scores = (scores - np.mean(scores)) / np.std(scores)
        
        return scores
    
    def _ml_ranking_score(self, returns, features, correlation_matrix):
        """Use machine learning to rank stocks based on features."""
        try:
            num_stocks = features.shape[0]
            
            # Create labels: top 30% = 1, bottom 30% = 0, middle = skip
            mean_returns = np.nan_to_num(np.nanmean(returns.values, axis=0), nan=0.0)
            top_threshold = np.percentile(mean_returns, 70)
            bottom_threshold = np.percentile(mean_returns, 30)
            
            labels = []
            train_features = []
            
            for i in range(num_stocks):
                if mean_returns[i] >= top_threshold:
                    labels.append(1)
                    train_features.append(features[i])
                elif mean_returns[i] <= bottom_threshold:
                    labels.append(0)
                    train_features.append(features[i])
            
            if len(labels) < 10:
                # Not enough data for ML
                return np.ones(num_stocks) / num_stocks
            
            train_features = np.array(train_features)
            labels = np.array(labels)
            
            # Handle NaN in features
            train_features = np.nan_to_num(train_features, nan=0.0)
            
            # Train ensemble model
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(train_features, labels)
            
            # Predict probabilities for all stocks
            all_features = np.nan_to_num(features, nan=0.0)
            scores = rf.predict_proba(all_features)[:, 1]
            
            return scores
            
        except Exception as e:
            warnings.warn(f"ML ranking failed: {e}")
            return np.ones(features.shape[0]) / features.shape[0]
    
    def _risk_parity_score(self, returns, correlation_matrix):
        """Calculate risk parity scores to balance portfolio risk."""
        try:
            # Calculate volatility for each stock
            volatilities = np.nan_to_num(np.nanstd(returns.values, axis=0), nan=1.0)
            volatilities = np.where(volatilities == 0, 1.0, volatilities)
            
            # Inverse volatility weights
            inv_vol_weights = 1.0 / volatilities
            
            # Adjust for correlations (penalize highly correlated stocks)
            correlation_penalty = np.mean(np.abs(correlation_matrix), axis=1)
            correlation_penalty = np.nan_to_num(correlation_penalty, nan=1.0)
            
            # Combined risk-adjusted score
            risk_scores = inv_vol_weights / (correlation_penalty + 1e-8)
            
            # Normalize
            risk_scores = risk_scores / (np.sum(risk_scores) + 1e-8)
            
            return risk_scores
            
        except Exception as e:
            warnings.warn(f"Risk parity calculation failed: {e}")
            return np.ones(returns.shape[1]) / returns.shape[1]
    
    def generate_hybrid_score(self, returns, features, correlation_matrix):
        """Combine all scoring methods into a unified hybrid score."""
        if isinstance(returns, np.ndarray):
            returns = np.atleast_2d(returns)
        returns = pd.DataFrame(returns)
        num_stocks = returns.shape[1]
        
        # Get scores from each method
        bl_scores = self._black_litterman_score(returns, correlation_matrix, num_stocks)
        momentum_scores = self._momentum_score(returns, features)
        ml_scores = self._ml_ranking_score(returns, features, correlation_matrix)
        risk_scores = self._risk_parity_score(returns, correlation_matrix)
        
        # Normalize all scores to [0, 1]
        def normalize(scores):
            scores = np.nan_to_num(scores, nan=0.0)
            scores = np.maximum(scores, 0)  # Remove negative scores
            if np.sum(scores) > 0:
                return scores / np.sum(scores)
            return np.ones(len(scores)) / len(scores)
        
        bl_scores = normalize(bl_scores)
        momentum_scores = normalize(momentum_scores)
        ml_scores = normalize(ml_scores)
        risk_scores = normalize(risk_scores)
        
        # Weighted combination
        hybrid_scores = (
            self.bl_weight * bl_scores +
            self.momentum_weight * momentum_scores +
            self.ml_weight * ml_scores +
            self.risk_weight * risk_scores
        )
        
        return hybrid_scores
    
    def generate_actions(
        self,
        returns,
        features,
        correlation_matrix,
        industry_relation_matrix
    ):
        """Generate final expert actions with industry constraints."""
        num_stocks = features.shape[0]
        
        # Get hybrid scores
        hybrid_scores = self.generate_hybrid_score(returns, features, correlation_matrix)
        
        # Apply industry diversification constraints
        top_n = max(1, int(num_stocks * self.top_k))
        candidate_indices = np.argsort(-hybrid_scores).tolist()
        expert_actions = np.zeros(num_stocks, dtype=int)
        selected_stocks = []
        
        while len(selected_stocks) < top_n and candidate_indices:
            idx = candidate_indices.pop(0)
            
            # Check industry concentration
            industry_cluster = np.where(industry_relation_matrix[idx] > 0)[0].tolist()
            industry_cluster.append(idx)
            selected_in_cluster = sum(expert_actions[industry_cluster])
            max_allowed = max(1, int(top_n * self.max_industry_ratio))
            
            if selected_in_cluster >= max_allowed:
                continue
            
            expert_actions[idx] = 1
            selected_stocks.append(idx)
        
        # Ensure minimum selection
        if len(selected_stocks) == 0:
            top_stocks = np.argsort(-hybrid_scores)[:top_n]
            expert_actions[top_stocks] = 1
        
        return expert_actions


def generate_hybrid_expert_trajectories(args, dataset, num_trajectories=100):
    """
    Generate expert trajectories using the hybrid strategy.
    """
    expert_trajectories = []
    industry_relation_matrix = load_industry_relation_matrix(args.market)
    
    # Initialize hybrid strategy
    hybrid_strategy = HybridExpertStrategy(
        top_k=getattr(args, 'top_k', 0.1),
        max_industry_ratio=getattr(args, 'max_industry_ratio', 0.3),
        bl_weight=getattr(args, 'bl_weight', 0.3),
        momentum_weight=getattr(args, 'momentum_weight', 0.25),
        ml_weight=getattr(args, 'ml_weight', 0.25),
        risk_weight=getattr(args, 'risk_weight', 0.2),
    )
    
    successful_trajectories = 0
    attempts = 0
    max_attempts = num_trajectories * 3  # Allow some failures
    
    while successful_trajectories < num_trajectories and attempts < max_attempts:
        attempts += 1
        
        try:
            # Random sample from dataset
            idx = np.random.randint(0, len(dataset))
            data = dataset[idx]
            
            # Extract features
            time_series_features = data['ts_features'].numpy()
            features = data['features'].numpy()
            correlation_matrix = data['corr'].numpy()
            ind_matrix = data['industry_matrix'].numpy()
            pos_matrix = data['pos_matrix'].numpy()
            neg_matrix = data['neg_matrix'].numpy()
            returns = data['labels'].numpy()
            
            # Generate hybrid expert actions
            expert_actions = hybrid_strategy.generate_actions(
                returns=returns,
                features=features,
                correlation_matrix=correlation_matrix,
                industry_relation_matrix=industry_relation_matrix
            )
            
            # Create state representation
            state_parts = []
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
            
            state_parts.append(features.flatten())
            state = np.concatenate(state_parts)
            
            expert_trajectories.append((state, expert_actions))
            successful_trajectories += 1
            
            if successful_trajectories % 10 == 0:
                print(f"Generated {successful_trajectories}/{num_trajectories} trajectories...")
            
        except Exception as e:
            warnings.warn(f"Failed trajectory attempt {attempts}: {e}")
            continue
    
    if len(expert_trajectories) == 0:
        raise RuntimeError("Failed to generate any expert trajectories")
    
    print(f"Successfully generated {len(expert_trajectories)} trajectories from {attempts} attempts")
    return expert_trajectories

def evaluate_expert_performance(trajectories, dataset, risk_free_rate=0.02):
    """
    Evaluate performance metrics (Sharpe ratio, returns, volatility, drawdown)
    from hybrid expert trajectories.

    Args:
        trajectories: list of (state, expert_actions)
        dataset: same dataset used for generating trajectories
        risk_free_rate: annualized risk-free rate (e.g., 0.02 = 2%)

    Returns:
        metrics: dict of evaluation metrics
        portfolio_returns: array of portfolio returns
    """

    portfolio_returns = []
    all_weights = []
    all_dates = []

    for i, (state, actions) in enumerate(trajectories):
        data = dataset[i % len(dataset)]
        returns = data["labels"].numpy()  # (T, N) or (N,)
        weights = actions / (np.sum(actions) + 1e-8)  # normalize to sum=1

        # Handle both 1D and 2D returns
        if returns.ndim == 1:
            # Single time step: (N,)
            daily_portfolio_return = np.dot(returns, weights)
            portfolio_returns.append(daily_portfolio_return)
        elif returns.ndim == 2:
            # Multiple time steps: (T, N)
            daily_portfolio_returns = np.dot(returns, weights)  # (T,)
            portfolio_returns.extend(daily_portfolio_returns.tolist())
        else:
            raise ValueError(f"Unexpected returns shape: {returns.shape}")
        
        all_weights.append(weights)
        all_dates.append(getattr(data, "date", f"t{i}"))

    # Convert to numpy array
    portfolio_returns = np.array(portfolio_returns)
    portfolio_returns = np.nan_to_num(portfolio_returns, nan=0.0)

    # ---- Compute Metrics ----
    mean_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    sharpe_ratio = (mean_return - risk_free_rate / 252) / (volatility + 1e-8)

    # Cumulative return
    cumulative_return = np.prod(1 + portfolio_returns) - 1

    # Max drawdown
    cum_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdowns)

    # Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = np.sqrt(np.mean(np.square(downside_returns))) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return - risk_free_rate / 252) / (downside_vol + 1e-8)

    # Information Ratio (if benchmark exists)
    # For now, assume benchmark = 0 return per day
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

def load_industry_relation_matrix(market):
    """Load industry relation matrix."""
    with open(f"dataset_default/data_train_predict_{market}/industry.npy", 'rb') as f:
        industry_relation_matrix = np.load(f)
    return industry_relation_matrix


def save_expert_trajectories(trajectories, save_path):
    """Save expert trajectories to file."""
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)


def load_expert_trajectories(load_path):
    """Load expert trajectories from file."""
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


if __name__ == '__main__':
    class Args:
        market = 'hs300'
        input_dim = 6
        ind_yn = True
        pos_yn = True
        neg_yn = True
        top_k = 0.1
        max_industry_ratio = 0.3
        bl_weight = 0.3
        momentum_weight = 0.25
        ml_weight = 0.25
        risk_weight = 0.2

    args = Args()
    from dataloader.data_loader import AllGraphDataSampler

    # Load dataset
    data_dir = f'../dataset/data_train_predict_{args.market}/1_hy/'
    train_dataset = AllGraphDataSampler(
        base_dir=data_dir,
        date=True,
        train_start_date='2019-01-02',
        train_end_date='2022-12-30',
        mode="train"
    )

    # Generate hybrid expert trajectories
    expert_trajectories = generate_hybrid_expert_trajectories(
        args, train_dataset, num_trajectories=100
    )

    # Save trajectories
    save_path = f'../dataset/hybrid_expert_trajectories_{args.market}.pkl'
    save_expert_trajectories(expert_trajectories, save_path)
    print(f"Hybrid expert trajectories saved to {save_path}")