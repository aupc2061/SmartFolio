"""
Hybrid Ensemble Expert Strategy - Fixed for Continuous Weights
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.covariance import LedoitWolf, OAS
import warnings
warnings.filterwarnings('ignore')


class RobustMarkowitz:
    """Robust Markowitz with regularization and shrinkage"""
    
    def __init__(self, risk_aversion=1.0, regularization=0.01, shrinkage_method='ledoit_wolf'):
        self.risk_aversion = risk_aversion
        self.regularization = regularization
        self.shrinkage_method = shrinkage_method
    
    def estimate_covariance(self, returns):
        """Estimate covariance matrix with shrinkage"""
        if self.shrinkage_method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
        elif self.shrinkage_method == 'oas':
            oas = OAS()
            cov_matrix = oas.fit(returns).covariance_
        else:
            cov_matrix = np.cov(returns.T)
        
        # Add regularization
        cov_matrix += self.regularization * np.eye(len(cov_matrix))
        return cov_matrix
    
    def optimize(self, returns, expected_returns=None, constraints=None):
        """
        Optimize portfolio weights using robust Markowitz
        Returns CONTINUOUS weights, not binary
        """
        n_assets = returns.shape[1] if len(returns.shape) > 1 else len(returns)
        
        # Use historical mean if expected returns not provided
        if expected_returns is None:
            if len(returns.shape) > 1:
                expected_returns = returns.mean(axis=0)
            else:
                expected_returns = returns
        
        # Ensure returns is 2D for covariance estimation
        if len(returns.shape) == 1:
            returns_2d = np.tile(returns, (5, 1))
            returns_2d += np.random.randn(*returns_2d.shape) * 0.01
        else:
            returns_2d = returns
        
        cov_matrix = self.estimate_covariance(returns_2d)
        
        # Set default constraints
        if constraints is None:
            constraints = {}
        max_weight = constraints.get('max_weight', 0.3)
        min_weight = constraints.get('min_weight', 0.01)
        
        # Objective function: minimize -return + risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -portfolio_return + self.risk_aversion * portfolio_variance
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        
        weights = np.array(result.x, dtype=float)
        
        # Clean up small weights safely
        clipped = weights.copy()
        clipped[clipped < min_weight] = 0.0
        s = clipped.sum()
        if s <= 1e-12:
            # Fallback to non-negative normalization or equal weight
            weights = np.maximum(weights, 0)
            denom = weights.sum()
            if denom <= 1e-12:
                weights = np.ones(n_assets) / n_assets
            else:
                weights = weights / denom
        else:
            weights = clipped / s
        
        return weights  # ← CONTINUOUS weights


class BlackLitterman:
    """Black-Litterman model for incorporating views"""
    
    def __init__(self, tau=0.05, risk_aversion=2.5):
        self.tau = tau
        self.risk_aversion = risk_aversion
    
    def generate_views(self, returns, correlation_matrix, industry_matrix, n_views=5):
        """Generate views based on momentum and industry relationships"""
        n_assets = len(returns)
        
        # Generate momentum-based views
        top_performers = np.argsort(-returns)[:n_views]
        
        # Create view matrix P and view vector Q
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, idx in enumerate(top_performers):
            P[i, idx] = 1.0
            Q[i] = returns[idx] * 1.2
        
        return P, Q
    
    def optimize(self, returns, cov_matrix, correlation_matrix, industry_matrix, 
                 market_cap_weights=None, constraints=None):
        """Black-Litterman optimization - returns CONTINUOUS weights"""
        n_assets = len(returns)
        
        if market_cap_weights is None:
            market_cap_weights = np.ones(n_assets) / n_assets
        
        # Regularize covariance to avoid singular matrices
        eps = 1e-4
        cov_matrix = np.array(cov_matrix, dtype=float)
        cov_matrix = cov_matrix + eps * np.eye(n_assets)

        # Implied equilibrium returns
        pi = self.risk_aversion * np.dot(cov_matrix, market_cap_weights)
        
        # Generate views
        P, Q = self.generate_views(returns, correlation_matrix, industry_matrix)
        
        # View uncertainty (regularized)
        omega = np.dot(np.dot(P, self.tau * cov_matrix), P.T)
        if omega.ndim == 0:
            omega = np.array([[omega]])
        omega = omega + eps * np.eye(omega.shape[0])
        
        # Black-Litterman formula with pseudo-inverse fallbacks
        tau_cov = self.tau * cov_matrix
        inv_tau_cov = np.linalg.pinv(tau_cov)
        inv_omega = np.linalg.pinv(omega)
        
        M_inverse = inv_tau_cov + np.dot(np.dot(P.T, inv_omega), P)
        M = np.linalg.pinv(M_inverse)
        mu_bl = np.dot(M, np.dot(inv_tau_cov, pi) + np.dot(np.dot(P.T, inv_omega), Q))
        
        # Use Robust Markowitz to optimize with BL returns
        markowitz = RobustMarkowitz(risk_aversion=self.risk_aversion)
        weights = markowitz.optimize(returns, expected_returns=mu_bl, constraints=constraints)
        
        return weights  # ← CONTINUOUS weights


class RiskParity:
    """Risk Parity / Equal Risk Contribution"""
    
    def __init__(self):
        pass
    
    def optimize(self, returns, constraints=None):
        """Risk Parity optimization - returns CONTINUOUS weights"""
        if len(returns.shape) == 1:
            returns_2d = np.tile(returns, (5, 1))
            returns_2d += np.random.randn(*returns_2d.shape) * 0.01
        else:
            returns_2d = returns
        
        cov_matrix = np.cov(returns_2d.T)
        n_assets = cov_matrix.shape[0]
        
        # Set default constraints
        if constraints is None:
            constraints = {}
        max_weight = constraints.get('max_weight', 0.3)
        min_weight = constraints.get('min_weight', 0.01)
        
        # Objective: minimize sum of squared differences in risk contributions
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / (portfolio_vol + 1e-8)
            return risk_contrib
        
        def objective(weights):
            rc = risk_contribution(weights)
            target_rc = np.ones(n_assets) / n_assets
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        weights = np.array(result.x, dtype=float)
        
        # Safe cleanup for small weights
        clipped = weights.copy()
        clipped[clipped < min_weight] = 0.0
        s = clipped.sum()
        if s <= 1e-12:
            weights = np.maximum(weights, 0)
            denom = weights.sum()
            if denom <= 1e-12:
                weights = np.ones(n_assets) / n_assets
            else:
                weights = weights / denom
        else:
            weights = clipped / s
        
        return weights  # ← CONTINUOUS weights


class HRP:
    """Hierarchical Risk Parity"""
    
    def __init__(self):
        pass
    
    def get_quasi_diag(self, link):
        """Get quasi-diagonal matrix from linkage"""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = pd.Index(range(0, sort_ix.shape[0] * 2, 2))
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()
            sort_ix.index = pd.Index(range(sort_ix.shape[0]))
        
        return sort_ix.tolist()
    
    def get_cluster_var(self, cov, cluster_items):
        """Calculate cluster variance"""
        cov_slice = cov.iloc[cluster_items, cluster_items]
        diag = np.diag(cov_slice)
        diag = np.where(diag <= 1e-12, 1e-12, diag)
        w = 1.0 / diag
        w /= w.sum()
        return np.dot(w, np.dot(cov_slice, w))
    
    def get_rec_bipart(self, cov, sort_ix):
        """Recursive bisection to get weights"""
        w = pd.Series(1.0, index=sort_ix)
        cluster_items = [sort_ix]
        
        while len(cluster_items) > 0:
            cluster_items = [i[j:k] for i in cluster_items 
                           for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                           if len(i) > 1]
            
            for i in range(0, len(cluster_items), 2):
                cluster0 = cluster_items[i]
                cluster1 = cluster_items[i + 1]
                
                var0 = self.get_cluster_var(cov, cluster0)
                var1 = self.get_cluster_var(cov, cluster1)
                
                alpha = 1 - var0 / (var0 + var1)
                
                w[cluster0] *= alpha
                w[cluster1] *= 1 - alpha
        
        return w
    
    def optimize(self, returns, constraints=None):
        """HRP optimization - returns CONTINUOUS weights"""
        if len(returns.shape) == 1:
            returns_2d = np.tile(returns, (5, 1))
            returns_2d += np.random.randn(*returns_2d.shape) * 0.01
        else:
            returns_2d = returns
        
        cov_matrix = np.cov(returns_2d.T)
        corr_matrix = np.corrcoef(returns_2d.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        n_assets = cov_matrix.shape[0]
        
        # Set default constraints
        if constraints is None:
            constraints = {}
        min_weight = constraints.get('min_weight', 0.01)
        
        # Convert to DataFrame for easier manipulation
        cov_df = pd.DataFrame(cov_matrix)
        
        # Hierarchical clustering
        dist = ((1 - corr_matrix) / 2.0)
        dist = np.where(dist < 0, 0, dist) ** 0.5
        link = linkage(dist[np.triu_indices(n_assets, k=1)], method='single')
        
        # Get quasi-diagonal matrix
        sort_ix = self.get_quasi_diag(link)
        
        # Get weights through recursive bisection
        weights = self.get_rec_bipart(cov_df, sort_ix)
        weights = weights.values.astype(float)
        
        # Safe cleanup
        clipped = weights.copy()
        clipped[clipped < min_weight] = 0.0
        s = clipped.sum()
        if s <= 1e-12:
            weights = np.maximum(weights, 0)
            denom = weights.sum()
            if denom <= 1e-12:
                weights = np.ones(n_assets) / n_assets
            else:
                weights = weights / denom
        else:
            weights = clipped / s
        
        return weights  # ← CONTINUOUS weights


class HybridEnsembleExpert:
    """
    Hybrid ensemble of multiple expert strategies
    NOW OUTPUTS CONTINUOUS WEIGHTS (not binary)
    """
    
    def __init__(self, ensemble_weights=None, randomize_params=True):
        """
        Args:
            ensemble_weights: Dict of weights for each expert method
            randomize_params: Whether to randomize parameters for diversity
        """
        if ensemble_weights is None:
            ensemble_weights = {
                'robust_markowitz': 0.3,
                'black_litterman': 0.25,
                'risk_parity': 0.2,
                'hrp': 0.15,
            }
        
        self.ensemble_weights = ensemble_weights
        self.randomize_params = randomize_params
        
        # Normalize weights
        total = sum(self.ensemble_weights.values())
        self.ensemble_weights = {k: v/total for k, v in self.ensemble_weights.items()}
    
    def _get_randomized_params(self, method):
        """Get randomized parameters for diversity"""
        if not self.randomize_params:
            return {}
        
        if method == 'robust_markowitz':
            return {
                'risk_aversion': np.random.uniform(0.5, 3.0),
                'regularization': np.random.uniform(0.001, 0.05),
                'shrinkage_method': np.random.choice(['ledoit_wolf', 'oas'])
            }
        elif method == 'black_litterman':
            return {
                'tau': np.random.uniform(0.01, 0.1),
                'risk_aversion': np.random.uniform(1.0, 4.0)
            }
        else:
            return {}
    
    def generate_expert_action(self, returns, correlation_matrix, industry_matrix,
                              pos_matrix=None, neg_matrix=None, constraints=None):
        """
        Generate expert action using ensemble of methods
        NOW RETURNS CONTINUOUS WEIGHTS (not binary actions)
        
        Returns:
            Continuous weight vector [n_assets] that sums to 1.0
        """
        n_assets = len(returns)
        
        if constraints is None:
            constraints = {
                'max_weight': 0.3,
                'min_weight': 0.01,
            }
        
        # Initialize experts
        experts = {}
        
        # Robust Markowitz
        if 'robust_markowitz' in self.ensemble_weights:
            params = self._get_randomized_params('robust_markowitz')
            # Sanitize params to ensure correct types
            ra = params.get('risk_aversion', 1.0)
            reg = params.get('regularization', 0.01)
            sm = params.get('shrinkage_method', 'ledoit_wolf')
            if sm not in ('ledoit_wolf', 'oas'):
                sm = 'ledoit_wolf'
            experts['robust_markowitz'] = RobustMarkowitz(risk_aversion=float(ra), regularization=float(reg), shrinkage_method=str(sm))
        
        # Black-Litterman
        if 'black_litterman' in self.ensemble_weights:
            params = self._get_randomized_params('black_litterman')
            experts['black_litterman'] = BlackLitterman(**params)
        
        # Risk Parity
        if 'risk_parity' in self.ensemble_weights:
            experts['risk_parity'] = RiskParity()
        
        # HRP
        if 'hrp' in self.ensemble_weights:
            experts['hrp'] = HRP()
        
        # Collect weights from each expert
        expert_weights_map = {}
        
        for method, expert in experts.items():
            try:
                if method == 'black_litterman':
                    # BL needs covariance matrix
                    if len(returns.shape) == 1:
                        returns_2d = np.tile(returns, (5, 1))
                        returns_2d += np.random.randn(*returns_2d.shape) * 0.01
                    else:
                        returns_2d = returns
                    
                    cov_matrix = np.cov(returns_2d.T)
                    weights = expert.optimize(
                        returns, cov_matrix, correlation_matrix, 
                        industry_matrix, constraints=constraints
                    )
                else:
                    weights = expert.optimize(returns, constraints=constraints)
                
                # Sanitize any NaNs/Infs from expert
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                expert_weights_map[method] = weights
            except Exception as e:
                print(f"Warning: {method} failed with error: {e}")
                continue
        
        if len(expert_weights_map) == 0:
            # Fallback: simple return-weighted portfolio
            weights = np.maximum(returns, 0)
            weights = weights / (weights.sum() + 1e-8)
            return weights
        
        # Ensemble: weighted average of continuous weights by method name
        ensemble_weights = np.zeros(n_assets, dtype=float)
        for method, alpha in self.ensemble_weights.items():
            if method in expert_weights_map:
                ensemble_weights += alpha * expert_weights_map[method]
        
        # Normalize to sum to 1
        sum_w = ensemble_weights.sum()
        if sum_w <= 1e-12:
            # Fallback: use positive-return weights
            fallback = np.maximum(returns, 0)
            denom = fallback.sum()
            ensemble_weights = fallback / (denom + 1e-8)
        else:
            ensemble_weights = ensemble_weights / sum_w
        
        # Apply min weight threshold
        min_weight = constraints.get('min_weight', 0.01)
        clipped = ensemble_weights.copy()
        clipped[clipped < min_weight] = 0.0
        s = clipped.sum()
        if s <= 1e-12:
            # Keep original normalized ensemble if clipping would zero everything
            pass
        else:
            ensemble_weights = clipped / s
        
        return ensemble_weights  # ← CONTINUOUS weights that sum to 1.0


def generate_expert_trajectories(args, dataset, num_trajectories=100):
    """
    Generate expert trajectories using hybrid ensemble
    NOW OUTPUTS CONTINUOUS WEIGHTS
    
    Returns:
        List of (state, continuous_weights) tuples
    """
    expert_trajectories = []
    
    print(f"\nGenerating {num_trajectories} Ensemble expert trajectories")
    print(f"Ensemble strategy: Hybrid (Markowitz + BL + RP + HRP)")
    print("="*60)
    
    # Initialize hybrid ensemble expert
    expert = HybridEnsembleExpert(randomize_params=True)
    
    for traj_idx in range(num_trajectories):
        # Randomly select a data point
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]
        
        # Extract features
        features = data['features'].numpy()
        returns = data['labels'].numpy()
        correlation_matrix = data['corr'].numpy()
        industry_matrix = data['industry_matrix'].numpy()
        pos_matrix = data['pos_matrix'].numpy() if 'pos_matrix' in data else None
        neg_matrix = data['neg_matrix'].numpy() if 'neg_matrix' in data else None
        
        n_stocks = len(returns)
        
        # Set constraints
        constraints = {
            'max_weight': 0.3,
            'min_weight': 0.01,
        }
        
        try:
            # Generate expert CONTINUOUS weights (not binary!)
            expert_weights = expert.generate_expert_action(
                returns=returns,
                correlation_matrix=correlation_matrix,
                industry_matrix=industry_matrix,
                pos_matrix=pos_matrix,
                neg_matrix=neg_matrix,
                constraints=constraints
            )
        except Exception as e:
            print(f"Warning: Trajectory {traj_idx} failed: {e}")
            # Fallback: simple return-weighted
            expert_weights = np.maximum(returns, 0)
            expert_weights = expert_weights / (expert_weights.sum() + 1e-8)
        
        # Construct state (flattened)
        state_parts = []
        if args.ind_yn:
            state_parts.append(industry_matrix.flatten())
        else:
            state_parts.append(np.zeros(industry_matrix.size))
        if args.pos_yn and pos_matrix is not None:
            state_parts.append(pos_matrix.flatten())
        else:
            state_parts.append(np.zeros(n_stocks * n_stocks))
        if args.neg_yn and neg_matrix is not None:
            state_parts.append(neg_matrix.flatten())
        else:
            state_parts.append(np.zeros(n_stocks * n_stocks))
        
        state_parts.append(features.flatten())
        state = np.concatenate(state_parts)
        
        expert_trajectories.append((state, expert_weights))  # ← CONTINUOUS weights
        
        if (traj_idx + 1) % 100 == 0:
            print(f"  Generated {traj_idx + 1}/{num_trajectories} trajectories")
    
    print(f"\n✓ Generated {len(expert_trajectories)} trajectories")
    
    # Statistics for continuous weights
    all_weights = [a for _, a in expert_trajectories]
    avg_weight_sum = np.mean([np.sum(w) for w in all_weights])
    avg_nonzero = np.mean([np.sum(w > 0.01) for w in all_weights])
    max_concentration = np.mean([np.max(w) for w in all_weights])
    avg_entropy = np.mean([-np.sum(w * np.log(w + 1e-8)) for w in all_weights])
    
    print(f"  Weight statistics:")
    print(f"    - Average sum: {avg_weight_sum:.4f} (should be ~1.0)")
    print(f"    - Avg stocks with >1% allocation: {avg_nonzero:.1f}")
    print(f"    - Avg max single stock weight: {max_concentration:.1%}")
    print(f"    - Avg portfolio entropy: {avg_entropy:.2f}")
    
    # Show sample weights
    sample_weights = all_weights[0]
    top_5_idx = np.argsort(-sample_weights)[:5]
    print(f"  Sample portfolio (first trajectory):")
    for i, idx in enumerate(top_5_idx):
        print(f"    Stock {idx}: {sample_weights[idx]:.3%}")
    
    return expert_trajectories


def save_expert_trajectories(trajectories, save_path):
    """Save expert trajectories to file"""
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {save_path}")


def load_expert_trajectories(load_path):
    """Load expert trajectories from file"""
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


if __name__ == '__main__':
    print("Testing Hybrid Ensemble Expert Strategy...")
    
    # Create dummy data
    n_stocks = 50
    
    returns = np.random.randn(n_stocks) * 0.01
    correlation_matrix = np.random.rand(n_stocks, n_stocks)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    industry_matrix = (np.random.rand(n_stocks, n_stocks) > 0.7).astype(float)
    
    # Initialize expert
    expert = HybridEnsembleExpert(randomize_params=False)
    
    # Generate CONTINUOUS weights (not binary!)
    weights = expert.generate_expert_action(
        returns=returns,
        correlation_matrix=correlation_matrix,
        industry_matrix=industry_matrix
    )
    
    print(f"\nGenerated CONTINUOUS weights (not binary):")
    print(f"  Shape: {weights.shape}")
    print(f"  Sum: {weights.sum():.6f} (should be 1.0)")
    print(f"  Max weight: {weights.max():.3%}")
    print(f"  Stocks with >1%: {np.sum(weights > 0.01)}")
    print(f"  Top 5 allocations:")
    top_5_idx = np.argsort(-weights)[:5]
    for idx in top_5_idx:
        print(f"    Stock {idx}: {weights[idx]:.3%}")
    
    print("\n✓ Hybrid Ensemble Expert (CONTINUOUS WEIGHTS) implementation complete!")