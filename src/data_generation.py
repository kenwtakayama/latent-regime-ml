import numpy as np
import yaml

def load_config(difficulty):
    """
    Load configuration from a YAML file based on difficulty level.
    
    :param difficulty: Configuration difficulty level (e.g., 'easy', 'hard')
    :return: difficulty level, configuration dictionary
    """
    
    difficulty = difficulty.lower()

    with open(f"../configs/{difficulty}.yaml", "r") as f:
        config = yaml.safe_load(f)
    return difficulty, config

def generate_data(config, random_state=42):
    """
    Generate synthetic data with latent regimes.

    :param mu: Mean vectors for each regime
    :param Sigma: Covariance matrices for each regime
    :param random_state: Random state for reproducibility
    :return: Generated data X, true regime labels z_true, and data dimensionality d
    """

    np.random.seed(random_state)
    
    mu = config['mu']
    Sigma = config['Sigma']
    regime_lengths = config['regime_lengths']
    cycle = config['cycle']

    X = []
    regimes = []
    d = 4

    for r, L in zip(cycle, regime_lengths):
        regimes.extend([r] * L)

        samples = np.random.multivariate_normal(
            mean = mu[r],
            cov = Sigma[r],
            size = L
        )
        X.append(samples)

    z_true = np.array(regimes)
    X = np.vstack(X)

    return X, z_true, d