import yaml
from src.data_generation import generate_data
from src.models import fit_kmeans, fit_gmm, fit_hmm, fit_rf, fit_mlp, fit_xg
from src.evaluation import calculate_ari

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    X, z_true, d = generate_data(config)

    T = len(X)
    pct_split = 0.6
    split = int(pct_split * T)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = z_true[:split], z_true[split:]

    results = {}

    _, z_km  = fit_kmeans(X)
    _, z_gmm = fit_gmm(X)
    _, z_hmm = fit_hmm(X, 2000, 100.0)
    _, z_rf  = fit_rf(X_train, y_train, X_test)
    _, z_mlp = fit_mlp(X_train, y_train, X_test)
    _, z_xg = fit_xg(X_train, y_train, X_test)

    print("ARI scores:")
    results["kmeans"] = calculate_ari(z_true, z_km, "KMeans")
    results["gmm"]    = calculate_ari(z_true, z_gmm, "GMM")
    results["hmm"]    = calculate_ari(z_true, z_hmm, "HMM")
    results["rf"]     = calculate_ari(y_test, z_rf, "RF")
    results["mlp"]    = calculate_ari(y_test, z_mlp, "MLP")
    results["xg"]     = calculate_ari(y_test, z_xg, "XGBoost")

if __name__ == "__main__":
    main("configs/medium.yaml")
