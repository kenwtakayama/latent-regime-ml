from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def fit_kmeans(X_pca, random_state=42):
    """
    Fit a KMeans clustering model to the PCA-transformed data.

    :param X_pca: PCA-transformed data
    :param random_state: Random state for reproducibility
    :return: Fitted KMeans model and predicted cluster labels
    """

    kmeans = KMeans(n_clusters = 3, random_state= random_state)
    z_kmeans = kmeans.fit_predict(X_pca)
    return kmeans, z_kmeans

def fit_gmm(X_pca, random_state=42):
    """
    Fit a Gaussian Mixture Model to the PCA-transformed data.
    
    :param X_pca: PCA-transformed data
    :param random_state: Random state for reproducibility
    :return: Fitted GMM model and predicted cluster labels
    """

    gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=random_state)
    z_gmm = gmm.fit_predict(X_pca)
    return gmm, z_gmm

def fit_hmm(X_pca, n_iter, transmat_prior, random_state=42):
    """
    Fit a Hidden Markov Model to the PCA-transformed data.

    :param X_pca: PCA-transformed data
    :param n_iter: Number of iterations for training
    :param transmat_prior: Prior for transition matrix
    :param random_state: Random state for reproducibility
    :return : Fitted HMM model and predicted hidden states
    """

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        transmat_prior=transmat_prior,
    )
    hmm.fit(X_pca)
    z_hmm = hmm.predict(X_pca)
    return hmm, z_hmm

def fit_rf(X_train, y_train, X_test, random_state=42):
    """
    Fit a Random Forest classifier to the training data and predict on the test data.

    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param random_state: Random state for reproducibility
    :return: Fitted Random Forest model and predicted labels
    """

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    return rf, y_pred_rf

def fit_mlp(X_train, y_train, X_test, random_state=42):
    """
    Fit a Multi-Layer Perceptron classifier to the training data and predict on the test data.

    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param random_state: Random state for reproducibility
    :return: Predicted labels from the MLP classifier
    """

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation='relu',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        verbose=True
    )

    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    return mlp, y_pred_mlp

def fit_xg(X_train, y_train, X_test, random_state=42):
    """
    Fit an XGBoost classifier to the training data and predict on the test data.

    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param random_state: Random state for reproducibility
    :return: Predicted labels from the XGBoost classifier
    """
    
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    xgb.fit(X_train, y_train)
    y_pred_xg = xgb.predict(X_test)
    return xgb, y_pred_xg