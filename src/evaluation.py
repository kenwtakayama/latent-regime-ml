import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, accuracy_score

def calculate_ari(z_true, z_test, z_test_name):
    """
    Calculate Adjusted Rand Index (ARI) between true and predicted labels.

    :param z_true: True labels
    :param z_test: Predicted labels
    :param z_test_name: Name of the model being evaluated
    :return: ARI score
    """

    ari = adjusted_rand_score(z_true, z_test)
    print(f'{z_test_name} ARI: {ari:.3f}')
    return ari

def calculate_accuracy(z_true, z_test, z_test_name):
    """
    Calculate accuracy between true and predicted labels.

    :param z_true: True labels
    :param z_test: Predicted labels
    :param z_test_name: Name of the model being evaluated
    :return: Accuracy score
    """

    acc = accuracy_score(z_true, z_test)
    print(f'{z_test_name} Accuracy: {acc:.3f}')
    return acc

def transition_matrix(transmat, model_name, notebook_num, difficulty):
    """
    Plot the transition matrix for a Hidden Markov Model.

    :param transmat: Transition matrix
    :param model_name: Name of the model being evaluated
    """

    sns.heatmap(transmat, annot= True, fmt= '.3f')
    plt.title(f'{model_name} Transition Matrix')
    plt.savefig(f"../figures/0{notebook_num}_{model_name.replace(' ', '_')}_Transition_Matrix_{difficulty.capitalize()}.png", dpi=300, bbox_inches='tight')
    plt.show

def describe_PCA(pca):
    """
    Describe the PCA results.

    :param pca: PCA object
    """

    print(f'Cumulative Explained Variance: {pca.explained_variance_ratio_.sum():.2%}')
    print(f'Explained Variance Ratio: {np.round(pca.explained_variance_ratio_, decimals=3)}')