import matplotlib.pyplot as plt

def plot_PCA(X_pca, z_true, name, notebook_num, difficulty, cmap='viridis'):
    """
    Plot PCA visualization of regimes

    :param X_pca: PCA-transformed data
    :param z_true: True regime assignments
    :param name: Name of the plot
    :param notebook_num: Notebook number for saving the figure
    :param difficulty: Config level for saving the figure
    :param cmap: Colormap for the scatter plot
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(X_pca[:,0], X_pca[:, 1], c = z_true, s = 5, cmap=cmap)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(name)
    plt.savefig(f"../figures/0{notebook_num}_{name.replace(" ", "_")}_{difficulty.capitalize()}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_series(z_test, z_true, z_test_name, notebook_num, difficulty):
    """
    Plot time series of regime assignments

    :param z_test: Predicted regime assignments
    :param z_true: True regime assignments
    :param z_test_name: Name of the test model
    :param notebook_num: Notebook number for saving the figure
    :param difficulty: Config level for saving the figure
    """
    
    plt.figure(figsize=(10, 4))
    plt.plot(z_true, label="True Regime", alpha=0.9)
    plt.plot(z_test, label= f'{z_test_name} Assignment', alpha=0.5)
    plt.title(f'{z_test_name} vs True Regime')
    plt.legend()
    plt.savefig(f"../figures/0{notebook_num}_{z_test_name.replace(' ', '_')}_{difficulty.capitalize()}.png", dpi=300, bbox_inches='tight')
    plt.show()