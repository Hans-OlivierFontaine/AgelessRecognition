import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_exec(m_embeddings, m_labels, m_epoch, logger):
    tsne = TSNE(n_components=2, random_state=42)
    # Reduce dimensionality using t-SNE
    m_embeddings = np.squeeze(m_embeddings, axis=1)
    m_labels = np.squeeze(m_labels, axis=1)
    embeddings_2d = tsne.fit_transform(m_embeddings)

    # Plot 2D embeddings
    plt.figure(figsize=(10, 8))
    for label in np.unique(m_labels):
        embeddings_to_plot = embeddings_2d[m_labels == label]
        plt.scatter(embeddings_to_plot[:, 0], embeddings_to_plot[:, 1], label=label)
    plt.title("t-SNE Visualization of Embeddings")
    plt.legend()
    plt.savefig("./tmp.png")
    logger.log_image("./tmp.png", f"tsne-repr_{m_epoch}.png")