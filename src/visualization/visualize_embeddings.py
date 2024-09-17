from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_tsne(embeddings_loader):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    