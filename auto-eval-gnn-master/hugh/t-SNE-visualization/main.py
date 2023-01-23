import torch
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FEATURE_PATH = "./feat_cora_0.5_best_ntk_score_15.pt"
LABEL_PATH = './label_cora_0.5_best_ntk_score_15.pt'


def main():
    labels = torch.load(LABEL_PATH, map_location=torch.device('cpu'))
    features = torch.load(FEATURE_PATH, map_location=torch.device('cpu'))
    tsne = TSNE(n_components=2, early_exaggeration=30, perplexity=30, learning_rate=10, n_iter=10000, init='pca', verbose=1, random_state=501)
    z = tsne.fit_transform(features)

    df = pd.DataFrame()
    df["y"] = labels
    df['comp-1'] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x='comp-1', y='comp-2', hue=df.y.tolist(), palette=sns.color_palette("hls", 7), data=df).set(title="t-SNE embedding")

    plt.show()


if __name__ == "__main__":
    main()
