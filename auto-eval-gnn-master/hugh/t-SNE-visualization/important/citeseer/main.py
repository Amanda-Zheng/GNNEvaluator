import math

import torch
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob


def main():
    labelPaths = glob.glob('distill/**/label*best_ntk_score*.pt', recursive=True)
    featPaths = glob.glob('distill/**/feat*best_ntk_score*.pt', recursive=True)
    assert len(labelPaths) == len(featPaths)
    numberOfPaths = len(labelPaths)

    for i in range(numberOfPaths):
        print('------')
        print("Generating the graph with label:{}, feat:{}".format(labelPaths[i], featPaths[i]))
        print('------')
        generate_t_SNE_visualization(labelPaths[i], featPaths[i])


def generate_t_SNE_visualization(labelPath, featPath):
    labels = torch.load(labelPath, map_location=torch.device('cpu'))
    features = torch.load(featPath, map_location=torch.device('cpu'))
    if is_citeseer_1_0(labelPath):
        perplexity = 8  # the value for perplexity cannot be more than the number of samples
    elif is_citeseer_0_5(labelPath):
        perplexity = 5  # the value for perplexity cannot be more than the number of samples
    elif is_citeseer_0_25(labelPath):
        perplexity = 10  # the value for perplexity cannot be more than the number of samples

    tsne = TSNE(n_components=2, early_exaggeration=12, perplexity=perplexity, learning_rate='auto', n_iter=10000, init='pca', verbose=1, random_state=501)
    z = tsne.fit_transform(features)

    df = pd.DataFrame()
    df["y"] = labels
    df['comp-1'] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x='comp-1', y='comp-2', hue=df.y.tolist(), palette=sns.color_palette("hls", 6), data=df).set(title="t-SNE embedding")

    plt.show()


def is_citeseer_0_5(path):
    return path.find('citeseer_0.5') != -1


def is_citeseer_1_0(path):
    return path.find('citeseer_1.0') != -1


def is_citeseer_0_25(path):
    return path.find('citeseer_0.25') != -1


if __name__ == "__main__":
    main()
