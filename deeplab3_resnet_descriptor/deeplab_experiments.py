
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm



experiment_dataset='../experiment-dataset'
range_of_k = range(50, 550, 50)



def get_precision_at_k(query_results_dataframe, query_class, K, number_relevant):
    top_K = query_results_dataframe.sort_values('similarity', ascending=False).iloc[0:K, :]
    match1 = np.flatnonzero(top_K.label == query_class[0])
    #match2 = np.flatnonzero(top_K.label2 == query_class[0])


    _precision = len(match1) / K
    _recall = len(match1) /number_relevant
    _fmeasure =0
    try:
        _fmeasure = (2 * _precision * _recall)/(_precision + _recall)
    except:
            pass
    return [_precision,_recall,_fmeasure]


def generate_image(_dict, title, name):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 6))
    for k, v in _dict.items():
        ax1.plot(range_of_k, np.array(v)[:,0], lw=2, label=k)
        ax2.plot(range_of_k, np.array(v)[:,1], lw=2, label=k)

    ax1.set_xlabel("K")
    ax1.set_ylabel("precision")
    ax1.legend(loc="best")
    ax1.set_title(name,"precision at different K")

    ax2.set_xlabel("K")
    ax2.set_ylabel("recall")
    ax2.legend(loc="best")
    ax2.set_title("recall at different K")
    ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0])

    plt.savefig(title)
    plt.show()


def generate_image2(_dict, title):
    measures=['precision','recall','F1-Measure']
    plt.figure(figsize=(10, 5))

    for index, m in enumerate(measures):
        for k, v in _dict.items():
            plt.plot(range_of_k, np.array(v)[:, index], lw=2, label=k)
        plt.xlabel('K')
        plt.ylabel(m)
        plt.legend(loc="best")
        plt.title(title + ' dataset ' + m + " at different Top K values with DeepLab3")
        plt.savefig(title + ' ' + m)
        plt.show()


def get_similarity(descriptor1, descriptor2):
    cos_sim = np.dot(descriptor1[1:20], descriptor2[1:20]) / \
              (norm(descriptor1[1:20]) * norm(descriptor2[1:20]))
    return cos_sim


if __name__ == "__main__":

    df_deeplab = pd.read_pickle('deeplab_pascal_all.pkl')

    df_deeplab['label'] = df_deeplab.classes.apply(lambda x: x[0])
    df_deeplab['label2'] = df_deeplab.classes.apply(lambda x: -1 if len(x) == 1 else x[1])

    data_subset = []

    for i in range(20):
        data_subset.extend(df_deeplab[df_deeplab['label'] == i].head(50).file_name.values)

    # get one image from each class

    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    # Gotcha: the indexing of the labels in the dataset a off by one compared to the list. so a label of 1 means bicycle

    query_set = []
    for index, label in enumerate(LABEL_NAMES):

        if index == 0:
            continue
        df_maybe = df_deeplab[df_deeplab['label'] == (index-1)]
        one_class = df_maybe[df_maybe['label2'] == -1].iloc[1, :].to_dict()

        query_set.append(one_class)

    deeplab_precision_recall = {}
    df_deeplab_subset = df_deeplab[df_deeplab['file_name'].isin(data_subset)]

    for i, v in enumerate(query_set):
        q_i = df_deeplab[df_deeplab['file_name'] == v['file_name']]
        df_deeplab_subset['similarity'] = df_deeplab_subset.descriptor.apply(get_similarity, args=[q_i.descriptor.values[0]])

        relevant = df_deeplab_subset[df_deeplab_subset['label'] == q_i.label.values[0]].shape[0]

        dict_key = LABEL_NAMES[v['label']+1]+' '+v['file_name'];
        deeplab_precision_recall[dict_key] = []
        for ii in range_of_k:
            deeplab_precision_recall[dict_key].append(get_precision_at_k(df_deeplab_subset, q_i.classes.values[0], ii,
                                                                               relevant))

    generate_image2(deeplab_precision_recall, 'pascal ')