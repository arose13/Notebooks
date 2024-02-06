import torch
import numpy as np
from tqdm.auto import tqdm
from itertools import product
from faiss.contrib.datasets import SyntheticDataset


embedding_size = [256, 1024]
n_users_array = [int(1e3), int(100e3)]#, int(3e6)]
n_items_array = [100, 1000, int(10e3)]
def experiment_generator():
    output = product(embedding_size, n_users_array, n_items_array)
    output = [(256, 1000, int(1e6)), (1024, 1000, int(1e6))] + list(output)
    for dim, n_user, n_items in output:
        if dim >= 1024 and n_items >= int(1e6):
            continue
        yield dim, n_user, n_items


def mock_vectors(n_user, n_items, dim):
    try:
        user_vectors = torch.load(f'__cache__/user_vectors_{dim}_{n_user}.pt')
        item_vectors = torch.load(f'__cache__/item_vectors_{dim}_{n_items}.pt')
        return user_vectors, item_vectors
    except FileNotFoundError:
        pass

    user_vectors = torch.rand(n_user, dim)
    item_vectors = torch.rand(n_items, dim)
    return user_vectors, item_vectors


def better_mock_vectors(n_user, n_items, dim):
    try:
        user_vectors = torch.load(f'__cache__/user_fbvectors_{dim}_{n_user}.pt')
        item_vectors = torch.load(f'__cache__/item_fbvectors_{dim}_{n_items}.pt')
        return user_vectors, item_vectors
    except FileNotFoundError:
        pass

    dataset = SyntheticDataset(dim, 0, n_items, n_user, 'IP', seed=42)
    user_vectors = torch.from_numpy(dataset.get_queries())
    item_vectors = torch.from_numpy(dataset.get_database())
    np.save(f'__cache__/gt-fb-matches_{dim}_{n_user}_{n_items}.npy', dataset.get_groundtruth(k=1))
    return user_vectors, item_vectors


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def calc_recall_stats(ground_truth, matches):
    ground_truth = ground_truth[:, 0].reshape((len(ground_truth), 1))
    recall_table = (ground_truth == matches).cumsum(axis=1)

    return {
        'top1_recall': recall_table[:, 0].mean(),
        'top3_recall': recall_table[:, 2].mean(),
        'top5_recall': recall_table[:, 4].mean(),
        'top10_recall': recall_table[:, 9].mean()
    }


if __name__ == '__main__':
    for dim, n_user, n_items in tqdm(experiment_generator()):
        user_vectors, item_vectors = better_mock_vectors(n_user, n_items, dim)
        # user_vectors, item_vectors = mock_vectors(n_user, n_items, dim)
        torch.save(user_vectors, f'__cache__/user_vectors_{dim}_{n_user}.pt')
        torch.save(item_vectors, f'__cache__/item_vectors_{dim}_{n_items}.pt')
