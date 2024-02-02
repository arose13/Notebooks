import torch
from itertools import product


embedding_size = [256, 1024]
n_users_array = [int(1e3), int(100e3), int(3e6)]
n_items_array = [100, 1000, int(10e3)]
def experiment_generator():
    return product(embedding_size, n_users_array, n_items_array)


def mock_vectors(n_user, n_items, dim):
    # TODO (2024/01/31) Replace with non-random vectors
    user_vectors = torch.rand(n_user, dim)
    item_vectors = torch.rand(n_items, dim)
    return user_vectors, item_vectors


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == '__main__':
    mock_vectors(100, 1000, 256)
