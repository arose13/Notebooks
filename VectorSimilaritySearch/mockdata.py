import torch


def mock_vectors(n_user, n_items, dim):
    # TODO (2024/01/31) Replace with non-random vectors
    user_vectors = torch.rand(n_user, dim)
    item_vectors = torch.rand(n_items, dim)
    return user_vectors, item_vectors

if __name__ == '__main__':
    mock_vectors(100, 1000, 256)
