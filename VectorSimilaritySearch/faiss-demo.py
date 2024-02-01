#%%
"""
NOTE: I installed faiss-cpu using `conda install -c conda-forge faiss-cpu`
"""
import pandas as pd
from torch.nn import functional as pf
import faiss
from timeit import default_timer
from mockdata import mock_vectors
from itertools import product
from rosey_graph.notebook import auto_display


embedding_size = [256, 1024]
n_users_array = [int(1e3), int(100e3), int(3e6)]
n_items_array = [100, 1000, int(10e3)]
results = []
for dim, n_users, n_items in product(embedding_size, n_users_array, n_items_array):
    print(f"dim={dim:,} | n_users={n_users:,} | n_items={n_items:,}")
    user_vectors, item_vectors = mock_vectors(n_users, n_items, dim)

    # l2 norm of all vectors
    user_vectors = pf.normalize(user_vectors, p=2, dim=1)
    item_vectors = pf.normalize(item_vectors, p=2, dim=1)

    start = default_timer()
    # index = faiss.IndexFlatL2(dim)  # Exact NN Search
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatL2(dim),
        dim,
        20,  # n centeroids!gs
        faiss.METRIC_INNER_PRODUCT
    )
    index.train(item_vectors)
    index.add(item_vectors)
    indexing_dura = default_timer() - start
    print(f"Indexing duration: {indexing_dura:,.4f} seconds")

    start = default_timer()
    distances, indices = index.search(user_vectors, 10)
    search_dura = default_timer() - start
    print(f"Search duration: {search_dura:,.4f} seconds")
    print('-'*60)

    results.append({
        'dim': dim,
        'n_users': n_users,
        'n_items': n_items,
        'indexing_dura': indexing_dura,
        'search_dura': search_dura
    })
results = pd.DataFrame(results)
results.to_csv('faiss-ann-benchmark.csv', index=False)
auto_display(results)
