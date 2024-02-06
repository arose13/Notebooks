#%%
"""
Facebook's FAISS to compute the exact nearest neighbors
Install faiss-cpu using `conda install -c conda-forge faiss-cpu`
"""
import numpy as np
import pandas as pd
import torch.nn.functional as pf
import faiss
from pprint import pprint
from tqdm.auto import tqdm
from timeit import default_timer
from mockdata import mock_vectors, better_mock_vectors, experiment_generator, batched, calc_recall_stats
from rosey_graph.notebook import auto_display


results = []
for dim, n_users, n_items in experiment_generator():
    print(f"dim={dim:,} | n_users={n_users:,} | n_items={n_items:,}")
    # If using basic mock data
    # user_vectors, item_vectors = mock_vectors(n_users, n_items, dim)
    # user_vectors = pf.normalize(user_vectors, p=2, dim=1)
    # item_vectors = pf.normalize(item_vectors, p=2, dim=1)

    # Better mock data
    user_vectors, item_vectors = better_mock_vectors(n_users, n_items, dim)

    start = default_timer()
    index = faiss.IndexFlatIP(dim)  # Exact NN Search
    if n_items <= 10_000:
        index.add(item_vectors)
    else:
        batch_size = 10_000
        for batch_item_vectors in tqdm(
            batched(item_vectors, batch_size),
            total=len(item_vectors)//batch_size
        ):
            index.add(batch_item_vectors)
    indexing_dura = default_timer() - start
    print(f"Indexing duration: {indexing_dura:,.4f} seconds")

    # Search
    matches = []
    if n_users <= int(100e3):
        start = default_timer()
        _, matches = index.search(user_vectors, 10)
        search_dura = default_timer() - start
    else:
        start = default_timer()
        batch_size = 10_000
        for batch_user_vectors in tqdm(
            batched(user_vectors, batch_size),
            total=len(user_vectors)//batch_size
        ):
            _, matches_i = index.search(batch_user_vectors, 10)
            matches.append(matches_i)
        matches = np.vstack(matches)
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
    gt_matches = np.load(f'__cache__/gt-fb-matches_{dim}_{n_users}_{n_items}.npy')
    recall_states = calc_recall_stats(gt_matches, matches)
    results[-1].update(recall_states)
    print(f'-> Recall stats')
    pprint(recall_states)
    print('-'*60)
results = pd.DataFrame(results)
results.to_csv('faiss-exact-fb-benchmark.csv', index=False)
auto_display(results)
