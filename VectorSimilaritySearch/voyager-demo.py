#%%
"""
Spotify's Voyager Demo
package installed easier that faiss with
`pip install voyager`
"""
import numpy as np
import pandas as pd
import torch.nn.functional as pf
from pprint import pprint
from tqdm.auto import tqdm
from voyager import Index, Space
from timeit import default_timer
from mockdata import mock_vectors, better_mock_vectors, experiment_generator, batched, calc_recall_stats
from rosey_graph.notebook import auto_display


results = []
for dim, n_users, n_items in experiment_generator():
    print(f"dim={dim:,} | n_users={n_users:,} | n_items={n_items:,}")
    # If using basic mock data
    # user_vectors, item_vectors = mock_vectors(n_users, n_items, dim)
    # user_vectors = pf.normalize(user_vectors, p=2, dim=1).numpy()
    # item_vectors = pf.normalize(item_vectors, p=2, dim=1).numpy()
    
    user_vectors, item_vectors = better_mock_vectors(n_users, n_items, dim)

    # Indexing
    start = default_timer()
    # index = Index(Space.Cosine, num_dimensions=dim)
    index = Index(
        Space.InnerProduct,
        num_dimensions=dim,
        M=24,
        ef_construction=40,
    )
    if n_items <= 10_000:
        index.add_items(item_vectors)
    else:
        batch_size = 10_000
        for batch_item_vectors in tqdm(
            batched(item_vectors, batch_size),
            total=len(item_vectors)//batch_size
        ):
            index.add_items(batch_item_vectors)
    indexing_dura = default_timer() - start
    print(f"Indexing duration: {indexing_dura:,.4f} seconds")

    # Search
    matches = []
    if n_users <= int(100e3):
        start = default_timer()
        matches, _ = index.query(user_vectors, k=10)
        search_dura = default_timer() - start
    else:
        start = default_timer()
        batch_size = 10_000
        for batch in tqdm(
            batched(user_vectors, batch_size),
            total=len(user_vectors)//batch_size
        ):
            matches_i, _ = index.query(batch, k=10)
            matches.append(matches_i)
        matches = np.vstack(matches)
        search_dura = default_timer() - start
    print(f"Search duration: {search_dura:,.4f} seconds")

    results.append({
        'dim': dim,
        'n_users': n_users,
        'n_items': n_items,
        'indexing_dura': indexing_dura,
        'search_dura': search_dura
    })
    # Calculate recall statistics
    gt_matches = np.load(f'__cache__/gt-fb-matches_{dim}_{n_users}_{n_items}.npy')
    recall_states = calc_recall_stats(gt_matches, matches)
    results[-1].update(recall_states)
    print(f'-> Recall stats:')
    pprint(recall_states)
    print('-'*60)
results = pd.DataFrame(results)
results.to_csv('voyager-ann-fb-benchmark.csv', index=False)
auto_display(results)
