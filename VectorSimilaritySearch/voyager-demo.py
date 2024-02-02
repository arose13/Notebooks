#%%
"""
Spotify's Voyager Demo
package installed easier that faiss with
`pip install voyager`
"""
import pandas as pd
from tqdm.auto import tqdm
from voyager import Index, Space
from timeit import default_timer
from mockdata import mock_vectors, experiment_generator, batched
from rosey_graph.notebook import auto_display


results = []
for dim, n_users, n_items in experiment_generator():
    print(f"dim={dim:,} | n_users={n_users:,} | n_items={n_items:,}")
    user_vectors, item_vectors = mock_vectors(n_users, n_items, dim)

    # convert to numpy arrays
    user_vectors = user_vectors.numpy()
    item_vectors = item_vectors.numpy()

    # Indexing
    start = default_timer()
    index = Index(Space.Cosine, num_dimensions=dim)
    index.add_items(item_vectors)
    indexing_dura = default_timer() - start
    print(f"Indexing duration: {indexing_dura:,.4f} seconds")

    # Search
    if n_users <= int(100e3):
        start = default_timer()
        index.query(user_vectors, k=10)
        search_dura = default_timer() - start
    else:
        start = default_timer()
        batch_size = 10_000
        for batch in tqdm(
            batched(user_vectors, batch_size),
            total=len(user_vectors)//batch_size
        ):
            index.query(batch, k=10)
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
results.to_csv('voyager-ann-benchmark.csv', index=False)
auto_display(results)
