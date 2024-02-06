#%%
import warnings
import pandas as pd
import matplotlib.pyplot as graph 
import seaborn as sns
from IPython.display import display, HTML
graph.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


label_path_combos = [
    ('Exact', 'faiss-exact'),
    ('FAISS', 'faiss-ann'),
    ('Voyager', 'voyager-ann')
]
results = []
for label, path in label_path_combos:
    df = pd.read_csv(f'results/{path}-fb-benchmark.csv')
    df['label'] = label
    results.append(df)
results = pd.concat(results).reset_index(drop=True)
results['total_dura'] = results['indexing_dura'] + results['search_dura']
display(HTML(results.to_html(index=False)))


#%%
for is_log in [False, True]:
    print(f'=== {"Log" if is_log else "Linear"} Scale ===')
    for title, subresults in results.groupby('n_items'):
        plot = sns.relplot(
            subresults,
            x='n_users', y='total_dura',
            hue='label',
            col='dim',
            kind='line'
        )
        plot.fig.suptitle(f'N Creatives = {title:,}')
        if is_log:
            plot.set(yscale='log')
        plot.set(xscale='log')
        plot.set_axis_labels('N Queries', 'Total Duration (sec)')
        graph.show()

#%%
#################################################################################################
# RUNTIME PLOTS
#################################################################################################
# make a barplot of the total_dura for each label where n_items = 1000000
graph.figure(figsize=(8, 4))
subresults = results.query('n_items == 1000000')
sns.barplot(
    data=subresults,
    x='n_users', y='total_dura',
    hue='label'
)
graph.title('N Creatives = 1,000,000')
graph.ylabel('Total Duration (sec)')
graph.xlabel('N Queries')
graph.show()

#%%
for title, subresults in results.groupby('dim'):
    subresults = subresults.query('n_users == 1000')
    sns.lineplot(
        data=subresults,
        x='n_items', y='indexing_dura',
        hue='label'
    )
    graph.title(f'dim = {title:,}')
    graph.ylabel('Indexing Duration (sec)')
    graph.xlabel('N Creatives')
    graph.show()

    sns.lineplot(
        data=subresults,
        x='n_items', y='search_dura',
        hue='label'
    )
    graph.title(f'dim = {title:,}')
    graph.ylabel('Search Duration (sec)')
    graph.xlabel('N Creatives')
    graph.show()

# %%
#################################################################################################
# ACCURACY PLOTS
#################################################################################################
for topk_recall in [col for col in results.columns if col.endswith('_recall')]:
    sns.lineplot(
        data=results,
        x='n_items', y=topk_recall,
        hue='label'
    )
    graph.ylabel(topk_recall.replace('_', ' ').title())
    graph.xlabel('N Creatives')
    graph.xscale('log')
    graph.show()

