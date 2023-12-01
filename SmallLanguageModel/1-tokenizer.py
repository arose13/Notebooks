#%%
import torch
import itertools
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from tqdm.auto import tqdm, trange
from collections import Counter

import matplotlib.pyplot as graph
from rosey_graph import plot_barplot

model_max_length = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility functions
def batch_feeder(data, batch_size=1):
    iterable_training_dataset = iter(data['train'])

    for _ in trange(0, len(data['train']), batch_size):
        yield [
            next(iterable_training_dataset)['text']
            for _ in range(batch_size)
        ]


#%%
# Read raw.txt and split by the newline character
with open('data/raw.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read().split('\n\n')  # ✅
print(f'Number of lines: {len(raw_text)}')
print(raw_text[:5])

#%%
# Create a DatasetDict of Datasets that have train and validation splits
# 90% Train, 10% Validation
n = len(raw_text)
train = raw_text[:int(0.9*n)]
val = raw_text[int(0.9*n):]
dataset = DatasetDict({
    'train': Dataset.from_dict({'text': train}),
    'val': Dataset.from_dict({'text': val})
})
print(dataset)

# %%
# Train custom tokenizer based on GPT2
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
tokenizer = tokenizer.train_new_from_iterator(
    batch_feeder(dataset),
    vocab_size=2048,
    initial_alphabet=bytes_to_unicode(),
)
tokenizer.model_max_length = model_max_length
tokenizer.save_pretrained('data/tokenizer')

# Let's see what we've trained
print(tokenizer)
print(f'=== Vocab size: {tokenizer.vocab_size:,} ===')
tokens = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
print('\n== Most common tokens ==')
print(tokens[:25])
print('\n== Least common tokens ==')
print(tokens[-25:])
print('\n== Random sample ==')
print(tokenizer('Hello world!').tokens())
print(tokenizer('Before we proceed any further, hear me speak.').tokens())

#%%
# Functions for creating the dataset
def tokenize(batch):
    # NOTE (2023/11/29) this batch MUST be of size 1
    return {'tokens': tokenizer(batch['text'])['input_ids'] + [tokenizer.eos_token_id]}

def tokenize_and_shift(batch):
    text = batch['text']
    x_tokens = tokenizer(text)['input_ids']
    y_tokens = x_tokens[1:] + [tokenizer.eos_token_id]

    # Pad with pad_token_id until max length
    x_tokens = x_tokens + [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(x_tokens))
    y_tokens = y_tokens + [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(y_tokens))

    return {'x': x_tokens, 'y': y_tokens}

def token_list_to_x_y(tokens: list, split=''):
    x = torch.stack([
        torch.tensor(tokens[i:i+tokenizer.model_max_length])
        for i in trange(len(tokens) - tokenizer.model_max_length, desc=f'Generating X {split}')
    ])
    y = torch.stack([
        torch.tensor(tokens[i+1:i+tokenizer.model_max_length+1])
        for i in trange(len(tokens) - tokenizer.model_max_length, desc=f'Generating Y {split}')
    ])
    return {'x': x, 'y': y}


#%%
# Processed dataset
dataset = dataset.map(tokenize, batch_size=1)
dataset = dataset.map(tokenize_and_shift, batch_size=1)
dataset.set_format(type='pt', columns=['x', 'y'])
print(dataset)
dataset.save_to_disk('data/processed-dataset')

#%% 
# Compact Dataset
compacted = {
    split: list(itertools.chain(*dataset[split]['tokens']))
    for split in dataset.keys()
}
compacted = DatasetDict({
    split: Dataset.from_dict({
        **token_list_to_x_y(compacted[split], split)
    })
    for split in dataset.keys()
})
compacted.set_format(type='pt', columns=['x', 'y'])
print(compacted)
compacted.save_to_disk('data/compacted-dataset')
print('=== Datasets saved! ===')

# %%
print('Generating tokens per word distribution...')
def n_tokens_from_word(tokenizer, word):
    return len(tokenizer(word).tokens())

def dataset_to_words(dataset) -> list:
    for i in range(len(dataset)):
        yield dataset['text'][i].split(' ')

def compute_tokens_per_word_dist(tokens_per_word):
    tokens_per_word = Counter(tokens_per_word)
    tokens_per_word = {k:v for k, v in sorted(tokens_per_word.items())}
    return tokens_per_word

tokens_per_word = []
for word in tqdm(itertools.chain(*dataset_to_words(dataset['train']))):
    tokens_per_word.append(n_tokens_from_word(tokenizer, word))
print(f'Average number of tokens per word: {sum(tokens_per_word) / len(tokens_per_word):.2f}')

graph.title(f'Tokens per word distribution (N = {len(tokens_per_word):,})')
plot_barplot(compute_tokens_per_word_dist(tokens_per_word))
graph.xlabel('Number of words')
graph.show()
