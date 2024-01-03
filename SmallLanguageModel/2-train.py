#%% [markdown]
# # Create a Small Language Model

#%%
import torch
from torch.utils.data import DataLoader
from time import time
from datasets import DatasetDict
from transformers import AutoTokenizer
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as graph
from model import SmallLM
# torch.set_default_dtype(torch.float16)  # TODO: fix this to save GPU memory and double the batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_name = 'compacted'  # 'compacted' or 'processed'
n_epochs = 200

# load required files
dataset = DatasetDict.load_from_disk(f'data/{dataset_name}-dataset')
tokenizer = AutoTokenizer.from_pretrained('data/tokenizer')
context_size = tokenizer.model_max_length

print(dataset)
print(tokenizer)

slm = SmallLM(tokenizer).to(device)
print(slm)

# %%
# Run the data through the model.forward() method to check 
start_time = time()
output = slm.forward(torch.tensor(dataset['train']['x'][:16]).to(device))
print('=== Untrained Logits ===')
print(output)
print(output.shape)
print(output.dtype)
print(f'=== Time taken: {time() - start_time:.2f}s ===')

# %%
#################################################################################################
# TRAIN THAT LANGUAGE MODEL
#################################################################################################
hist = []
best_val_loss, best_epoch = float('inf'), None
optimizer = torch.optim.AdamW(
    slm.parameters(),
    lr=1e-5,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    
)  # bayesian normal prior on the weights

#%%
# Train loop
train_dataloader = DataLoader(dataset['train'], batch_size=64, shuffle=True)
valid_dataloader = DataLoader(dataset['val'], batch_size=64, shuffle=True)

max_n_minibatches = len(train_dataloader)
if dataset_name == 'compacted':
    max_n_minibatches = min(len(valid_dataloader), max_n_minibatches)

iterator = trange(n_epochs)
for epoch in iterator:
    loss, val_loss = 0.0, 0.0
    subiterator = tqdm(enumerate(train_dataloader), total=max_n_minibatches)
    for i, batch in subiterator:
        # Move to GPU
        x_train = batch['x'].to(device)
        y_train = batch['y'].to(device)

        # Forward pass
        optimizer.zero_grad()
        _, loss_i = slm.forward(x_train, y_train)
        
        # Backward pass
        loss_i.backward()
        torch.nn.utils.clip_grad_norm_(slm.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        loss += loss_i.item()
        subiterator.set_description(f'Minibatch {i + 1:,} | Loss: {loss_i.item():.4f}') 
        if i >= max_n_minibatches:
            break
    loss /= max_n_minibatches

    # Calculate validation loss
    with torch.inference_mode():
        for batch in tqdm(valid_dataloader, total=len(valid_dataloader), desc='Calculating Validation Loss'):
            x_valid = batch['x'].to(device)
            y_valid = batch['y'].to(device)
            _, val_loss_i = slm.forward(x_valid, y_valid)
            val_loss += val_loss_i.item()
    val_loss /= len(valid_dataloader)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss, best_epoch = val_loss, epoch
        # torch.save(slm.state_dict(), f'data/model/model-{dataset_name}.pt')

    # Track metrics
    hist.append([loss, val_loss])
    iterator.set_description(f'Epoch {epoch+1:,} ({best_epoch+1}) | Loss: {loss:.2f} | Val Loss: {val_loss:.4f} ({best_val_loss:.4f})')
print('=== Training Complete ===')

graph.plot([x[0] for x in hist], label='Training Loss')
graph.plot([x[1] for x in hist], label='Validation Loss')
graph.legend()
graph.xlabel('Epoch')
graph.ylabel('Loss')
graph.show()

# %%
