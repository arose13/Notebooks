#%%
# # 3. Generate Outputs!

#%%
import warnings
import torch
from transformers import AutoTokenizer
from model import SmallLM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

# Hyperparameters
dataset_name = 'compacted'  # 'compacted' or 'processed'

tokenizer = AutoTokenizer.from_pretrained('data/tokenizer')
context_size = tokenizer.model_max_length
print(tokenizer)

# load pytorch model from a saved state_dict
slm = SmallLM(tokenizer).to(device)
slm.load_state_dict(torch.load(f'data/model/model-{dataset_name}.pt'))
print(slm)

#%%
# Generate some text!
print('Sample')
_ = slm.generate('Before we proceed any further,')

# %%
