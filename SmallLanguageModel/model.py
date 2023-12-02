import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('data/tokenizer')
context_size = tokenizer.model_max_length


class MultiHeadSelfAttention(nn.Module):
    """
    Implementation of Multi-Head Self Attention from "Attention is all you need"

    https://arxiv.org/pdf/1706.03762.pdf
    Check Section 3.2.2 for more details
    """
    def __init__(self, n_heads, embedding_size):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_size = embedding_size
        self.head_size = embedding_size // n_heads
        assert self.head_size * n_heads == embedding_size, f'Embedding size must be divisible by number of heads'

        # project x into Q, K, V matrices
        self.qkv_reprojector = nn.Linear(embedding_size, 3*embedding_size)
        self.linear_reproject = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        # x.shape = (n, seq_len, embedding_size)
        n, seq_len, embedding_size = x.shape
        if embedding_size != self.embedding_size:
            raise AssertionError(f'Expected embedding size of {self.embedding_size}, got {embedding_size}')

        # Implement the attention mechanism! (Figure 2 in the paper)
        qkv = self.qkv_reprojector(x)  # (n, seq_len, 3*embedding_size) || (n, seq_len, 3*head_size*n_heads)
        query, key, value = qkv.chunk(3, dim=2)  # each shape = (n, seq_len, embedding_size)

        query = query.view(n, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        key = key.view(n, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        value = value.view(n, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len).bool().to(device)
        attention_scaled_logits = query @ key.transpose(-2, -1) / self.head_size**0.5  # (n, n_heads, seq_len, seq_len)
        attention_scaled_logits = attention_scaled_logits.masked_fill(causal_mask, float('-inf'))
        attention = F.softmax(attention_scaled_logits, dim=-1)

        # x.shape => (n, n_heads, seq_len, head_size)
        x = attention @ value
        
        # x.shape => (n, seq_len, embedding_size)
        x = x.transpose(1, 2).contiguous().view(n, seq_len, embedding_size)

        # x.shape => (n, seq_len, embedding_size)
        x = self.linear_reproject(x)
        return x


class NonlinearReprojection(nn.Module):
    """
    I mean come on, it's just a 2-layer MLP

    embedding_size * 4 comes from the paper (2nd paragraph of section 2.1)
    https://arxiv.org/pdf/2005.14165v4.pdf
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size * 4

        self.fc1 = nn.Linear(embedding_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, embedding_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    

class SubBlock(nn.Module):
    """
    ith layer for a Small Language Model modelled after GPT3

    https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    Check '2.3 Model' for architecture details

    https://arxiv.org/pdf/2005.14165v4.pdf
    Check table 2.1 for size details
    """
    def __init__(self, embedding_size, n_heads):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads

        self.pre_norm = nn.LayerNorm(self.embedding_size, bias=False, eps=1e-6)
        self.msa = MultiHeadSelfAttention(self.n_heads, self.embedding_size)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_size, bias=False, eps=1e-6)
        self.nonlinear_reprojection = NonlinearReprojection(self.embedding_size)
    
    def forward(self, x):
        # residual/skip connections for all steps in the subblock
        x = x + self.msa(self.pre_norm(x))
        x = x + self.nonlinear_reprojection(self.layer_norm_2(x))
        return x


class SmallLM(nn.Module):
    """
    This model is modelled of the GPT2 model (which should be very similar to GPT3)
    """
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            context_size=context_size,
            n_heads=8,
            head_size=64,
            n_layers=6,
            use_dropout=False
        ):
            """
            Initialize the Small Language Model

            :param tokenizer (AutoTokenizer): The trained huggingface tokenizer
            :param embedding_size (int): The size of the embedding
            :param context_size (int): The maximum size of the context window
            :param n_heads (int): The number of heads in the MultiHeadSelfAttention
            :param n_layers (int): The number of layers of MSAs
            """
            super().__init__()
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.vocab_size
            self.context_size = context_size
            self.embedding_size = n_heads * head_size
            self.n_heads = n_heads
            self.head_size = head_size

            # Create token and position embeddings
            self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
            self.position_embedding = nn.Embedding(self.context_size, self.embedding_size)

            # Optional dropout
            if use_dropout:
                self.dropout = nn.Dropout(0.1)

            # Stack of subblocks
            self.subblocks = nn.ModuleList([
                SubBlock(self.embedding_size, self.n_heads)
                for _ in range(n_layers)
            ])

            self.ln_no_bias = nn.LayerNorm(self.embedding_size, bias=False, eps=1e-6)

            self.output_projection = nn.Linear(self.embedding_size, self.vocab_size, bias=False)

    def forward(self, input_token_ids, target_token_ids=None):
        """
        Unlike typical forward methods, this outputs both the logits and the loss

        :param token_ids: torch.Tensor of shape (batch_size, seq_len)
        :param target_token_ids: torch.Tensor of shape (batch_size, seq_len)
        """
        _, seq_len = input_token_ids.shape
        if seq_len > self.context_size:
            raise AssertionError(f'Expected sequence length of {self.context_size}, got {seq_len}')
        
        pos_i = torch.arange(seq_len, dtype=torch.long).to(device)

        # Produce input data
        token_embeddings = self.token_embedding(input_token_ids)  # (batch_size, seq_len, embedding_size)
        token_embeddings /= math.sqrt(self.embedding_size)
        
        position_embeddings = self.position_embedding(pos_i)  # (seq_len, embedding_size)
        x = token_embeddings + position_embeddings  # (batch_size, seq_len, embedding_size)

        # Optional dropout
        if hasattr(self, 'dropout'):
            x = self.dropout(x)

        # Actual forward pass
        for subblock in self.subblocks:
            x = subblock(x)
        x = self.ln_no_bias(x)
        
        if target_token_ids is None:
            # This would only be used for generation/inference time
            # return self.output_projection(x[:, -1, :])  # or x[:, [-1], :]
            return self.output_projection(x)
        else:
            # This would only be used for training
            logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)
            logits = logits.view(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
            target_token_ids = target_token_ids.view(-1)  # (batch_size * seq_len,)
            loss = F.cross_entropy(
                logits,
                target_token_ids,
                ignore_index=self.tokenizer.pad_token_id
            )
            return logits, loss

    @torch.inference_mode()             
    def generate(self, input_string: str, temp=1.0, max_length=None, deterministic=False):
        if max_length is None:
            max_length = self.context_size
        else:
            max_length = min(max_length, self.context_size)

        encoded = self.tokenizer(input_string, padding='max_length', return_tensors='pt')
        tokens, attention_mask = encoded['input_ids'].to(device), encoded['attention_mask'].to(device)

        # Generation loop
        print(input_string, end='')
        new_index = attention_mask.squeeze().sum().item()-1
        last_token = tokens[:, new_index]
        while last_token != self.tokenizer.eos_token_id and new_index < max_length-1:
            output = self.forward(tokens)
            
            # Generate a new token
            if deterministic:
                output = output.argmax(dim=2)
            else:
                output = F.softmax(output / temp, dim=2).squeeze()
                output = torch.multinomial(output, num_samples=1).squeeze()

            # Get the newly generated token
            new_token = output[new_index].item()

            # Update tokens and attention_mask with the new token and increment new_index
            tokens[:, new_index+1] = new_token
            new_index += 1
            last_token = new_token

            print(
                self.tokenizer.decode(new_token),
                end=''  # Continue on the same line
            )
        print('\n')

        return self.tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

    @property
    def n_params(self):
        n = sum(param.numel() for param in self.parameters())
        n -= self.position_embedding.weight.numel()
        return n
    
    def __repr__(self):
        representation = super().__repr__()
        representation += f'\n[[Embedding Size (d_model): {self.embedding_size:,}]]'
        representation += f'\n[[Context Size: {self.context_size:,}]]'
        representation += f'\n[[Vocab Size: {self.vocab_size:,}]]'
        representation += f'\n[[n_params: {self.n_params:,}]]'
        representation += f'\n[[n_layers: {len(self.subblocks)} | n_heads: {self.n_heads:,} | d_head: {self.head_size:,}]]'
        return representation
