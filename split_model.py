import pytest
import torch
import torch.nn as nn
from model import LayerNorm, MLP, GPT, GPTConfig, CausalSelfAttention, Block
from copy import deepcopy
import inspect
import math
import torch.nn.functional as F

class SplitAttention(nn.Module):
    """
    SplitAttention is a wrapper around CausalSelfAttention that splits the key, query, and value projections.
    This allows us to apply separate weight decay to the key and query projections.
    """
    def __init__(self, csa : CausalSelfAttention):
        super().__init__()

        # Check if csa.c_attn has a bias
        has_bias = csa.c_attn.bias is not None

        # key, query, value projections split into self.w_k, self.w_q, and self.w_v
        self.w_q = nn.Linear(csa.n_embd, csa.n_embd, bias=has_bias)
        self.w_k = nn.Linear(csa.n_embd, csa.n_embd, bias=has_bias)
        self.w_v = nn.Linear(csa.n_embd, csa.n_embd, bias=has_bias)

        # Copying weights from the original c_attn layer to self.w_q, self.w_k, and self.w_v
        self.w_q.weight.data.copy_(csa.c_attn.weight.data[:csa.n_embd, :])
        self.w_k.weight.data.copy_(csa.c_attn.weight.data[csa.n_embd:2*csa.n_embd, :])
        self.w_v.weight.data.copy_(csa.c_attn.weight.data[2*csa.n_embd:, :])
        
        # Copying bias from the original c_attn layer if it exists
        if has_bias:
            self.w_q.bias.data.copy_(csa.c_attn.bias.data[:csa.n_embd])
            self.w_k.bias.data.copy_(csa.c_attn.bias.data[csa.n_embd:2*csa.n_embd])
            self.w_v.bias.data.copy_(csa.c_attn.bias.data[2*csa.n_embd:])

        # The rest is the same as CausalSelfAttention, just copying
        
        # output projection
        self.c_proj = deepcopy(csa.c_proj)
        # regularization
        self.attn_dropout = deepcopy(csa.attn_dropout)
        self.resid_dropout = deepcopy(csa.resid_dropout)
        self.n_head = csa.n_head
        self.n_embd = csa.n_embd
        self.dropout = csa.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # get block size from CausalSelfAttention
            block_size = csa.bias.size(-1)
            # causal mask to ensure that attention is only applied to the left in the input sequence  
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.w_k(x)
        q = self.w_q(x)
        v = self.w_v(x)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def get_average_magnitude(self):
        """
        Calculate and return the magnitudes of all heads.
        This is equivalent to the beta (inverse temperature) in a hopfield layer.
        The magnitude of attention is the norm of W_k^T W_q, per head.
        Splitting it into seperate heads is too much of a hassle, for now, so we just take it over all heads.
        TODO: Split it into seperate heads, look how TransformerLens does it.
        """

        # Find the product of W_Q and W_K^T
        W_Q_W_KT = torch.einsum('ij, kj -> ik', self.w_q.weight, self.w_k.weight)
        assert W_Q_W_KT.shape == (self.n_embd, self.n_embd)

        # Compute Frobenius norm
        att_magnitude = torch.norm(W_Q_W_KT)

        # return magnitude, watch out, it's a tensor
        return att_magnitude


class SplitGPTWrapper():
    """
    SplitGPTWrapper is a wrapper around GPT that replaces all CausalSelfAttention layers with SplitAttention.
    """
    def __init__(self, gpt_instance : GPT):
        self.gpt = gpt_instance

        # Replace all CausalSelfAttention layers with SplitAttention
        for block in self.gpt.transformer.h:
            block.attn = SplitAttention(block.attn)

    def __getattr__(self, name):
        # Delegate calls to non-overridden methods to the GPT instance
        return getattr(self.gpt, name)
    
    def __call__(self, *args, **kwargs):
        # Delegate the call to the GPT instance's forward method
        return self.gpt(*args, **kwargs)

    def configure_optimizers(self, weight_decay, qk_weight_decay, learning_rate, betas, device_type):
        # Collect parameters, applying separate weight decay for w_k and w_q in SplitAttention
        decay_params = []
        nodecay_params = []
        qk_decay_params = []
        for name, param in self.named_parameters():
            if 'w_k.weight' in name or 'w_q.weight' in name:
                qk_decay_params.append(param)
            elif param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        # Define optimizer groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': qk_decay_params, 'weight_decay': qk_weight_decay}
        ]

        # Create AdamW optimizer with fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer
    
    def get_average_beta(self):
        """
        Calculate and return the magnitudes of all heads.
        This is equivalent to the beta (inverse temperature) in a hopfield layer.
        The magnitude of attention is the norm of W_k^T W_q, per head.
        Splitting it into seperate heads is too much of a hassle, for now, so we just take it over all heads.
        TODO: Split it into seperate heads, look how TransformerLens does it.
        TODO: Parallelize this over all heads.
        """
            
        # Get all SplitAttention layers
        split_attentions = []
        for block in self.gpt.transformer.h:
            split_attentions.append(block.attn)
    
        # Calculate and return the average magnitude of all SplitAttention layers
        return sum([split_attention.get_average_magnitude() for split_attention in split_attentions]) / len(split_attentions)